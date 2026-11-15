from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace
from typing import Collection

import numpy as np
from ortools.sat.python import cp_model

from src.problem import Problem
from src.solution import Route, Solution

# CP-SAT needs integer coefficients: costs are multiplied by this and rounded.
# Smaller coefficients propagate better. Raise it if you need more precision.
SCALE = 1000

BIG = 1 << 60


class Timeout(RuntimeError):
    ''' The search stopped at the time limit without a solution and without proving that none exists. '''


def min_chain_times(times: np.ndarray, service: np.ndarray) -> np.ndarray:
    ''' chain[i, j]: lower bound of (service start at j) - (service start at i) when j is visited after i.

    Shortest path over weights service[i] + times[i, j]; valid without the triangle inequality.
    '''

    chain = service[:, None] + times
    np.fill_diagonal(chain, 0)

    for m in range(chain.shape[0]):
        np.minimum(chain, chain[:, m][:, None] + chain[m, :][None, :], out=chain)

    return chain


def _prepare(problem: Problem) -> SimpleNamespace:
    ''' Numpy view of the instance restricted to the depot, pickups and deliveries (the matrices may be bigger).

    All time data is divided by the greatest common divisor of every time value: the schedule can always be
    expressed in multiples of it, and smaller domains make the search faster.
    '''

    n = problem.requests
    size = 2 * n + 1
    nodes = problem.nodes[:size]

    # The int64 conversion below truncates: the data must already be integral (see `Problem`)
    assert np.array_equal(problem.times, np.floor(problem.times)), 'times must be integral'
    assert all(
        float(node.ready_time).is_integer() and float(node.due_time).is_integer()
        and float(node.service_time).is_integer()
        for node in nodes
    ), 'time windows and service times must be integral'

    ready = np.array([node.ready_time for node in nodes], dtype=np.int64)
    due_raw = np.array([node.due_time for node in nodes], dtype=np.int64)
    due = np.maximum(ready, due_raw)
    window_ok = ready <= due_raw
    service = np.array([node.service_time for node in nodes], dtype=np.int64)
    demand = np.array([node.demand for node in nodes], dtype=np.int64)
    demand[0] = 0

    times = np.asarray(problem.times, dtype=np.int64)[:size, :size].copy()
    max_ride = int(problem.max_ride_time)
    max_duration = int(problem.max_duration_route)

    g = int(np.gcd.reduce(np.concatenate([ready, due, service, times.ravel()])))

    if g > 1:
        ready //= g
        due //= g
        service //= g
        times //= g
        max_ride //= g
        max_duration //= g

    return SimpleNamespace(
        ready=ready,
        due=due,
        service=service,
        demand=demand,
        times=times,
        max_ride=max_ride,
        max_duration=max_duration,
        request_ok=window_ok[1:n + 1] & window_ok[n + 1:],
    )


def static_arcs(
    n: int,
    service: np.ndarray,
    demand: np.ndarray,
    times: np.ndarray,
    capacity: int,
    max_ride: int,
    chain: np.ndarray,
    need_load: bool = True,
) -> np.ndarray:
    ''' Arcs that no feasible route can use for structural, load or ride-time reasons (no time windows). '''

    size = 2 * n + 1
    ok = np.ones((size, size), dtype=bool)
    ks = np.arange(1, n + 1)

    # Structure: no depot -> delivery, pickup -> depot, delivery of k -> pickup of k
    ok[0, n + 1:] = False
    ok[1:n + 1, 0] = False
    ok[ks + n, ks] = False

    # Load: load[j] = load[i] + demand[j] must fit the domain of load[j]
    if need_load:
        lo = np.maximum(0, demand)
        hi = np.minimum(capacity, capacity + demand)
        lo[0] = hi[0] = 0
        load_ok = (lo[:, None] + demand[None, :] <= hi[None, :]) & (hi[:, None] + demand[None, :] >= lo[None, :])
        load_ok[:, 0] = True  # no load constraint towards the depot
        ok &= load_ok

    # Ride time: a node right after pickup k (or right before delivery k + n) makes k ride at least this long
    after = times[ks, :] + chain[:, ks + n].T  # arc k -> j
    before = chain[ks, :] - service[ks][:, None] + service[None, :] + times[:, ks + n].T  # arc j -> k + n
    ok[ks, :] &= after <= max_ride
    ok[:, ks + n] &= (before <= max_ride).T

    np.fill_diagonal(ok, False)

    return ok


@lru_cache(maxsize=None)
def feasible_arcs(problem: Problem) -> np.ndarray:
    ''' Arcs that can belong to some feasible route of the instance (necessary conditions). '''

    size = 2 * problem.requests + 1
    times = problem.times[:size, :size]
    service = np.array([node.service_time for node in problem.nodes])
    ready = np.array([node.ready_time for node in problem.nodes])
    due = np.array([node.due_time for node in problem.nodes])
    demand = np.array([node.demand for node in problem.nodes])
    chain = min_chain_times(times, service)

    arcs = static_arcs(problem.requests, service, demand, times, problem.capacity, problem.max_ride_time, chain)
    arcs &= ready[:, None] + service[:, None] + times <= due[None, :]  # time windows

    # Route duration: a route using (i, j) lasts at least depot -> i -> j -> depot
    duration = chain[0, :, None] + (service[:, None] + times) + chain[:, 0][None, :]
    arcs &= duration <= problem.max_duration_route

    return arcs


def _restrict(mask: np.ndarray, n: int, request_ok: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ''' Iteratively drops requests whose nodes lose all incoming or outgoing arcs (fixpoint). '''

    alive = request_ok.copy()

    while True:
        keep = np.concatenate(([True], alive, alive))
        sub = mask & keep[:, None] & keep[None, :]
        reachable = sub.any(axis=0) & sub.any(axis=1)
        new_alive = alive & reachable[1:n + 1] & reachable[n + 1:]

        if np.array_equal(new_alive, alive):
            return sub, [int(r) + 1 for r in np.flatnonzero(alive)]

        alive = new_alive


def _tighten_windows(
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    times: np.ndarray,
    arcs: np.ndarray,
    n: int,
    max_ride: int,
    chain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Bound propagation over the allowed arcs: earliest / latest arrival at every node.

    A node served by the private fleet has a predecessor and a successor through allowed arcs, so its arrival
    is at least the best (earliest predecessor + service + travel) and at most the best (latest successor
    - service - travel). Precedence and ride time between pickup and delivery tighten them further.
    '''

    ks = np.arange(1, n + 1)
    ds = ks + n
    lb, ub = ready.copy(), due.copy()

    for _ in range(100):
        previous_lb, previous_ub = lb.copy(), ub.copy()

        forward = np.where(arcs, lb[:, None] + service[:, None] + times, BIG)
        lb[1:] = np.maximum(lb[1:], forward[:, 1:].min(axis=0))

        backward = np.where(arcs, ub[None, :] - service[:, None] - times, -BIG)
        ub[1:] = np.minimum(ub[1:], backward[1:, :].max(axis=1))

        lb[ds] = np.maximum(lb[ds], lb[ks] + chain[ks, ds])
        ub[ks] = np.minimum(ub[ks], ub[ds] - chain[ks, ds])
        lb[ks] = np.maximum(lb[ks], lb[ds] - service[ks] - max_ride)
        ub[ds] = np.minimum(ub[ds], ub[ks] + service[ks] + max_ride)

        np.minimum(lb, BIG, out=lb)
        np.maximum(ub, -BIG, out=ub)

        if np.array_equal(lb, previous_lb) and np.array_equal(ub, previous_ub):
            break

    return lb, ub


@lru_cache(maxsize=4)
def _context(problem: Problem) -> SimpleNamespace:
    ''' Everything `solve` derives from the instance alone (it never changes between LNS calls).

    The arrays are shared between calls: `solve` must not modify them in place.
    '''

    n = problem.requests
    data = _prepare(problem)
    capacity = problem.capacity
    need_load = int(np.maximum(0, data.demand[1:n + 1]).sum()) > capacity
    duration_binding = int(data.due[0] - data.ready[0]) > data.max_duration
    chain = min_chain_times(data.times, data.service)
    static = static_arcs(
        n, data.service, data.demand, data.times, capacity, data.max_ride, chain, need_load
    )

    return SimpleNamespace(
        data=data,
        need_load=need_load,
        duration_binding=duration_binding,
        chain=chain,
        static=static,
        Tl=data.times.tolist(),
        Cl=chain.tolist(),
        ready_l=data.ready.tolist(),
        due_l=data.due.tolist(),
        service_l=data.service.tolist(),
        demand_l=data.demand.tolist(),
    )


def solve(
    problem: Problem,
    initial_solution: Solution,
    matrices: list[np.ndarray],
    time_limit: float = 120.0,
    max_integrated: int | None = None,
    workers: int = 0,
    gap: float = 0.0,
    log: bool = False,
    upper_bound_cut: bool = True,
    params: dict | None = None,
    seed: int = 0,
    deterministic: bool = True,
    strict: bool = False,
    hint: bool = True,
    fixed_requests: Collection[int] = (),
) -> Solution | None:
    ''' Exact (within the restricted arcs) CP-SAT model for the IDARP, warm-started with `initial_solution`.

    Each request is served in exactly one way: by one vehicle of the private fleet, by the common carrier
    or by the bus integration (the last two only if feasible, with fixed costs). Vehicle v can only use the
    arcs allowed in `matrices[v]` (see `neighborhoods.expand`), and each vehicle route is a circuit.
    The objective is the total cost: private_per_km * distance + carrier + bus integration.

    The matrices must come from `initial_solution` (used routes first), so that its routes are in the model.

    Extra parameters: `workers` (0 = all cores), `gap` (relative gap to stop early, e.g. 0.01), `log`,
    `upper_bound_cut` (adds "cost <= cost of the initial solution" to the model) and `params` (any
    CpSolverParameters field, e.g. {'linearization_level': 1}).

    Reproducibility: with `deterministic=True` (default) the parallel search is interleaved deterministically,
    the limit is in deterministic time (not wall clock) and `seed` fixes the random choices, so the same
    input gives the same solution with the same OR-Tools version. The running time may then differ from
    `time_limit`. With `deterministic=False` the search is faster but different on every run.

    `fixed_requests` are requests served by routes that stay outside the model (the untouched routes of a
    route-subset LNS): they get no variables and no alternative here, and `initial_solution` must not contain
    them. `matrices` then only describe the routes (and idle vehicles) that the model may change.

    With `strict=True` the solver is used to look for an improvement of `initial_solution`. The model is warm
    started with the initial solution (`hint`) and cut at "cost <= initial cost", so the search is anytime: it
    starts from the incumbent and improves it, instead of having to find from scratch a solution that is
    already better than the incumbent. Afterwards:
      - the best solution is strictly cheaper than the initial one: it is returned;
      - the solver proves that nothing is cheaper in these arcs (OPTIMAL at the initial cost, or
        INFEASIBLE): returns `None` (the neighborhood is exhausted);
      - the limit is reached without finding anything cheaper: raises `Timeout`.
    '''

    n = problem.requests
    size = 2 * n + 1
    requests = range(1, n + 1)
    request_of = lambda i: i if i <= n else i - n

    fixed = frozenset(fixed_requests)
    context = _context(problem)
    data, chain, static = context.data, context.chain, context.static
    ready, due, service, demand, T = data.ready, data.due, data.service, data.demand, data.times
    max_ride, max_duration = data.max_ride, data.max_duration
    capacity = problem.capacity

    # Constraints that are redundant for this instance are not generated

    need_load = context.need_load
    duration_binding = context.duration_binding

    # Allowed arcs per vehicle (distinct matrices are processed once)

    raw_index: dict[bytes, int] = {}
    raw_masks: list[np.ndarray] = []
    raw_of: list[int] = []

    for matrix in matrices:
        mask = matrix[:size, :size] >= 0
        key = mask.tobytes()

        if key not in raw_index:
            raw_index[key] = len(raw_masks)
            raw_masks.append(mask)

        raw_of.append(raw_index[key])

    def restrict_all(lb: np.ndarray, ub: np.ndarray, alive: np.ndarray):
        arcs = static & (lb[:, None] + service[:, None] + T <= ub[None, :])
        return [_restrict(mask & arcs, n, alive) for mask in raw_masks]

    # Fixpoint: arcs -> tighter time windows -> fewer arcs / requests -> ...

    lb, ub, alive = ready.copy(), due.copy(), data.request_ok.copy()

    for k in fixed:
        alive[k - 1] = False  # served by an untouched route: never part of the model


    for _ in range(10):
        results = restrict_all(lb, ub, alive)
        union = np.zeros((size, size), dtype=bool)

        for sub, _elig in results:
            union |= sub

        new_lb, new_ub = _tighten_windows(ready, due, service, T, union, n, max_ride, chain)
        new_alive = alive & (new_lb[1:n + 1] <= new_ub[1:n + 1]) & (new_lb[n + 1:] <= new_ub[n + 1:])

        stable = (
            np.array_equal(new_alive, alive) and np.array_equal(new_lb, lb) and np.array_equal(new_ub, ub)
        )
        lb, ub, alive = new_lb, new_ub, new_alive

        if stable:
            break

    results = restrict_all(lb, ub, alive)

    # Plain Python ints for the model building loops

    Tl, Cl = context.Tl, context.Cl
    ready_l, due_l = context.ready_l, context.due_l
    service_l, demand_l = context.service_l, context.demand_l

    lo, hi = ready.tolist(), due.tolist()  # domain of the arrival time of each node
    for i in range(1, size):
        if alive[request_of(i) - 1]:
            lo[i], hi[i] = int(lb[i]), int(ub[i])

    # Groups of interchangeable vehicles (same allowed arcs); surplus vehicles of a group are dropped,
    # since a group can never use more vehicles than it has eligible requests

    group_index: dict[bytes, int] = {}
    group_of: list[int] = []
    group_members: list[list[int]] = []
    group_eligible: list[list[int]] = []
    sub_keys = [sub.tobytes() for sub, _elig in results]

    for v in range(len(matrices)):
        u = raw_of[v]

        if sub_keys[u] not in group_index:
            group_index[sub_keys[u]] = len(group_members)
            group_members.append([])
            group_eligible.append(results[u][1])

        group_of.append(group_index[sub_keys[u]])
        group_members[group_of[-1]].append(v)

    kept_members = [members[:len(group_eligible[g])] for g, members in enumerate(group_members)]
    vehicles = sorted(v for members in kept_members for v in members)
    mask_of = {v: results[raw_of[v]][0] for v in vehicles}
    eligible = {v: group_eligible[group_of[v]] for v in vehicles}

    # Only requests that some vehicle can serve need time / load variables (with clustered matrices, as
    # produced by `neighborhoods.expand`, most requests are outside every cluster and stay with carrier / bus)
    active = sorted({r for v in vehicles for r in eligible[v]})
    active_nodes = [i for r in active for i in (r, r + n)]

    model = cp_model.CpModel()

    # Variables

    y = {(r, v): model.NewBoolVar(f'y_{r}_{v}') for v in vehicles for r in eligible[v]}
    used = {v: model.NewBoolVar(f'used_{v}') for v in vehicles}

    x: dict[tuple[int, int, int], cp_model.IntVar] = {}
    arcs_of: dict[int, list[tuple[int, int, cp_model.IntVar]]] = {v: [] for v in vehicles}
    circuit_arcs: dict[int, list[tuple[int, int, cp_model.IntVar]]] = {v: [] for v in vehicles}
    visited_of: dict[int, list[int]] = {}

    for v in vehicles:
        visited = [0] + [i for r in eligible[v] for i in (r, r + n)]
        visited_of[v] = visited

        idx = np.array(visited)
        sub = mask_of[v][np.ix_(idx, idx)]

        for p, q in np.argwhere(sub):
            p, q = int(p), int(q)
            i, j = visited[p], visited[q]
            var = model.NewBoolVar(f'x_{i}_{j}_{v}')
            x[i, j, v] = var
            arcs_of[v].append((i, j, var))
            circuit_arcs[v].append((p, q, var))  # circuit index == position in `visited`

    free = [k for k in requests if k not in fixed]
    carrier = {k: model.NewBoolVar(f'carrier_{k}') for k in free if problem.carrier_feasible(k)}
    bus = {k: model.NewBoolVar(f'bus_{k}') for k in free if problem.integration_feasible(k)}
    private = {k: model.NewBoolVar(f'private_{k}') for k in active}

    a = {i: model.NewIntVar(lo[i], max(lo[i], hi[i]), f'a_{i}') for i in active_nodes}
    load = {
        i: model.NewIntVar(
            max(0, demand_l[i]),
            min(capacity, capacity + demand_l[i]),
            f'load_{i}',
        )
        for i in active_nodes
    } if need_load else {}

    if duration_binding:
        start = {v: model.NewIntVar(ready_l[0], due_l[0], f'start_{v}') for v in vehicles}
        end = {v: model.NewIntVar(ready_l[0], due_l[0], f'end_{v}') for v in vehicles}

    # Each request is served in exactly one way

    for k in free:
        options = []
        if k in private:
            model.Add(private[k] == sum(y[k, v] for v in vehicles if (k, v) in y))
            options.append(private[k])
        if k in carrier:
            options.append(carrier[k])
        if k in bus:
            options.append(bus[k])

        model.AddExactlyOne(options)

    if max_integrated is not None:
        model.Add(sum(bus.values()) <= max_integrated)

    # One circuit per vehicle: nodes not visited by the vehicle take a self-loop

    for v in vehicles:
        visited = visited_of[v]

        circuit = [(0, 0, used[v].Not())]
        circuit += [(p, p, y[request_of(visited[p]), v].Not()) for p in range(1, len(visited))]
        circuit += circuit_arcs[v]

        model.AddCircuit(circuit)

    # Symmetry breaking inside each group: used vehicles first, ordered by their smallest request

    for members in kept_members:
        for v, w in zip(members, members[1:]):
            model.Add(used[v] >= used[w])

        if len(members) > 1:
            first = {}

            for v in members:
                first[v] = model.NewIntVar(1, n + 1, f'first_{v}')
                model.AddMinEquality(first[v], [(n + 1) - (n + 1 - r) * y[r, v] for r in eligible[v]])

            for v, w in zip(members, members[1:]):
                model.Add(first[v] < first[w]).OnlyEnforceIf(used[w])

    # Time and load propagation along the arcs (constraints already implied by the domains are skipped)

    for (i, j, v), var in x.items():
        travel = Tl[i][j]

        if i == 0:
            if duration_binding:
                model.Add(a[j] >= start[v] + service_l[0] + travel).OnlyEnforceIf(var)
            elif ready_l[0] + service_l[0] + travel > lo[j]:
                model.Add(a[j] >= ready_l[0] + service_l[0] + travel).OnlyEnforceIf(var)

            if need_load:
                model.Add(load[j] == demand_l[j]).OnlyEnforceIf(var)
        elif j == 0:
            if duration_binding:
                model.Add(end[v] >= a[i] + service_l[i] + travel).OnlyEnforceIf(var)
            elif due_l[0] - service_l[i] - travel < hi[i]:
                model.Add(a[i] <= due_l[0] - service_l[i] - travel).OnlyEnforceIf(var)
        else:
            if lo[j] < hi[i] + service_l[i] + travel:
                model.Add(a[j] >= a[i] + service_l[i] + travel).OnlyEnforceIf(var)

            if need_load:
                model.Add(load[j] == load[i] + demand_l[j]).OnlyEnforceIf(var)

    # Precedence and ride time, only for requests served by the private fleet

    for k in requests:
        d = k + n

        if k not in private:
            continue  # no vehicle can serve it privately

        if hi[k] + Cl[k][d] > lo[d]:
            model.Add(a[d] >= a[k] + Cl[k][d]).OnlyEnforceIf(private[k])

        if hi[d] - lo[k] - service_l[k] > max_ride:
            model.Add(a[d] - a[k] - service_l[k] <= max_ride).OnlyEnforceIf(private[k])

    if duration_binding:
        for v in vehicles:
            model.Add(end[v] - start[v] <= max_duration)

    # Objective: total cost, in integer units of 1 / SCALE

    arc_cost = lambda i, j: round(problem.private_cost(i, j) * SCALE)
    carrier_cost = { k: round(problem.carrier_cost(k) * SCALE) for k in carrier }
    bus_cost = { k: round(problem.integration_cost(k) * SCALE) for k in bus }

    terms = [(var, arc_cost(i, j)) for (i, j, _), var in x.items()]
    terms += [(var, carrier_cost[k]) for k, var in carrier.items()]
    terms += [(var, bus_cost[k]) for k, var in bus.items()]
    terms = [(var, coef) for var, coef in terms if coef != 0]

    objective = model.NewIntVar(0, sum(coef for _, coef in terms), 'objective')
    model.Add(objective == cp_model.LinearExpr.WeightedSum([t[0] for t in terms], [t[1] for t in terms]))
    model.Minimize(objective)

    # Hint: the complete initial solution (its routes are the first vehicles, as in `neighborhoods.expand`).
    # Routes of interchangeable vehicles are sorted by their smallest request to respect the symmetry breaking.

    hint_routes = [route for route in initial_solution.routes if route.nodes][:len(matrices)]
    routes_by_group: dict[int, list[list[int]]] = {}

    for v, route in enumerate(hint_routes):
        routes_by_group.setdefault(group_of[v], []).append(list(route.nodes))

    hint_assign: dict[int, list[int]] = {}

    for g, group_routes in routes_by_group.items():
        group_routes.sort(key=lambda nodes: min(request_of(i) for i in nodes))

        for v, nodes in zip(group_members[g], group_routes):
            hint_assign[v] = nodes

    hint_arcs = {
        (u, w, v)
        for v, nodes in hint_assign.items()
        for u, w in zip([0, *nodes], [*nodes, 0])
    }
    hint_requests = {(request_of(i), v) for v, nodes in hint_assign.items() for i in nodes}
    hint_private = {k for k, _ in hint_requests}

    if hint:
        a_hint = {i: lo[i] for i in range(1, size)}
        load_hint = {i: max(0, demand_l[i]) for i in range(1, size)}
        end_hint = {v: ready_l[0] for v in vehicles}

        for v, nodes in hint_assign.items():
            # Earliest schedule along the route (clipped to the domains)
            leave = ready_l[0] + service_l[0]
            previous = 0
            current_load = 0

            for node in nodes:
                a_hint[node] = min(hi[node], max(lo[node], leave + Tl[previous][node]))
                current_load += demand_l[node]
                load_hint[node] = min(
                    max(current_load, max(0, demand_l[node])), min(capacity, capacity + demand_l[node])
                )
                leave = a_hint[node] + service_l[node]
                previous = node

            end_hint[v] = min(due_l[0], max(ready_l[0], leave + Tl[previous][0]))

        for key, var in x.items():
            model.AddHint(var, int(key in hint_arcs))
        for key, var in y.items():
            model.AddHint(var, int(key in hint_requests))
        for k, var in private.items():
            model.AddHint(var, int(k in hint_private))
        for v in vehicles:
            model.AddHint(used[v], int(v in hint_assign))
            if duration_binding:
                model.AddHint(start[v], ready_l[0])
                model.AddHint(end[v], end_hint[v])
        for i in a:
            model.AddHint(a[i], a_hint[i])
        for i in load:
            model.AddHint(load[i], load_hint[i])
        for k, var in carrier.items():
            model.AddHint(var, int(k in initial_solution.carrier_requests))
        for k, var in bus.items():
            model.AddHint(var, int(k in initial_solution.integrated_requests))

    # Upper bound: the initial solution is feasible, so the optimum cannot cost more than it

    initial_cost = None

    if upper_bound_cut or strict:
        carrier_set = set(initial_solution.carrier_requests)
        bus_set = set(initial_solution.integrated_requests)
        complete = (
            all(key in x for key in hint_arcs)
            and all(key in y for key in hint_requests)
            and carrier_set <= set(carrier)
            and bus_set <= set(bus)
            and len(hint_private) + len(carrier_set) + len(bus_set) == len(free)
            and hint_private | carrier_set | bus_set == set(free)
        )

        if complete:
            initial_cost = (
                sum(arc_cost(i, j) for i, j, _ in hint_arcs)
                + sum(carrier_cost[k] for k in carrier_set)
                + sum(bus_cost[k] for k in bus_set)
            )
            model.Add(objective <= initial_cost)
        elif strict:
            raise RuntimeError('The initial solution is not in the model, so no strict improvement can be proved')

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = workers or os.cpu_count() or 8

    if deterministic:
        # Wall-clock limits make the stopping point depend on machine load, so the limit is deterministic time;
        # the wall-clock limit is only a safety net
        solver.parameters.interleave_search = True
        solver.parameters.max_deterministic_time = time_limit
        solver.parameters.max_time_in_seconds = 10 * time_limit
    else:
        solver.parameters.max_time_in_seconds = time_limit

    solver.parameters.relative_gap_limit = gap
    solver.parameters.log_search_progress = log
    # `repair_hint` is off: it can abort the process ("Check failed: heuristics.fixed_search != nullptr")
    # when the repair model is unsatisfiable. It can be re-enabled with `params={'repair_hint': True}`.
    solver.parameters.hint_conflict_limit = 100
    solver.parameters.linearization_level = 2

    for name, value in (params or {}).items():
        setattr(solver.parameters, name, value)

    status = solver.Solve(model)

    if status == cp_model.INFEASIBLE and strict:
        return None  # nothing strictly better than the initial solution inside these arcs

    if status == cp_model.UNKNOWN:
        raise Timeout('Time limit reached without a solution')

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError('No solution found (the model is infeasible)')

    if strict and initial_cost is not None and round(solver.ObjectiveValue()) >= initial_cost:
        if status == cp_model.OPTIMAL:
            return None  # proved: nothing cheaper than the initial solution inside these arcs

        raise Timeout('Time limit reached without improving the initial solution')

    routes: list[Route] = []

    for v in vehicles:
        successor = {i: j for i, j, var in arcs_of[v] if solver.BooleanValue(var)}

        if 0 not in successor:
            continue

        route_nodes = []
        current = successor[0]

        while current != 0:
            route_nodes.append(current)
            current = successor[current]

        routes.append(Route(problem, route_nodes))

    return Solution(
        problem,
        routes,
        {k for k, var in carrier.items() if solver.BooleanValue(var)},
        {k for k, var in bus.items() if solver.BooleanValue(var)},
    )