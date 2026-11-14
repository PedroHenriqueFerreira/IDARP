from __future__ import annotations

import math
import os
import time

import numpy as np
from ortools.sat.python import cp_model

from src.problem import Problem
from src.route import Route
from src.solution import Solution

# CP-SAT needs integer coefficients: costs are multiplied by this and rounded.
# Lower than before (was 10_000): smaller coefficients propagate better. Raise it if you need more precision.
COST_SCALE = 1_000


def _static_arc_filter(problem: Problem) -> np.ndarray:
    ''' Boolean (N x N) matrix of arcs that can never be used, regardless of the vehicle, set to False.

    Removes arcs that are infeasible by time window, by load, or by structure (depot -> delivery,
    pickup -> depot, delivery of k -> pickup of k). Every removed arc is one that the model would
    have to discover as infeasible during the search, multiplied by the number of vehicles.
    '''

    n = problem.requests
    nodes = problem.nodes
    capacity = problem.capacity

    size = 2 * n + 1  # depot + pickups + deliveries (the matrices may contain extra nodes)
    times = np.asarray(problem.times, dtype=np.int64)[:size, :size]
    nodes = nodes[:size]
    ready = np.array([node.ready_time for node in nodes], dtype=np.int64)
    due = np.maximum(ready, np.array([node.due_time for node in nodes], dtype=np.int64))
    service = np.array([node.service_time for node in nodes], dtype=np.int64)
    demand = np.array([node.demand for node in nodes], dtype=np.int64)
    demand[0] = 0

    load_lo = np.maximum(0, demand)
    load_hi = np.minimum(capacity, capacity + demand)
    load_lo[0] = load_hi[0] = 0

    # Time: leaving i as early as possible must still reach j before its due time
    ok = (ready + service)[:, None] + times <= due[None, :]

    # Load: load[j] = load[i] + demand[j] must intersect the domain of load[j]
    ok &= (load_lo[:, None] + demand[None, :] <= load_hi[None, :]) & (
        load_hi[:, None] + demand[None, :] >= load_lo[None, :]
    )
    ok[:, 0] = (ready + service)[:, None][:, 0] + times[:, 0] <= due[0]  # no load constraint towards the depot

    # Structure
    ks = np.arange(1, n + 1)
    ok[0, n + 1:] = False       # depot -> delivery
    ok[1:n + 1, 0] = False      # pickup -> depot
    ok[ks + n, ks] = False      # delivery of k -> pickup of k
    np.fill_diagonal(ok, False)

    return ok


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


def solve(
    problem: Problem,
    initial_solution: Solution,
    matrices: list[np.ndarray],
    time_limit: float = 120.0,
    max_integrated: int | None = None,
    workers: int = 0,
    gap: float = 0.0,
    log: bool = False,
) -> Solution:
    ''' Exact (within the restricted arcs) CP-SAT model for the IDARP, warm-started with `initial_solution`.

    Each request is served in exactly one way: by one vehicle of the private fleet, by the common carrier
    or by the bus integration (the last two only if feasible, with fixed costs). Vehicle v can only use the
    arcs allowed in `matrices[v]` (see `expand_neighbors`), and each vehicle route is a circuit.
    The objective is the total cost: private_per_km * distance + carrier + bus integration.

    The matrices must come from `initial_solution` (used routes first), so that its routes are in the model.

    Extra parameters: `workers` (0 = all cores), `gap` (relative gap to stop early, e.g. 0.01) and `log`.
    '''

    n = problem.requests
    nodes = problem.nodes
    vehicles = range(len(matrices))
    requests = range(1, n + 1)
    request_of = lambda i: i if i <= n else i - n

    # Plain Python ints: much faster than indexing numpy arrays inside the loops
    T = np.asarray(problem.times, dtype=np.int64).tolist()
    D = np.asarray(problem.distances, dtype=np.int64).tolist()
    ready = [node.ready_time for node in nodes]
    due = [max(node.ready_time, node.due_time) for node in nodes]
    service = [node.service_time for node in nodes]
    demand = [node.demand for node in nodes]
    capacity = problem.capacity

    model = cp_model.CpModel()

    # Arcs and requests each vehicle can use (vectorized, with the static filters of `_static_arc_filter`)

    static_ok = _static_arc_filter(problem)
    request_ok = np.array([
        nodes[r].ready_time <= nodes[r].due_time and nodes[r + n].ready_time <= nodes[r + n].due_time
        for r in requests
    ])

    masks: list[np.ndarray] = []
    eligible: list[list[int]] = []

    for matrix in matrices:
        size = 2 * n + 1
        mask, elig = _restrict(static_ok & (matrix[:size, :size] >= 0), n, request_ok)
        masks.append(mask)
        eligible.append(elig)

    # Variables

    y = {(r, v): model.NewBoolVar(f'y_{r}_{v}') for v in vehicles for r in eligible[v]}
    used = [model.NewBoolVar(f'used_{v}') for v in vehicles]

    x: dict[tuple[int, int, int], cp_model.IntVar] = {}
    arcs_of: list[list[tuple[int, int, cp_model.IntVar]]] = [[] for _ in vehicles]
    circuit_arcs: list[list[tuple[int, int, cp_model.IntVar]]] = [[] for _ in vehicles]
    visited_of: list[list[int]] = []

    for v in vehicles:
        visited = [0] + [i for r in eligible[v] for i in (r, r + n)]
        visited_of.append(visited)

        idx = np.array(visited)
        sub = masks[v][np.ix_(idx, idx)]

        for p, q in np.argwhere(sub):
            i, j = visited[p], visited[q]
            var = model.NewBoolVar(f'x_{i}_{j}_{v}')
            x[i, j, v] = var
            arcs_of[v].append((i, j, var))
            circuit_arcs[v].append((int(p), int(q), var))  # circuit index == position in `visited`

    carrier = {k: model.NewBoolVar(f'carrier_{k}') for k in requests if problem.carrier_feasible(k)}
    bus = {k: model.NewBoolVar(f'bus_{k}') for k in requests if problem.integration_feasible(k)}
    private = {k: model.NewBoolVar(f'private_{k}') for k in requests}

    a = {i: model.NewIntVar(ready[i], due[i], f'a_{i}') for i in range(1, 2 * n + 1)}
    load = {
        i: model.NewIntVar(
            max(0, demand[i]),
            min(capacity, capacity + demand[i]),
            f'load_{i}',
        )
        for i in range(1, 2 * n + 1)
    }
    start = [model.NewIntVar(ready[0], due[0], f'start_{v}') for v in vehicles]
    end = [model.NewIntVar(ready[0], due[0], f'end_{v}') for v in vehicles]

    # Each request is served in exactly one way

    for k in requests:
        model.Add(private[k] == sum(y[k, v] for v in vehicles if (k, v) in y))

        options = [private[k]]
        if k in carrier:
            options.append(carrier[k])
        if k in bus:
            options.append(bus[k])

        model.AddExactlyOne(options)

    if max_integrated is not None:
        model.Add(sum(bus.values()) <= max_integrated)

    # One circuit per vehicle: nodes not visited by the vehicle take a self-loop

    for v in vehicles:
        if not eligible[v]:
            model.Add(used[v] == 0)
            continue

        visited = visited_of[v]

        circuit = [(0, 0, used[v].Not())]
        circuit += [(p, p, y[request_of(visited[p]), v].Not()) for p in range(1, len(visited))]
        circuit += circuit_arcs[v]

        model.AddCircuit(circuit)

        # A used vehicle serves at least one request
        model.AddBoolOr([y[r, v] for r in eligible[v]]).OnlyEnforceIf(used[v])

    # Symmetry breaking: consecutive vehicles with exactly the same allowed arcs are interchangeable

    for v in range(len(matrices) - 1):
        if np.array_equal(masks[v], masks[v + 1]):
            model.Add(used[v] >= used[v + 1])

    # Time and load propagation along the arcs

    for (i, j, v), var in x.items():
        travel = T[i][j]

        if i == 0:
            model.Add(a[j] >= start[v] + service[0] + travel).OnlyEnforceIf(var)
            model.Add(load[j] == demand[j]).OnlyEnforceIf(var)
        elif j == 0:
            model.Add(end[v] >= a[i] + service[i] + travel).OnlyEnforceIf(var)
        else:
            model.Add(a[j] >= a[i] + service[i] + travel).OnlyEnforceIf(var)
            model.Add(load[j] == load[i] + demand[j]).OnlyEnforceIf(var)

    # Precedence and ride time, only for requests served by the private fleet

    for k in requests:
        d = k + n

        model.Add(a[d] >= a[k] + service[k] + T[k][d]).OnlyEnforceIf(private[k])
        model.Add(a[d] - a[k] - service[k] <= problem.max_ride_time).OnlyEnforceIf(private[k])

    for v in vehicles:
        model.Add(end[v] - start[v] <= problem.max_duration_route)

    # Objective: total cost, in integer units of 1 / COST_SCALE

    unit = problem.costs.private_per_km / problem.scale
    terms = [(var, round(unit * D[i][j] * COST_SCALE)) for (i, j, _), var in x.items()]
    terms += [(var, round(problem.carrier_cost(k) * COST_SCALE)) for k, var in carrier.items()]
    terms += [(var, round(problem.integration_cost(k) * COST_SCALE)) for k, var in bus.items()]
    terms = [(var, coef) for var, coef in terms if coef != 0]

    model.Minimize(cp_model.LinearExpr.WeightedSum([t[0] for t in terms], [t[1] for t in terms]))

    # Hint: the complete initial solution (its routes are the first vehicles, as in `expand_neighbors`).
    # A partial hint (only arcs) often cannot be completed by the solver, so times and loads are hinted too.

    hint_routes = [route for route in initial_solution.routes if route.nodes][:len(matrices)]
    hint_arcs = {
        (u, w, v)
        for v, route in enumerate(hint_routes)
        for u, w in zip([0, *route.nodes], [*route.nodes, 0])
    }
    hint_requests = {(request_of(i), v) for v, route in enumerate(hint_routes) for i in route.nodes}
    hint_private = {k for k, _ in hint_requests}

    a_hint = {i: ready[i] for i in range(1, 2 * n + 1)}
    load_hint = {i: max(0, demand[i]) for i in range(1, 2 * n + 1)}
    start_hint = [ready[0]] * len(matrices)
    end_hint = [ready[0]] * len(matrices)

    for v, route in enumerate(hint_routes):
        # Earliest schedule along the route (clipped to the windows); `repair_hint` fixes small violations
        leave = ready[0] + service[0]
        previous = 0
        current_load = 0

        for node in route.nodes:
            a_hint[node] = min(due[node], max(ready[node], leave + T[previous][node]))
            current_load += demand[node]
            load_hint[node] = min(max(current_load, max(0, demand[node])), min(capacity, capacity + demand[node]))
            leave = a_hint[node] + service[node]
            previous = node

        end_hint[v] = min(due[0], max(ready[0], leave + T[previous][0]))

    for key, var in x.items():
        model.AddHint(var, int(key in hint_arcs))
    for key, var in y.items():
        model.AddHint(var, int(key in hint_requests))
    for k, var in private.items():
        model.AddHint(var, int(k in hint_private))
    for v in vehicles:
        model.AddHint(used[v], int(v < len(hint_routes)))
        model.AddHint(start[v], start_hint[v])
        model.AddHint(end[v], end_hint[v])
    for i in a:
        model.AddHint(a[i], a_hint[i])
        model.AddHint(load[i], load_hint[i])
    for k, var in carrier.items():
        model.AddHint(var, int(k in initial_solution.carrier_requests))
    for k, var in bus.items():
        model.AddHint(var, int(k in initial_solution.integrated_requests))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers or os.cpu_count() or 8
    solver.parameters.relative_gap_limit = gap
    solver.parameters.log_search_progress = log
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 100

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError('No solution found (infeasible model or time limit reached without a solution)')

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


class _RouteOracle:
    ''' Subproblem: best route of ONE vehicle that serves exactly a given set of requests.

    The vehicle can only use the arcs allowed in its group mask (the same arcs as the monolithic model).
    Results are cached by (group, set of requests): vehicles with the same mask share the cache.
    '''

    def __init__(self, problem: Problem, group_masks: list[np.ndarray], time_limit: float):
        n = problem.requests
        size = 2 * n + 1
        nodes = problem.nodes[:size]

        unit = problem.costs.private_per_km / problem.scale
        distances = np.asarray(problem.distances, dtype=np.int64)[:size, :size]

        self.problem = problem
        self.n = n
        self.masks = group_masks
        self.time_limit = time_limit
        self.T = np.asarray(problem.times, dtype=np.int64)[:size, :size].tolist()
        self.cost_matrix = np.rint(unit * distances * COST_SCALE).astype(np.int64)
        self.C = self.cost_matrix.tolist()
        self.ready = [node.ready_time for node in nodes]
        self.due = [max(node.ready_time, node.due_time) for node in nodes]
        self.service = [node.service_time for node in nodes]
        self.demand = [node.demand for node in nodes]
        self.cache: dict[tuple[int, frozenset[int]], tuple[bool, int, list[int]]] = {}
        self.calls = 0

    def solve(self, group: int, requests: frozenset[int]) -> tuple[bool, int, list[int]]:
        ''' Returns (feasible, cost in 1 / COST_SCALE units, route nodes). '''

        key = (group, requests)

        if key not in self.cache:
            self.cache[key] = self._solve(group, requests)

        return self.cache[key]

    def minimal_infeasible(self, group: int, requests: frozenset[int]) -> frozenset[int]:
        ''' Deletion filter: smallest subset (not necessarily minimum) that is still infeasible. '''

        core = set(requests)

        for k in sorted(requests):
            trial = core - {k}

            if trial and not self.solve(group, frozenset(trial))[0]:
                core = trial

        return frozenset(core)

    def _solve(self, group: int, requests: frozenset[int]) -> tuple[bool, int, list[int]]:
        self.calls += 1

        n = self.n
        problem = self.problem
        ready, due, service, demand, T, C = self.ready, self.due, self.service, self.demand, self.T, self.C

        visited = [0] + [i for r in sorted(requests) for i in (r, r + n)]
        position = {node: p for p, node in enumerate(visited)}
        idx = np.array(visited)
        pairs = np.argwhere(self.masks[group][np.ix_(idx, idx)])

        if len(pairs) == 0:
            return False, 0, []

        model = cp_model.CpModel()

        x = {}
        circuit = []
        for p, q in pairs:
            p, q = int(p), int(q)
            x[p, q] = model.NewBoolVar(f'x_{p}_{q}')
            circuit.append((p, q, x[p, q]))

        model.AddCircuit(circuit)

        a = {node: model.NewIntVar(ready[node], due[node], f'a_{node}') for node in visited[1:]}
        load = {
            node: model.NewIntVar(
                max(0, demand[node]), min(problem.capacity, problem.capacity + demand[node]), f'load_{node}'
            )
            for node in visited[1:]
        }
        start = model.NewIntVar(ready[0], due[0], 'start')
        end = model.NewIntVar(ready[0], due[0], 'end')

        for (p, q), var in x.items():
            i, j = visited[p], visited[q]

            if i == 0:
                model.Add(a[j] >= start + service[0] + T[0][j]).OnlyEnforceIf(var)
                model.Add(load[j] == demand[j]).OnlyEnforceIf(var)
            elif j == 0:
                model.Add(end >= a[i] + service[i] + T[i][0]).OnlyEnforceIf(var)
            else:
                model.Add(a[j] >= a[i] + service[i] + T[i][j]).OnlyEnforceIf(var)
                model.Add(load[j] == load[i] + demand[j]).OnlyEnforceIf(var)

        for k in requests:
            d = k + n
            model.Add(a[d] >= a[k] + service[k] + T[k][d])
            model.Add(a[d] - a[k] - service[k] <= problem.max_ride_time)

        model.Add(end - start <= problem.max_duration_route)

        model.Minimize(sum(C[visited[p]][visited[q]] * var for (p, q), var in x.items()))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_workers = 1

        status = solver.Solve(model)

        if status == cp_model.INFEASIBLE:
            return False, 0, []

        if status != cp_model.OPTIMAL:
            raise RuntimeError(
                f'Subproblem not solved to optimality (status {solver.StatusName(status)}) '
                f'for requests {sorted(requests)}: increase `sub_time_limit`'
            )

        successor = {visited[p]: visited[q] for (p, q), var in x.items() if solver.BooleanValue(var)}
        route, current = [], successor[0]

        while current != 0:
            route.append(current)
            current = successor[current]

        return True, int(round(solver.ObjectiveValue())), route


def solve(
    problem: Problem,
    initial_solution: Solution,
    matrices: list[np.ndarray],
    time_limit: float = 120.0,
    max_integrated: int | None = None,
    workers: int = 0,
    gap: float = 0.0,
    log: bool = False,
    monotone: bool = True,
    sub_time_limit: float = 60.0,
) -> Solution:
    ''' Logic-based Benders decomposition for the IDARP (same inputs/outputs as `solver.solve`).

    Master: assigns every request to a vehicle, the common carrier or the bus integration, with one variable
    `theta[v]` that under-estimates the routing cost of vehicle v (no arcs, no times, no loads).
    Subproblem: for each used vehicle, the best route serving exactly its requests (feasibility + cost).
    Cuts (added to the master every iteration, CP-SAT has no lazy callbacks so it is re-solved):
      - infeasible set S: no vehicle of the group can serve (a minimal infeasible subset of) S;
      - feasible set S with cost c: theta >= c when S is served.
    Every master solution with cost >= the incumbent is cut off, so the loop stops when the master is
    infeasible (incumbent proved optimal), at the relative `gap`, or at `time_limit` (best incumbent returned).

    `monotone=True` assumes that removing a request from a feasible route keeps it feasible and does not
    increase its cost (true when times and distances satisfy the triangle inequality and the allowed arcs
    allow the shortcuts). It gives strong cuts (supersets of S are also infeasible / at least as expensive),
    and every solution returned is feasible anyway, but optimality within the restricted arcs is only
    guaranteed if the assumption holds. `monotone=False` uses exact-set no-good cuts, always valid but slow.
    '''

    started = time.monotonic()
    deadline = started + time_limit

    n = problem.requests
    size = 2 * n + 1
    vehicles = range(len(matrices))
    requests = range(1, n + 1)
    request_of = lambda i: i if i <= n else i - n

    # Allowed arcs, eligible requests and groups of interchangeable vehicles (same allowed arcs)

    static_ok = _static_arc_filter(problem)
    request_ok = np.array([
        problem.nodes[r].ready_time <= problem.nodes[r].due_time
        and problem.nodes[r + n].ready_time <= problem.nodes[r + n].due_time
        for r in requests
    ])

    eligible: list[list[int]] = []
    group_of: list[int] = []
    group_masks: list[np.ndarray] = []
    group_members: list[list[int]] = []
    seen: dict[bytes, int] = {}

    for v, matrix in enumerate(matrices):
        mask, elig = _restrict(static_ok & (matrix[:size, :size] >= 0), n, request_ok)
        key = mask.tobytes()

        if key not in seen:
            seen[key] = len(group_masks)
            group_masks.append(mask)
            group_members.append([])

        group_of.append(seen[key])
        group_members[seen[key]].append(v)
        eligible.append(elig)

    oracle = _RouteOracle(problem, group_masks, sub_time_limit)

    # Master model

    model = cp_model.CpModel()

    y = {(r, v): model.NewBoolVar(f'y_{r}_{v}') for v in vehicles for r in eligible[v]}
    used = [model.NewBoolVar(f'used_{v}') for v in vehicles]
    theta_ub = int(oracle.cost_matrix.max()) * size + 1
    theta = [model.NewIntVar(0, theta_ub, f'theta_{v}') for v in vehicles]

    carrier_cost = {k: round(problem.carrier_cost(k) * COST_SCALE) for k in requests if problem.carrier_feasible(k)}
    bus_cost = {k: round(problem.integration_cost(k) * COST_SCALE) for k in requests if problem.integration_feasible(k)}
    carrier = {k: model.NewBoolVar(f'carrier_{k}') for k in carrier_cost}
    bus = {k: model.NewBoolVar(f'bus_{k}') for k in bus_cost}

    for k in requests:
        options = [y[k, v] for v in vehicles if (k, v) in y]
        if k in carrier:
            options.append(carrier[k])
        if k in bus:
            options.append(bus[k])

        model.AddExactlyOne(options)

    if max_integrated is not None:
        model.Add(sum(bus.values()) <= max_integrated)

    for v in vehicles:
        for r in eligible[v]:
            model.Add(y[r, v] <= used[v])

        model.Add(used[v] <= sum(y[r, v] for r in eligible[v]))
        model.Add(theta[v] == 0).OnlyEnforceIf(used[v].Not())

    for v in range(len(matrices) - 1):
        if group_of[v] == group_of[v + 1]:
            model.Add(used[v] >= used[v + 1])

    if monotone:
        # Single-request routes: infeasible singletons are removed, feasible ones give a lower bound on theta
        for gid, members in enumerate(group_members):
            for k in eligible[members[0]]:
                feasible, cost, _ = oracle.solve(gid, frozenset([k]))

                for w in members:
                    if not feasible:
                        model.Add(y[k, w] == 0)
                    else:
                        model.Add(theta[w] >= cost * y[k, w])

    total_ub = theta_ub * len(matrices) + sum(carrier_cost.values()) + sum(bus_cost.values()) + 1
    z = model.NewIntVar(0, total_ub, 'z')
    model.Add(z == sum(theta) + sum(c * carrier[k] for k, c in carrier_cost.items())
              + sum(c * bus[k] for k, c in bus_cost.items()))
    model.Minimize(z)

    # Cuts

    added: set = set()

    def add_cost_cut(gid: int, served: frozenset[int], cost: int) -> None:
        if (key := ('cost', gid, served)) in added:
            return
        added.add(key)

        for w in group_members[gid]:
            expr = sum(y[k, w] for k in served)
            if not monotone:
                expr = expr - sum(y[k, w] for k in eligible[w] if k not in served)

            model.Add(theta[w] >= cost * expr - cost * (len(served) - 1))

    def add_infeasibility_cut(gid: int, served: frozenset[int]) -> None:
        core = oracle.minimal_infeasible(gid, served) if monotone else served

        if (key := ('infeasible', gid, core)) in added:
            return
        added.add(key)

        for w in group_members[gid]:
            expr = sum(y[k, w] for k in core)
            if not monotone:
                expr = expr - sum(y[k, w] for k in eligible[w] if k not in core)

            model.Add(expr <= len(core) - 1)

    def evaluate(sets: list[frozenset[int]], carrier_set: set[int], bus_set: set[int]):
        ''' Solves the subproblems of an assignment and adds the cuts. Returns (total cost, routes) or None. '''

        total = sum(carrier_cost[k] for k in carrier_set) + sum(bus_cost[k] for k in bus_set)
        routes: list[list[int]] = []
        feasible = True

        for v, served in enumerate(sets):
            if not served:
                continue

            ok, cost, route = oracle.solve(group_of[v], served)

            if ok:
                total += cost
                routes.append(route)
                add_cost_cut(group_of[v], served, cost)
            else:
                feasible = False
                add_infeasibility_cut(group_of[v], served)

        return (total, routes) if feasible else None

    # Incumbent from the initial solution (its routes are the first vehicles, as in `expand_neighbors`)

    incumbent: tuple[int, list[list[int]], set[int], set[int]] | None = None
    lower_bound = 0

    hint_routes = [route for route in initial_solution.routes if route.nodes][:len(matrices)]
    hint_sets = [frozenset(request_of(i) for i in route.nodes) for route in hint_routes]
    hint_sets += [frozenset()] * (len(matrices) - len(hint_sets))

    if all(served <= set(eligible[v]) for v, served in enumerate(hint_sets)):
        result = evaluate(hint_sets, set(initial_solution.carrier_requests), set(initial_solution.integrated_requests))

        if result is not None:
            incumbent = (result[0], result[1], set(initial_solution.carrier_requests),
                         set(initial_solution.integrated_requests))
            model.Add(z <= incumbent[0] - 1)

    if log:
        print(f'[benders] initial incumbent: {incumbent[0] if incumbent else None}, subproblems: {oracle.calls}')

    # Benders loop

    iteration = 0
    status_text = 'time limit'

    while (remaining := deadline - time.monotonic()) > 0:
        iteration += 1

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_workers = workers or 0
        status = solver.Solve(model)

        if status == cp_model.INFEASIBLE:
            status_text = 'optimal (master infeasible below the incumbent)'
            break

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        if status == cp_model.OPTIMAL:
            lower_bound = max(lower_bound, int(round(solver.ObjectiveValue())))
        else:
            lower_bound = max(lower_bound, math.ceil(solver.BestObjectiveBound()))

        sets = [frozenset(r for r in eligible[v] if solver.BooleanValue(y[r, v])) for v in vehicles]
        carrier_set = {k for k, var in carrier.items() if solver.BooleanValue(var)}
        bus_set = {k for k, var in bus.items() if solver.BooleanValue(var)}

        result = evaluate(sets, carrier_set, bus_set)

        if result is not None and (incumbent is None or result[0] < incumbent[0]):
            incumbent = (result[0], result[1], carrier_set, bus_set)
            model.Add(z <= incumbent[0] - 1)

        if log:
            print(
                f'[benders] iter {iteration}: lb {lower_bound} ub {incumbent[0] if incumbent else None} '
                f'subproblems {oracle.calls} cuts {len(added)} elapsed {time.monotonic() - started:.1f}s'
            )

        if incumbent is not None and incumbent[0] - lower_bound <= gap * incumbent[0]:
            status_text = f'gap <= {gap}'
            break

    if incumbent is None:
        raise RuntimeError('No solution found (infeasible model or time limit reached without a solution)')

    if log:
        print(f'[benders] finished: {status_text}, cost {incumbent[0]}, lb {lower_bound}, iterations {iteration}')

    return Solution(
        problem,
        [Route(problem, route) for route in incumbent[1]],
        incumbent[2],
        incumbent[3],
    )