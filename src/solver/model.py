from __future__ import annotations

from types import SimpleNamespace
from typing import Collection

import numpy as np
from ortools.sat.python import cp_model

from src.problem import Problem
from src.solution import Solution
from src.solver.scope import Scope

# CP-SAT needs integer coefficients: costs are multiplied by this and rounded
SCALE = 1000


class IdarpModel:
    ''' CP-SAT model of the IDARP restricted to the arcs of `scope`.

    Each request is served by one vehicle (one circuit per vehicle), by the common carrier or by the bus
    integration (the last two only if feasible). The objective is the total cost.
    '''

    def __init__(
        self,
        problem: Problem,
        ctx: SimpleNamespace,
        scope: Scope,
        fixed: Collection[int],
        max_integrated: int | None,
    ):
        self.problem, self.ctx, self.scope = problem, ctx, scope
        self.n = problem.requests
        self.free = [k for k in range(1, self.n + 1) if k not in fixed]
        self.model = cp_model.CpModel()

        self._variables()
        self._assignment(max_integrated)
        self._circuits()
        self._symmetry()
        self._propagation()
        self._objective()

    @staticmethod
    def request_of(i: int, n: int) -> int:
        return i if i <= n else i - n

    # Model

    def _variables(self):
        problem, ctx, s, n, m = self.problem, self.ctx, self.scope, self.n, self.model

        self.y = {(r, v): m.NewBoolVar(f'y_{r}_{v}') for v in s.vehicles for r in s.eligible[v]}
        self.used = {v: m.NewBoolVar(f'used_{v}') for v in s.vehicles}

        self.x: dict[tuple[int, int, int], cp_model.IntVar] = {}
        self.arcs_of = {v: [] for v in s.vehicles}
        self.circuit_arcs = {v: [] for v in s.vehicles}
        self.visited_of: dict[int, list[int]] = {}

        for v in s.vehicles:
            visited = [0] + [i for r in s.eligible[v] for i in (r, r + n)]
            self.visited_of[v] = visited

            idx = np.array(visited)
            sub = s.mask_of[v][np.ix_(idx, idx)]

            for p, q in np.argwhere(sub):
                p, q = int(p), int(q)
                i, j = visited[p], visited[q]
                var = m.NewBoolVar(f'x_{i}_{j}_{v}')
                self.x[i, j, v] = var
                self.arcs_of[v].append((i, j, var))
                self.circuit_arcs[v].append((p, q, var))  # circuit index == position in `visited`

        self.carrier = {k: m.NewBoolVar(f'carrier_{k}') for k in self.free if problem.carrier_feasible(k)}
        self.bus = {k: m.NewBoolVar(f'bus_{k}') for k in self.free if problem.integration_feasible(k)}
        self.private = {k: m.NewBoolVar(f'private_{k}') for k in s.active}

        lo, hi, dl, cap = s.lo, s.hi, ctx.demand_l, problem.capacity
        self.a = {i: m.NewIntVar(lo[i], max(lo[i], hi[i]), f'a_{i}') for i in s.active_nodes}
        self.load = {
            i: m.NewIntVar(max(0, dl[i]), min(cap, cap + dl[i]), f'load_{i}') for i in s.active_nodes
        } if ctx.need_load else {}

        self.start, self.end = {}, {}

        if ctx.duration_binding:
            r0, d0 = ctx.ready_l[0], ctx.due_l[0]
            self.start = {v: m.NewIntVar(r0, d0, f'start_{v}') for v in s.vehicles}
            self.end = {v: m.NewIntVar(r0, d0, f'end_{v}') for v in s.vehicles}

    def _assignment(self, max_integrated: int | None):
        ''' Each request is served in exactly one way. '''

        m = self.model

        for k in self.free:
            options = []

            if k in self.private:
                m.Add(self.private[k] == sum(self.y[k, v] for v in self.scope.vehicles if (k, v) in self.y))
                options.append(self.private[k])
            if k in self.carrier:
                options.append(self.carrier[k])
            if k in self.bus:
                options.append(self.bus[k])

            m.AddExactlyOne(options)

        if max_integrated is not None:
            m.Add(sum(self.bus.values()) <= max_integrated)

    def _circuits(self):
        ''' One circuit per vehicle: nodes it does not visit take a self-loop. '''

        for v in self.scope.vehicles:
            visited = self.visited_of[v]
            circuit = [(0, 0, self.used[v].Not())]
            circuit += [
                (p, p, self.y[self.request_of(visited[p], self.n), v].Not()) for p in range(1, len(visited))
            ]
            circuit += self.circuit_arcs[v]

            self.model.AddCircuit(circuit)

    def _symmetry(self):
        ''' Within a group of identical vehicles: used first, ordered by their smallest request. '''

        n, m, s = self.n, self.model, self.scope

        for members in s.kept_members:
            for v, w in zip(members, members[1:]):
                m.Add(self.used[v] >= self.used[w])

            if len(members) > 1:
                first = {}

                for v in members:
                    first[v] = m.NewIntVar(1, n + 1, f'first_{v}')
                    m.AddMinEquality(
                        first[v], [(n + 1) - (n + 1 - r) * self.y[r, v] for r in s.eligible[v]]
                    )

                for v, w in zip(members, members[1:]):
                    m.Add(first[v] < first[w]).OnlyEnforceIf(self.used[w])

    def _propagation(self):
        ''' Time and load along the arcs, precedence and ride time (constraints implied by the domains are skipped). '''

        ctx, s, n, m = self.ctx, self.scope, self.n, self.model
        lo, hi, a = s.lo, s.hi, self.a
        T, C = ctx.Tl, ctx.Cl
        ready, due, service, demand = ctx.ready_l, ctx.due_l, ctx.service_l, ctx.demand_l
        binding, need_load = ctx.duration_binding, ctx.need_load
        max_ride, max_duration = ctx.data.max_ride, ctx.data.max_duration

        for (i, j, v), var in self.x.items():
            travel = T[i][j]

            if i == 0:
                if binding:
                    m.Add(a[j] >= self.start[v] + service[0] + travel).OnlyEnforceIf(var)
                elif ready[0] + service[0] + travel > lo[j]:
                    m.Add(a[j] >= ready[0] + service[0] + travel).OnlyEnforceIf(var)

                if need_load:
                    m.Add(self.load[j] == demand[j]).OnlyEnforceIf(var)
            elif j == 0:
                if binding:
                    m.Add(self.end[v] >= a[i] + service[i] + travel).OnlyEnforceIf(var)
                elif due[0] - service[i] - travel < hi[i]:
                    m.Add(a[i] <= due[0] - service[i] - travel).OnlyEnforceIf(var)
            else:
                if lo[j] < hi[i] + service[i] + travel:
                    m.Add(a[j] >= a[i] + service[i] + travel).OnlyEnforceIf(var)

                if need_load:
                    m.Add(self.load[j] == self.load[i] + demand[j]).OnlyEnforceIf(var)

        # Precedence and ride time, only for requests served by the private fleet
        for k, private in self.private.items():
            d = k + n

            if hi[k] + C[k][d] > lo[d]:
                m.Add(a[d] >= a[k] + C[k][d]).OnlyEnforceIf(private)

            if hi[d] - lo[k] - service[k] > max_ride:
                m.Add(a[d] - a[k] - service[k] <= max_ride).OnlyEnforceIf(private)

        if binding:
            for v in s.vehicles:
                m.Add(self.end[v] - self.start[v] <= max_duration)

    def _objective(self):
        ''' Total cost in integer units of 1 / SCALE. '''

        problem = self.problem
        self.arc_cost = lambda i, j: round(problem.private_cost(i, j) * SCALE)
        self.carrier_cost = {k: round(problem.carrier_cost(k) * SCALE) for k in self.carrier}
        self.bus_cost = {k: round(problem.integration_cost(k) * SCALE) for k in self.bus}

        terms = [(var, self.arc_cost(i, j)) for (i, j, _), var in self.x.items()]
        terms += [(var, self.carrier_cost[k]) for k, var in self.carrier.items()]
        terms += [(var, self.bus_cost[k]) for k, var in self.bus.items()]
        terms = [(var, coef) for var, coef in terms if coef != 0]

        self.objective = self.model.NewIntVar(0, sum(coef for _, coef in terms), 'objective')
        self.model.Add(
            self.objective == cp_model.LinearExpr.WeightedSum([t[0] for t in terms], [t[1] for t in terms])
        )
        self.model.Minimize(self.objective)

    # Initial solution

    def plan(self, initial: Solution, num_matrices: int):
        ''' Routes of `initial` assigned to vehicles: the first matrices are the used routes (see `expand`);
        routes of interchangeable vehicles are sorted by smallest request to respect the symmetry breaking. '''

        s, n = self.scope, self.n
        routes = [route for route in initial.routes if route.nodes][:num_matrices]
        by_group: dict[int, list[list[int]]] = {}

        for v, route in enumerate(routes):
            by_group.setdefault(s.group_of[v], []).append(list(route.nodes))

        assign: dict[int, list[int]] = {}

        for g, group_routes in by_group.items():
            group_routes.sort(key=lambda nodes: min(self.request_of(i, n) for i in nodes))

            for v, nodes in zip(s.group_members[g], group_routes):
                assign[v] = nodes

        arcs = {(u, w, v) for v, nodes in assign.items() for u, w in zip([0, *nodes], [*nodes, 0])}
        requests = {(self.request_of(i, n), v) for v, nodes in assign.items() for i in nodes}

        return assign, arcs, requests, {k for k, _ in requests}

    def add_hint(self, initial: Solution, plan):
        ''' Warm start: the complete initial solution with its earliest schedule (clipped to the domains). '''

        ctx, s, m = self.ctx, self.scope, self.model
        lo, hi, cap = s.lo, s.hi, self.problem.capacity
        T, service, demand = ctx.Tl, ctx.service_l, ctx.demand_l
        ready0, due0 = ctx.ready_l[0], ctx.due_l[0]
        assign, arcs, requests, private = plan

        a_hint = {i: lo[i] for i in range(1, 2 * self.n + 1)}
        load_hint = {i: max(0, demand[i]) for i in range(1, 2 * self.n + 1)}
        end_hint = {v: ready0 for v in s.vehicles}

        for v, nodes in assign.items():
            leave, previous, current_load = ready0 + service[0], 0, 0

            for node in nodes:
                a_hint[node] = min(hi[node], max(lo[node], leave + T[previous][node]))
                current_load += demand[node]
                load_hint[node] = min(
                    max(current_load, max(0, demand[node])), min(cap, cap + demand[node])
                )
                leave = a_hint[node] + service[node]
                previous = node

            end_hint[v] = min(due0, max(ready0, leave + T[previous][0]))

        for key, var in self.x.items():
            m.AddHint(var, int(key in arcs))
        for key, var in self.y.items():
            m.AddHint(var, int(key in requests))
        for k, var in self.private.items():
            m.AddHint(var, int(k in private))
        for v in s.vehicles:
            m.AddHint(self.used[v], int(v in assign))

            if ctx.duration_binding:
                m.AddHint(self.start[v], ready0)
                m.AddHint(self.end[v], end_hint[v])

        for i, var in self.a.items():
            m.AddHint(var, a_hint[i])
        for i, var in self.load.items():
            m.AddHint(var, load_hint[i])
        for k, var in self.carrier.items():
            m.AddHint(var, int(k in initial.carrier_requests))
        for k, var in self.bus.items():
            m.AddHint(var, int(k in initial.integrated_requests))

    def add_upper_bound(self, initial: Solution, plan, strict: bool) -> int | None:
        ''' Cuts "cost <= cost of `initial`" (it is feasible, so the optimum cannot cost more).
        Returns that cost, or None if `initial` is not inside the model (an error when `strict`). '''

        _, arcs, requests, private = plan
        carrier_set, bus_set = set(initial.carrier_requests), set(initial.integrated_requests)
        free = set(self.free)

        complete = (
            all(key in self.x for key in arcs)
            and all(key in self.y for key in requests)
            and carrier_set <= set(self.carrier)
            and bus_set <= set(self.bus)
            and len(private) + len(carrier_set) + len(bus_set) == len(free)
            and private | carrier_set | bus_set == free
        )

        if not complete:
            if strict:
                raise RuntimeError('The initial solution is not in the model, so no strict improvement can be proved')

            return None

        cost = (
            sum(self.arc_cost(i, j) for i, j, _ in arcs)
            + sum(self.carrier_cost[k] for k in carrier_set)
            + sum(self.bus_cost[k] for k in bus_set)
        )
        self.model.Add(self.objective <= cost)

        return cost
