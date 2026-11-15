from __future__ import annotations

import time

from src.problem import Problem, CARRIER, INTEGRATION, ROUTE
from src.solution import Route, Solution

EPSILON = 1e-9
EMPTY = -1  # construction cache key of the empty-route candidate
MISSING_PENALTY = 1e6  # regret contribution of an alternative that does not exist

# Construction

def _regret(sorted_costs: list[float], regret_k: int) -> float:
    best = sorted_costs[0]

    return sum(
        sorted_costs[j] - best if j < len(sorted_costs) else MISSING_PENALTY
        for j in range(1, regret_k)
    )

def construct(problem: Problem, *, regret_k: int = 2, max_integrated: int | None = None) -> Solution:
    ''' Parallel regret-k construction with insertion caching.

    Each request chooses among its best insertion in every route (plus one empty route while vehicles
    remain), the common carrier and the bus integration. The request with the largest regret goes first,
    to its cheapest alternative. The carrier guarantees feasibility.
    '''

    requests = range(1, problem.requests + 1)
    per_km = problem.costs.private_per_km

    routes: list[Route] = []
    carrier: set[int] = set()
    integrated: set[int] = set()

    fixed = {k: problem.fixed_options(k) for k in requests}
    unserved = list(requests)

    # cache[k][route index] = (cost, route after inserting k)
    cache: dict[int, dict[int, tuple[float, Route]]] = {k: {} for k in requests}

    def refresh(k: int, index: int, route: Route):
        if result := route.best_insertion(k):
            cache[k][index] = (per_km * result[0], result[1])
        else:
            cache[k].pop(index, None)

    if problem.vehicles > 0:
        empty = Route(problem, [])

        for k in unserved:
            refresh(k, EMPTY, empty)

    while unserved:
        integration_closed = max_integrated is not None and len(integrated) >= max_integrated
        selected = None

        for k in unserved:
            options = [(cost, ROUTE, index) for index, (cost, _) in cache[k].items()]
            options += [
                (cost, tag, None) for cost, tag in fixed[k]
                if not (tag == INTEGRATION and integration_closed)
            ]

            if not options:
                raise RuntimeError(f'Request {k} cannot be served by any alternative')

            options.sort(key=lambda o: (o[0], o[1], -1 if o[2] is None else o[2]))
            score = (-_regret([o[0] for o in options], regret_k), options[0][0], k)

            if selected is None or score < selected[0]:
                selected = (score, k, options[0])

        _, k, (_, tag, index) = selected
        unserved.remove(k)
        entries = cache.pop(k)

        if tag == CARRIER:
            carrier.add(k)
            continue

        if tag == INTEGRATION:
            integrated.add(k)
            continue

        new_route = entries[index][1]

        if index == EMPTY:
            routes.append(new_route)
            index = len(routes) - 1

            if len(routes) >= problem.vehicles:  # no vehicle left to open
                for other in cache.values():
                    other.pop(EMPTY, None)
        else:
            routes[index] = new_route

        for other in unserved:  # only the changed route needs new insertions
            refresh(other, index, new_route)

    return Solution(problem, routes, carrier, integrated)

# Local search

class _State:
    ''' Mutable copy of a solution used by the moves. '''

    def __init__(self, problem: Problem, solution: Solution, max_integrated: int | None):
        self.problem = problem
        self.max_integrated = max_integrated
        self.routes = [Route(problem, list(route.nodes)) for route in solution.routes if route.nodes]
        self.carrier = set(solution.carrier_requests)
        self.integrated = set(solution.integrated_requests)
        self.fixed = {k: problem.fixed_options(k) for k in range(1, problem.requests + 1)}

    @property
    def outside(self) -> set[int]:
        return self.carrier | self.integrated

    def solution(self) -> Solution:
        return Solution(self.problem, self.routes, self.carrier, self.integrated)

    def can_open(self) -> bool:
        ''' Whether there is a vehicle left to open an empty route. '''

        return len(self.routes) < self.problem.vehicles

    def best_outside(self, k: int, freed: set[int] = frozenset()) -> tuple[float, int] | None:
        ''' Cheapest carrier / bus alternative of `k`: (cost, tag) or None.

        `freed` are requests that leave the bus in the same move (they free a seat of `max_integrated`).
        '''

        taken = len(self.integrated - set(freed))
        closed = self.max_integrated is not None and taken >= self.max_integrated
        options = [o for o in self.fixed[k] if not (o[1] == INTEGRATION and closed)]

        return min(options) if options else None

    def send_outside(self, k: int, tag: int):
        (self.carrier if tag == CARRIER else self.integrated).add(k)

    def bring_inside(self, k: int):
        self.carrier.discard(k)
        self.integrated.discard(k)

    def place(self, index: int, route: Route):
        ''' Stores `route` at `index` (`index == len(routes)` opens a new vehicle). '''

        if index == len(self.routes):
            self.routes.append(route)
        else:
            self.routes[index] = route

    def update(self, index: int, route: Route):
        ''' Replaces the route at `index`, dropping it when it became empty. '''

        if route.nodes:
            self.routes[index] = route
        else:
            del self.routes[index]

    def private_requests(self) -> list[int]:
        n = self.problem.requests

        return sorted(i for route in self.routes for i in route.nodes if i <= n)


def _without(problem: Problem, route: Route, k: int) -> Route | None:
    ''' The route without request `k` (None if not feasible: the time matrix may break the triangle
    inequality, so removing nodes is not always safe). '''

    n = problem.requests
    nodes = [i for i in route.nodes if i != k and i != k + n]
    new_route = Route(problem, nodes)

    return new_route if not nodes or new_route.feasible else None


def _best_gain(
    problem: Problem, targets: list[tuple[int, Route]], k: int, base: float
) -> tuple[float, int, Route] | None:
    ''' Best insertion of `k` among `targets` (index, route): (gain, index, new route), where
    gain = base - insertion cost. None if no insertion has a positive gain. '''

    per_km = problem.costs.private_per_km
    best = None

    for index, route in targets:
        if result := route.best_insertion(k):
            gain = base - per_km * result[0]

            if gain > EPSILON and (best is None or gain > best[0]):
                best = (gain, index, result[1])

    return best


def _move_in(state: _State) -> bool:
    ''' Carrier / bus -> private fleet: an outside request goes to its best insertion (an empty vehicle
    included) when it is cheaper than its current alternative. '''

    problem = state.problem
    improved = False

    for k in sorted(state.outside):
        targets = list(enumerate(state.routes))

        if state.can_open():
            targets.append((len(state.routes), Route(problem, [])))

        if best := _best_gain(problem, targets, k, problem.outside_cost(k, state.carrier)):
            _, index, route = best
            state.bring_inside(k)
            state.place(index, route)
            improved = True

    return improved


def _move_out(state: _State) -> bool:
    ''' Private fleet -> carrier / bus: a request leaves its route when the route saves more than the
    cheapest outside alternative costs. '''

    n = state.problem.requests
    improved = False
    r = 0

    while r < len(state.routes):
        route = state.routes[r]
        moved = False

        for k in [i for i in route.nodes if i <= n]:
            removed = _without(state.problem, route, k)
            option = state.best_outside(k)

            if removed is None or option is None:
                continue

            if route.cost - removed.cost - option[0] > EPSILON:
                state.send_outside(k, option[1])
                state.update(r, removed)
                improved = moved = True
                break

        if not moved:
            r += 1  # after a move the same index holds a changed (or the next) route

    return improved


def _relocate(state: _State) -> bool:
    ''' Moves a private request to its best position in any route (the same one, another one or an empty
    vehicle). Moving inside the same route reorders it. '''

    problem = state.problem
    improved = False

    for k in state.private_requests():
        a = next(i for i, route in enumerate(state.routes) if k in route.nodes)
        source = state.routes[a]
        removed = _without(problem, source, k)

        if removed is None:
            continue

        targets = [(a, removed)] if removed.nodes else []
        targets += [(b, route) for b, route in enumerate(state.routes) if b != a]

        if removed.nodes and state.can_open():
            targets.append((len(state.routes), Route(problem, [])))

        if (best := _best_gain(problem, targets, k, source.cost - removed.cost)) is None:
            continue

        _, b, new_target = best

        if b == a:
            state.routes[a] = new_target
        else:
            state.place(b, new_target)
            state.update(a, removed)

        improved = True

    return improved


def _swap(state: _State) -> bool:
    ''' Exchange: a private request j goes to the carrier / bus and an outside request k takes its place in
    the route. Finds what `_move_in` and `_move_out` cannot do alone (each is a loss, together a gain).

    A pair is only tried when its gain without the insertion cost (never negative, except for small
    triangle inequality violations) is positive. Applies the first improvement found.
    '''

    problem = state.problem
    per_km = problem.costs.private_per_km

    for r, route in enumerate(state.routes):
        for j in [i for i in route.nodes if i <= problem.requests]:
            removed = _without(problem, route, j)

            if removed is None:
                continue

            saving = route.cost - removed.cost

            for k in sorted(state.outside):
                option = state.best_outside(j, freed={k})

                if option is None:
                    continue

                bound = saving + problem.outside_cost(k, state.carrier) - option[0]

                if bound <= EPSILON:
                    continue

                if result := removed.best_insertion(k):
                    if bound - per_km * result[0] > EPSILON:
                        state.bring_inside(k)
                        state.send_outside(j, option[1])
                        state.routes[r] = result[1]

                        return True

    return False

MOVES = (('in', _move_in), ('out', _move_out), ('relocate', _relocate), ('swap', _swap))

def local_search(
    problem: Problem,
    solution: Solution,
    *,
    max_integrated: int | None = None,
    time_limit: float | None = None,
    log: bool = False,
) -> Solution:
    ''' Variable neighborhood descent over a feasible solution, to polish the result of `construct`.

    Moves, cheapest first: carrier / bus -> private (`in`), private -> carrier / bus (`out`), relocation of a
    private request (`relocate`, also reorders a route) and exchange of a private request with an outside one
    (`swap`). After any improvement it restarts from the first move; it stops at a local optimum or at
    `time_limit` seconds. Never returns a solution worse than the given one.
    '''

    start = time.perf_counter()
    state = _State(problem, solution, max_integrated)
    improved = True

    while improved and (time_limit is None or time.perf_counter() - start < time_limit):
        improved = False

        for name, move in MOVES:
            if move(state):
                improved = True

                if log:
                    print(f'[{time.perf_counter() - start:7.1f}s] local search {name:<9} cost {state.solution().cost:.4f}')

                break

    result = state.solution()
    result.validate()

    return result if result.cost < solution.cost - EPSILON else solution