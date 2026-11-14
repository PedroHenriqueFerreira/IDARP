from __future__ import annotations

from src.problem import Problem
from src.route import Route
from src.solution import Solution

ROUTE, INTEGRATION, CARRIER = 0, 1, 2  # tie-break order: private fleet first
EMPTY = -1  # cache key of the single empty route candidate
MISSING_PENALTY = 1e6  # regret contribution of an alternative that does not exist

def _best_insertion(problem: Problem, route: Route, k: int) -> tuple[int, Route] | None:
    ''' Best feasible insertion of request `k` in `route`: (distance increase, new route) or None. '''

    dist = problem.distances
    nodes = route.nodes
    m = len(nodes)
    p, d = k, k + problem.requests

    load = [0] * (m + 1)  # load before each position
    for t, node_id in enumerate(nodes):
        load[t + 1] = load[t] + problem.nodes[node_id].demand

    candidates = []

    for i in range(m + 1):
        if load[i] + 1 > problem.capacity:
            continue

        prev = nodes[i - 1] if i > 0 else 0

        for j in range(i, m + 1):
            if j > i and load[j] + 1 > problem.capacity:
                break

            if j == i:
                nxt = nodes[i] if i < m else 0
                delta = dist[prev, p] + dist[p, d] + dist[d, nxt] - dist[prev, nxt]
            else:
                after_p, before_d = nodes[i], nodes[j - 1]
                after_d = nodes[j] if j < m else 0
                delta = (
                    dist[prev, p] + dist[p, after_p] - dist[prev, after_p]
                    + dist[before_d, d] + dist[d, after_d] - dist[before_d, after_d]
                )

            candidates.append((int(delta), i, j))

    for delta, i, j in sorted(candidates):
        new_route = Route(problem, nodes[:i] + [p] + nodes[i:j] + [d] + nodes[j:])

        if new_route.feasible:
            return delta, new_route

    return None


def _fixed_options(problem: Problem, k: int) -> list[tuple[float, int]]:
    ''' Alternatives whose cost does not depend on the routes: (cost, tag). '''

    options = []

    if problem.carrier_feasible(k):
        options.append((problem.carrier_cost(k), CARRIER))

    if problem.integration_feasible(k):
        options.append((problem.integration_cost(k), INTEGRATION))

    return options


def _regret(sorted_costs: list[float], regret_k: int) -> float:
    best = sorted_costs[0]

    return sum(
        sorted_costs[j] - best if j < len(sorted_costs) else MISSING_PENALTY
        for j in range(1, regret_k)
    )

def construct(problem: Problem, regret_k: int = 2, max_integrated: int | None = None) -> Solution:
    ''' Parallel regret-k construction for the IDARP, with insertion caching.

    Each request chooses among its best insertion in every route, opening one empty route (if there are
    vehicles left), the common carrier and the bus integration. The request with the largest regret
    is assigned first, to its cheapest alternative. The carrier guarantees a feasible solution.
    '''

    requests = range(1, problem.requests + 1)
    unit_cost = problem.costs.private_per_km / problem.scale

    routes: list[Route] = []
    carrier: set[int] = set()
    integrated: set[int] = set()

    fixed = {k: _fixed_options(problem, k) for k in requests}

    # cache[k][route_index] = (cost, route after inserting k)
    cache: dict[int, dict[int, tuple[float, Route]]] = {k: {} for k in requests}
    unserved = list(requests)

    if problem.vehicles > 0:
        empty = Route(problem, [])

        for k in unserved:
            if result := _best_insertion(problem, empty, k):
                cache[k][EMPTY] = (unit_cost * result[0], result[1])

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

        _, k, (_, tag, route_index) = selected
        unserved.remove(k)
        entries = cache.pop(k)

        if tag == CARRIER:
            carrier.add(k)
            continue

        if tag == INTEGRATION:
            integrated.add(k)
            continue

        new_route = entries[route_index][1]

        if route_index == EMPTY:
            routes.append(new_route)
            route_index = len(routes) - 1

            if len(routes) >= problem.vehicles:
                for other_entries in cache.values():
                    other_entries.pop(EMPTY, None)
        else:
            routes[route_index] = new_route

        for other in unserved:
            if result := _best_insertion(problem, new_route, other):
                cache[other][route_index] = (unit_cost * result[0], result[1])
            else:
                cache[other].pop(route_index, None)

    return Solution(problem, routes, carrier, integrated)