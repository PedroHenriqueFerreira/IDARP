from __future__ import annotations

from src.problem import Problem
from src.route import Route
from src.solution import Solution

# Tags used to break ties between options of equal cost (private fleet first)
ROUTE, INTEGRATION, CARRIER = 0, 1, 2

# Key of the "empty route" candidate in the insertion cache. Only ONE empty route is
# ever considered: all empty routes are identical, so keeping several would give
# an artificial regret of 0 to every request whose best option is "open a route".
EMPTY = -1

# Regret contribution of an option that does not exist (infeasible). Large and finite,
# so requests with fewer alternatives are served first without producing inf/nan.
MISSING_PENALTY = 1e6

def _best_insertion(problem: Problem, route: Route, k: int) -> tuple[float, Route] | None:
    ''' Best feasible insertion of request `k` in `route`.

    Instead of building and checking every (pickup, delivery) position pair, it:
      1. discards positions that violate capacity, using the load profile of the route;
      2. computes the distance increase of each remaining pair in O(1);
      3. checks full feasibility in increasing order of increase and stops at the first feasible one.

    Returns (distance increase in scaled units, new route) or None if the request does not fit.
    '''

    dist = problem.distances
    p, d = k, k + problem.requests

    # Lb[t] = vehicle load before visiting position t (Lb[m] = load at the end)
    Lb = [0] * (len(route.nodes) + 1)
    for t, node_id in enumerate(route.nodes):
        Lb[t + 1] = Lb[t] + problem.nodes[node_id].demand

    candidates: list[tuple[int, int, int]] = []

    for i in range(len(route.nodes) + 1):  # pickup goes before original position i
        if Lb[i] + 1 > problem.capacity:
            continue

        prev = route.nodes[i - 1] if i > 0 else 0

        for j in range(i, len(route.nodes) + 1):  # delivery goes before original position j (j >= i)
            if j > i and Lb[j] + 1 > problem.capacity:
                break  # the vehicle is full after original node j - 1: cannot extend further

            if j == i:
                nxt = route.nodes[i] if i < len(route.nodes) else 0
                delta = dist[prev, p] + dist[p, d] + dist[d, nxt] - dist[prev, nxt]
            else:
                after_p = route.nodes[i]
                before_d = route.nodes[j - 1]
                after_d = route.nodes[j] if j < len(route.nodes) else 0
                delta = (
                    dist[prev, p] + dist[p, after_p] - dist[prev, after_p]
                    + dist[before_d, d] + dist[d, after_d] - dist[before_d, after_d]
                )

            candidates.append((int(delta), i, j))

    candidates.sort()

    for delta, i, j in candidates:
        new_route = Route(problem, route.nodes[:i] + [p] + route.nodes[i:j] + [d] + route.nodes[j:])

        if new_route.feasible:
            return delta, new_route

    return None


# ---------------------------------------------------------------------------
# Parallel Regret-k construction
# ---------------------------------------------------------------------------

def _regret(sorted_costs: list[float], regret_k: int) -> float:
    best = sorted_costs[0]
    regret = 0.0

    for j in range(1, regret_k):
        regret += (sorted_costs[j] - best) if j < len(sorted_costs) else MISSING_PENALTY

    return regret

def construct(problem: Problem, regret_k: int = 2, max_integrated: int | None = None) -> Solution:
    ''' Parallel Regret-k construction for the IDARP, with insertion caching.

    Every unserved request has a set of alternatives with a cost:
      - best insertion in each non-empty route of the private fleet (cost: private_per_km * increase in km);
      - best insertion in ONE empty route, if there are vehicles left (opens a route);
      - common carrier (fixed cost, independent of the routes);
      - bus integration (fixed cost, independent of the routes), if allowed and feasible.

    At each iteration the request with the largest regret (sum over j = 2..k of c_j - c_1) is assigned to its
    cheapest alternative. Requests whose cheapest alternative is a fixed one (carrier / bus) have a small
    regret, so they are left for last: by then the routes have changed and their insertion costs may have
    dropped (synergy) or confirmed that the fixed alternative is the right choice.

    The carrier is the safety net, so the construction never fails for lack of vehicles.

    Args:
        regret_k: number of alternatives considered by the regret (2 or 3).
        max_integrated: optional cap on the number of integrated requests.
        validate: runs Solution.validate() on the result.
    '''

    n = problem.requests

    routes: list[Route] = []
    carrier: set[int] = set()
    integrated: set[int] = set()
    total = 0.0

    # Alternatives that do not depend on the routes: computed once
    fixed: dict[int, list[tuple[float, int]]] = {}
    for k in range(1, n + 1):
        options = []
        if problem.carrier_feasible(k):
            options.append((problem.carrier_cost(k), CARRIER))
        if problem.integration_feasible(k):
            options.append((problem.integration_cost(k), INTEGRATION))
        fixed[k] = options

    # cache[k][route_index] = (cost, new_route). EMPTY is the (single) empty route candidate.
    cache: dict[int, dict[int, tuple[float, Route]]] = {k: {} for k in range(1, n + 1)}
    unserved = list(range(1, n + 1))

    def private_cost(delta: int) -> float:
        return problem.costs.private_per_km * delta / problem.scale

    if problem.vehicles > 0:
        empty = Route(problem, [])
        for k in unserved:
            if result := _best_insertion(problem, empty, k):
                cache[k][EMPTY] = (private_cost(result[0]), result[1])

    while unserved:
        selected = None  # (score, k, option)
        integration_closed = max_integrated is not None and len(integrated) >= max_integrated

        for k in unserved:
            options = [(c, ROUTE, idx) for idx, (c, _) in cache[k].items()]
            options += [(c, tag, None) for c, tag in fixed[k] if not (tag == INTEGRATION and integration_closed)]

            if not options:
                raise RuntimeError(f'Request {k} cannot be served by any alternative')

            options.sort(key=lambda o: (o[0], o[1], -1 if o[2] is None else o[2]))
            regret = _regret([o[0] for o in options], regret_k)

            score = (-regret, options[0][0], k)

            if selected is None or score < selected[0]:
                selected = (score, k, options[0])

        _, k, (cost, tag, route_index) = selected
        unserved.remove(k)
        total += cost

        if tag == CARRIER:
            carrier.add(k)
            del cache[k]
            continue

        if tag == INTEGRATION:
            integrated.add(k)
            del cache[k]
            continue

        # Private fleet: update the route and refresh only the cache column of that route
        new_route = cache[k][route_index][1]
        del cache[k]

        if route_index == EMPTY:
            routes.append(new_route)
            route_index = len(routes) - 1

            if len(routes) >= problem.vehicles:  # no more vehicles: the empty candidate is gone
                for entries in cache.values():
                    entries.pop(EMPTY, None)
        else:
            routes[route_index] = new_route

        for other in unserved:
            if result := _best_insertion(problem, new_route, other):
                cache[other][route_index] = (private_cost(result[0]), result[1])
            else:
                cache[other].pop(route_index, None)

    return Solution(problem, routes, carrier, integrated)