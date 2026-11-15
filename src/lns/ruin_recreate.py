from __future__ import annotations

import math
import random
import time
from typing import Callable

import numpy as np

from src.heuristics import EMPTY, _regret, _without, local_search
from src.lns.neighborhoods import relatedness
from src.lns.params import (
    EPSILON, RR_COOLING, RR_MAX_FRACTION, RR_MAX_REMOVE, RR_MEMO_LIMIT, RR_MIN_REMOVE, RR_RANDOMNESS,
    RR_TEMPERATURE,
)
from src.problem import CARRIER, INTEGRATION, ROUTE, Problem
from src.route import Route
from src.solution import Solution


class _Memo:
    ''' Insertions and removals depend only on (route nodes, request): routes repeat between iterations. '''

    def __init__(self, problem: Problem):
        self.problem = problem
        self.insertions: dict[tuple[tuple[int, ...], int], tuple[float, Route] | None] = {}
        self.removals: dict[tuple[tuple[int, ...], int], Route | None] = {}

    def insert(self, route: Route, k: int) -> tuple[float, Route] | None:
        key = (tuple(route.nodes), k)

        if key not in self.insertions:
            if len(self.insertions) > RR_MEMO_LIMIT:
                self.insertions.clear()

            self.insertions[key] = route.best_insertion(k)

        return self.insertions[key]

    def remove(self, route: Route, k: int) -> Route | None:
        key = (tuple(route.nodes), k)

        if key not in self.removals:
            if len(self.removals) > RR_MEMO_LIMIT:
                self.removals.clear()

            self.removals[key] = _without(self.problem, route, k)

        return self.removals[key]


def _ruin(
    memo: _Memo, routes: list[Route], carrier: set[int], integrated: set[int],
    related_order: Callable[[int], list[int]], count: int, rng: random.Random,
) -> list[int]:
    ''' Shaw removal: a random request, then requests related to an already removed one. Private requests leave
    their route (an emptied route stays in `routes` until the end); the ones that cannot leave (the time matrix
    may break the triangle inequality) are skipped. '''

    n = memo.problem.requests
    removed: list[int] = []
    tried: set[int] = set()

    def take(k: int) -> bool:
        tried.add(k)

        if k in carrier or k in integrated:
            carrier.discard(k)
            integrated.discard(k)
            removed.append(k)

            return True

        index = next(i for i, route in enumerate(routes) if k in route.nodes)

        if (new_route := memo.remove(routes[index], k)) is None:
            return False

        routes[index] = new_route
        removed.append(k)

        return True

    while not removed and len(tried) < n:
        take(rng.choice([k for k in range(1, n + 1) if k not in tried]))

    while len(removed) < count and len(tried) < n:
        pivot = rng.choice(removed) if removed else None
        candidates = [k for k in related_order(pivot) if k not in tried] if pivot else []

        if not candidates:
            candidates = [k for k in range(1, n + 1) if k not in tried]

        take(candidates[int(len(candidates) * rng.random() ** RR_RANDOMNESS)])

    return removed


def _recreate(
    memo: _Memo, routes: list[Route], carrier: set[int], integrated: set[int],
    requests: list[int], regret_k: int, max_integrated: int | None,
) -> bool:
    ''' Regret-k reinsertion (same scheme as `construct`, over a partial solution).
    Returns False if a request has no alternative at all. '''

    problem = memo.problem
    per_km = problem.costs.private_per_km
    fixed = {k: problem.fixed_options(k) for k in requests}
    unserved = list(requests)
    cache: dict[int, dict[int, tuple[float, Route]]] = {k: {} for k in unserved}

    def refresh(k: int, index: int, route: Route):
        if result := memo.insert(route, k):
            cache[k][index] = (per_km * result[0], result[1])
        else:
            cache[k].pop(index, None)

    for index, route in enumerate(routes):
        for k in unserved:
            refresh(k, index, route)

    if len(routes) < problem.vehicles:
        empty = Route(problem, [])

        for k in unserved:
            refresh(k, EMPTY, empty)

    while unserved:
        closed = max_integrated is not None and len(integrated) >= max_integrated
        selected = None

        for k in unserved:
            options = [(cost, ROUTE, index) for index, (cost, _) in cache[k].items()]
            options += [(cost, tag, None) for cost, tag in fixed[k] if not (tag == INTEGRATION and closed)]

            if not options:
                return False

            options.sort(key=lambda o: (o[0], o[1], -1 if o[2] is None else o[2]))
            score = (-_regret([o[0] for o in options], regret_k), options[0][0], k)

            if selected is None or score < selected[0]:
                selected = (score, k, options[0])

        _, k, (_, tag, index) = selected
        unserved.remove(k)
        entries = cache.pop(k)

        if tag == CARRIER:
            carrier.add(k)
        elif tag == INTEGRATION:
            integrated.add(k)
        else:
            new_route = entries[index][1]

            if index == EMPTY:
                routes.append(new_route)
                index = len(routes) - 1

                if len(routes) >= problem.vehicles:
                    for other in cache.values():
                        other.pop(EMPTY, None)
            else:
                routes[index] = new_route

            for other in unserved:
                refresh(other, index, new_route)

    return True


def ruin_recreate(
    problem: Problem,
    solution: Solution,
    *,
    time_limit: float,
    max_integrated: int | None = None,
    seed: int = 0,
    log: bool = False,
) -> Solution:
    ''' Ruin and recreate with simulated annealing: remove a related group of requests (Shaw) and reinsert them
    by regret-k (k in 1..3 at random). A worse solution is accepted with probability exp(-delta / T), T cooling
    geometrically with the elapsed time. The best solution is polished by `local_search` at the end.
    Never returns a solution worse than the given one. '''

    start = time.perf_counter()
    rng = random.Random(seed * 7919 + 1)
    n = problem.requests
    memo = _Memo(problem)
    related = relatedness(problem, 0.5)
    orders: dict[int, list[int]] = {}

    def related_order(r: int) -> list[int]:
        if r not in orders:
            row = related[r - 1]
            orders[r] = [int(s) + 1 for s in np.argsort(row, kind='stable') if np.isfinite(row[s])]

        return orders[r]

    def total(routes: list[Route], carrier: set[int], integrated: set[int]) -> float:
        return (
            sum(route.cost for route in routes)
            + sum(problem.carrier_cost(k) for k in carrier)
            + sum(problem.integration_cost(k) for k in integrated)
        )

    routes = [route for route in solution.routes if route.nodes]
    carrier, integrated = set(solution.carrier_requests), set(solution.integrated_requests)
    cost = best_cost = solution.cost
    best = solution

    top = max(RR_MIN_REMOVE, min(RR_MAX_REMOVE, int(RR_MAX_FRACTION * n)))
    t0 = max(RR_TEMPERATURE * cost / max(n, 1), EPSILON)
    iteration = accepted = 0

    while (elapsed := time.perf_counter() - start) < time_limit and n > 1:
        iteration += 1
        temperature = t0 * RR_COOLING ** (elapsed / time_limit)

        new_routes, new_carrier, new_integrated = list(routes), set(carrier), set(integrated)
        removed = _ruin(
            memo, new_routes, new_carrier, new_integrated, related_order,
            rng.randint(RR_MIN_REMOVE, min(top, n)), rng,
        )

        if not removed or not _recreate(
            memo, new_routes, new_carrier, new_integrated, removed, rng.choice((1, 2, 2, 3)), max_integrated
        ):
            continue

        new_routes = [route for route in new_routes if route.nodes]
        new_cost = total(new_routes, new_carrier, new_integrated)
        delta = new_cost - cost

        if delta < -EPSILON or (delta > EPSILON and rng.random() < math.exp(-delta / temperature)):
            routes, carrier, integrated, cost = new_routes, new_carrier, new_integrated, new_cost
            accepted += 1

            if cost < best_cost - EPSILON:
                best_cost = cost
                best = Solution(problem, routes, carrier, integrated)

                if log:
                    print(f'[{time.perf_counter() - start:7.1f}s] rr it {iteration:6d} improved  cost {best_cost:.4f}')

    result = local_search(problem, best, max_integrated=max_integrated, time_limit=max(1.0, 0.05 * time_limit))
    result.validate()

    if log:
        print(
            f'[{time.perf_counter() - start:7.1f}s] rr done: {iteration} iterations, {accepted} accepted, '
            f'cost {solution.cost:.4f} -> {result.cost:.4f}'
        )

    return result if result.cost < solution.cost - EPSILON else solution
