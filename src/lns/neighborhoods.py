from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Callable

import numpy as np

from src.lns.params import K_NEAREST, K_OUTSIDE
from src.problem import Problem
from src.route import Route
from src.solution import Solution
from src.solver import feasible_arcs


@lru_cache(maxsize=16)
def relatedness(problem: Problem, time_weight: float) -> np.ndarray:
    ''' Request-to-request relatedness (lower = closer): travel time between pickups and between deliveries
    plus the weighted time-window difference. Requests without any feasible arc between them are inf. '''

    n = problem.requests
    arcs = feasible_arcs(problem)
    times = problem.times[:2 * n + 1, :2 * n + 1]
    pickups = np.arange(1, n + 1)
    deliveries = pickups + n

    middle = np.array([(node.ready_time + node.due_time) / 2 for node in problem.nodes])
    spatial = times[np.ix_(pickups, pickups)] + times[np.ix_(deliveries, deliveries)]
    temporal = (
        np.abs(middle[pickups][:, None] - middle[pickups][None, :])
        + np.abs(middle[deliveries][:, None] - middle[deliveries][None, :])
    )

    result = spatial + time_weight * temporal
    result = (result + result.T) / 2

    pp = arcs[np.ix_(pickups, pickups)]
    pd = arcs[np.ix_(pickups, deliveries)]
    dp = arcs[np.ix_(deliveries, pickups)]
    dd = arcs[np.ix_(deliveries, deliveries)]

    result[~(pp | pp.T | pd | pd.T | dp | dp.T | dd | dd.T)] = np.inf
    np.fill_diagonal(result, np.inf)

    return result


def route_savings(problem: Problem, route: Route, outside: set[int], carrier: set[int]) -> dict[int, float]:
    ''' Saving of moving each carrier / bus request into `route` (current cost - cheapest insertion).
    Requests that do not fit are left out. '''

    return {
        k: problem.outside_cost(k, carrier) - problem.costs.private_per_km * insertion[0]
        for k in sorted(outside) if (insertion := route.best_insertion(k))
    }


def outside_savings(problem: Problem, solution: Solution) -> list[dict[int, float]]:
    ''' `route_savings` of every used route (in `solution.routes` order) and of an empty vehicle (last). '''

    outside = solution.outside_requests
    routes = [route for route in solution.routes if route.nodes] + [Route(problem, [])]

    return [route_savings(problem, route, outside, solution.carrier_requests) for route in routes]


def route_distances(related: np.ndarray, routes: list[Route], n: int) -> np.ndarray:
    ''' Route-to-route relatedness (lower = closer): mean over the requests of one route of the relatedness to
    the closest request of the other, symmetrized. Pairs without any related request get a large finite value. '''

    finite = related[np.isfinite(related)]
    far = 2 * float(finite.max()) if finite.size else 1.0
    safe = np.where(np.isfinite(related), related, far)

    members = [np.array(sorted({(i if i <= n else i - n) - 1 for i in route.nodes})) for route in routes]
    size = len(routes)
    result = np.zeros((size, size))

    for a in range(size):
        for b in range(a + 1, size):
            block = safe[np.ix_(members[a], members[b])]
            result[a, b] = result[b, a] = (block.min(axis=1).mean() + block.min(axis=0).mean()) / 2

    np.fill_diagonal(result, -1.0)  # a route is always the closest to itself

    return result


def _cluster(
    related: np.ndarray,
    nearest: Callable[[int], list[int]],
    active: set[int],
    outside: set[int],
    k_outside: int,
    savings: dict[int, float] | None = None,
) -> set[int]:
    ''' Requests of a route, their nearest ones and the `k_outside` most promising outside requests. '''

    cluster = set(active)

    for request in active:
        cluster.update(nearest(request))

    candidates = sorted(outside - cluster)

    if candidates and k_outside > 0:
        closeness = related[np.ix_([r - 1 for r in active], [r - 1 for r in candidates])].min(axis=0)
        gain = [savings.get(r, 0.0) if savings else 0.0 for r in candidates]

        # Positive savings first (largest first), then the closest
        ranked = sorted(
            range(len(candidates)),
            key=lambda i: (0, -gain[i], closeness[i]) if gain[i] > 0 else (1, closeness[i], 0.0),
        )

        for index in ranked[:k_outside]:
            if gain[index] > 0 or np.isfinite(closeness[index]):
                cluster.add(candidates[index])

    return cluster


def _matrix(problem: Problem, arcs: np.ndarray, requests: set[int], route_nodes: list[int]) -> np.ndarray:
    ''' Distance matrix with -1 on the arcs not allowed. '''

    n = problem.requests
    size = 2 * n + 1
    pickups = np.array(sorted(requests))
    deliveries = pickups + n
    nodes = np.concatenate([pickups, deliveries])

    allowed = np.zeros((size, size), dtype=bool)
    allowed[np.ix_(nodes, nodes)] = arcs[np.ix_(nodes, nodes)]
    allowed[0, pickups] = arcs[0, pickups]
    allowed[deliveries, 0] = arcs[deliveries, 0]

    if route_nodes:
        path = [0, *route_nodes, 0]
        allowed[path[:-1], path[1:]] = True

    matrix = np.where(allowed, problem.distances[:size, :size], -1).astype(np.int64)
    np.fill_diagonal(matrix, 0)

    return matrix


def expand(
    problem: Problem,
    solution: Solution,
    k: int = K_NEAREST,
    k_outside: int = K_OUTSIDE,
    time_weight: float = 0.5,
    unused_vehicles: bool = True,
    savings: list[dict[int, float]] | None = None,
    max_idle: int | None = None,
    reserved: int = 0,
) -> list[np.ndarray]:
    ''' Restricted distance matrices around a (sub-)solution: one per used route and, optionally, per idle vehicle.

    `solution` holds only the routes the model may change, plus every carrier / bus request. `reserved` is the
    number of vehicles taken by routes kept outside the model; at most `max_idle` idle vehicles are added.

    A route's cluster is its requests, the `k` most related requests of each one and `k_outside` carrier / bus
    requests (the closest ones, or with `savings` the ones that save the most). Idle vehicles are seeded with the
    most expensive outside request (or the best saving). Each matrix keeps the feasible arcs among the cluster
    nodes and the arcs of the current route; the rest is -1.
    '''

    n = problem.requests
    arcs = feasible_arcs(problem)
    related = relatedness(problem, time_weight)

    outside = solution.outside_requests
    used = [route.nodes for route in solution.routes if route.nodes]

    pool = np.zeros(n, dtype=bool)  # requests this model may touch
    pool[[r - 1 for r in outside]] = True

    for nodes in used:
        pool[[(i if i <= n else i - n) - 1 for i in nodes]] = True

    memo: dict[int, list[int]] = {}

    def nearest(r: int) -> list[int]:
        if r not in memo:
            row = np.where(pool, related[r - 1], np.inf)
            memo[r] = [int(s) + 1 for s in np.argsort(row)[:k] if np.isfinite(row[s])]

        return memo[r]

    clusters = [
        _cluster(
            related, nearest, {i if i <= n else i - n for i in nodes}, outside, k_outside,
            savings[index] if savings else None,
        )
        for index, nodes in enumerate(used)
    ]

    if unused_vehicles:
        def seed_rank(r: int) -> float:
            if savings:
                return savings[-1].get(r, -np.inf)

            return problem.outside_cost(r, solution.carrier_requests)

        idle = max(0, problem.vehicles - reserved - len(used))

        if max_idle is not None:
            idle = min(idle, max_idle)

        seeds = sorted(outside, key=seed_rank, reverse=True)[:idle]
        clusters += [_cluster(related, nearest, {seed}, outside, k_outside) for seed in seeds]

    return [
        _matrix(problem, arcs, cluster, used[index] if index < len(used) else [])
        for index, cluster in enumerate(clusters)
    ]


def fingerprint(matrices: list[np.ndarray]) -> bytes:
    ''' Identity of a neighborhood: the arcs allowed to each vehicle (vehicle order is irrelevant). '''

    return b''.join(sorted(
        hashlib.blake2b(np.packbits(matrix >= 0).tobytes(), digest_size=16).digest()
        for matrix in matrices
    ))
