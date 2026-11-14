from __future__ import annotations

import numpy as np

from src.problem import Problem
from src.solution import Solution


def _feasible_arcs(problem: Problem) -> np.ndarray:
    ''' Boolean matrix of the arcs that can belong to some feasible route (necessary conditions). '''

    n = problem.requests
    size = 2 * n + 1

    times = problem.times[:size, :size]
    service = np.array([node.service_time for node in problem.nodes])
    ready = np.array([node.ready_time for node in problem.nodes])
    due = np.array([node.due_time for node in problem.nodes])

    pickups = np.arange(1, n + 1)
    deliveries = pickups + n

    # Time windows
    arcs = ready[:, None] + service[:, None] + times <= due[None, :]

    # Ride time: if x is visited right after pickup r or right before delivery r, request r rides at least this long
    via = service[pickups][:, None] + times[pickups, :] + service[None, :] + times[:, deliveries].T
    ride_ok = via <= problem.max_ride_time
    direct_ok = service[pickups] + times[pickups, deliveries] <= problem.max_ride_time
    ride_ok[np.arange(n), pickups] = direct_ok
    ride_ok[np.arange(n), deliveries] = direct_ok

    arcs[pickups, :] &= ride_ok
    arcs[:, deliveries] &= ride_ok.T

    # Route duration: a route using arc (i, j) lasts at least depot -> i -> j -> depot
    duration = times[0, :, None] + service[:, None] + times + service[None, :] + times[:, 0][None, :]
    arcs &= duration <= problem.max_duration_route

    # Routes start with a pickup and end with a delivery
    arcs[0, deliveries] = False
    arcs[pickups, 0] = False
    arcs[deliveries, pickups] = False

    if problem.capacity < 2:
        arcs[np.ix_(pickups, pickups)] = False

    np.fill_diagonal(arcs, False)

    return arcs


def _relatedness(problem: Problem, arcs: np.ndarray, time_weight: float) -> np.ndarray:
    ''' Request-to-request relatedness (lower is closer): travel time between pickups and between deliveries
    plus the weighted difference of their time windows. Requests with no feasible arc between them are inf. '''

    n = problem.requests
    times = problem.times[:2 * n + 1, :2 * n + 1]
    pickups = np.arange(1, n + 1)
    deliveries = pickups + n

    middle = np.array([(node.ready_time + node.due_time) / 2 for node in problem.nodes])
    spatial = times[np.ix_(pickups, pickups)] + times[np.ix_(deliveries, deliveries)]
    temporal = (
        np.abs(middle[pickups][:, None] - middle[pickups][None, :])
        + np.abs(middle[deliveries][:, None] - middle[deliveries][None, :])
    )

    relatedness = spatial + time_weight * temporal
    relatedness = (relatedness + relatedness.T) / 2

    pp = arcs[np.ix_(pickups, pickups)]
    pd = arcs[np.ix_(pickups, deliveries)]
    dp = arcs[np.ix_(deliveries, pickups)]
    dd = arcs[np.ix_(deliveries, deliveries)]

    relatedness[~(pp | pp.T | pd | pd.T | dp | dp.T | dd | dd.T)] = np.inf
    np.fill_diagonal(relatedness, np.inf)

    return relatedness

def _cluster(
    relatedness: np.ndarray,
    nearest: dict[int, list[int]],
    active: set[int],
    outside: set[int],
    k_outside: int,
) -> set[int]:
    cluster = set(active)

    for request in active:
        cluster.update(nearest[request])

    candidates = sorted(outside - cluster)

    if candidates and k_outside > 0:
        closeness = relatedness[np.ix_([r - 1 for r in active], [r - 1 for r in candidates])].min(axis=0)

        for index in np.argsort(closeness)[:k_outside]:
            if np.isfinite(closeness[index]):
                cluster.add(candidates[index])

    return cluster


def _matrix(problem: Problem, arcs: np.ndarray, requests: set[int], route_nodes: list[int]) -> np.ndarray:
    ''' Distance matrix with -1 on the arcs that are not allowed. '''

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
    k: int = 5,
    k_outside: int = 5,
    time_weight: float = 0.5,
    unused_vehicles: bool = True,
) -> list[np.ndarray]:
    ''' Restricted distance matrices around a solution, one per used route and, optionally, per unused vehicle.

    The cluster of a route is its requests, the `k` most related requests of each one and the `k_outside`
    closest requests currently served by the carrier or the bus. An unused vehicle is seeded with the most
    expensive request outside the private fleet. Each matrix keeps the feasible arcs among the cluster nodes,
    the depot arcs of the cluster and the arcs of the current route; the rest is -1.
    '''

    n = problem.requests

    arcs = _feasible_arcs(problem)
    relatedness = _relatedness(problem, arcs, time_weight)
    order = np.argsort(relatedness, axis=1)[:, :k]
    nearest = {
        r + 1: [int(s) + 1 for s in order[r] if np.isfinite(relatedness[r, s])]
        for r in range(n)
    }

    outside = set(solution.carrier_requests) | set(solution.integrated_requests)
    used = [route.nodes for route in solution.routes if route.nodes]

    clusters = [
        _cluster(relatedness, nearest, {id if id <= n else id - n for id in nodes}, outside, k_outside)
        for nodes in used
    ]

    if unused_vehicles:
        seeds = sorted(
            outside,
            key=lambda r: problem.carrier_cost(r) if r in solution.carrier_requests else problem.integration_cost(r),
            reverse=True,
        )[:problem.vehicles - len(used)]
        clusters += [_cluster(relatedness, nearest, {seed}, outside, k_outside) for seed in seeds]

    return [
        _matrix(problem, arcs, cluster, used[i] if i < len(used) else [])
        for i, cluster in enumerate(clusters)
    ]