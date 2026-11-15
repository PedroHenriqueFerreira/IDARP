from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from src.solver import Timeout, feasible_arcs, solve
from src.problem import Problem
from src.solution import Route, Solution, best_insertion

LEARNING_RATE = 0.3  # how fast a type's weight follows its results
MIN_WEIGHT = 0.05  # every type keeps a chance of being chosen
EPSILON = 1e-9
MIN_SIZE = 1
START_SIZE = 2
FAST = 0.3  # a solve ending within this fraction of the iteration time lets the size grow
MIN_REMAINING = 0.5  # seconds: below this a solver call is not worth it
K_NEAREST = 5  # related requests added to the cluster of each request of a route
K_OUTSIDE = 5  # carrier / bus requests added to the cluster of each route
MAX_IDLE = 2  # idle vehicles (each seeded with an outside request) allowed in one neighborhood

# Neighborhood types: (time_weight of the relatedness, outside requests chosen by savings)
NEIGHBORHOODS = {
    'spatial': (0.1, False),
    'temporal': (2.0, False),
    'savings': (0.5, True),
    'savings-temporal': (2.0, True),
}


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
        for k in sorted(outside) if (insertion := best_insertion(problem, route, k))
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

    `solution` holds only the routes the model may change, plus every carrier / bus request. The requests of
    other routes are not part of it, so they never enter a cluster. `reserved` is the number of vehicles taken
    by routes kept outside the model; at most `max_idle` idle vehicles are added.

    A route's cluster is its requests, the `k` most related requests of each one and `k_outside` requests
    served by the carrier / bus (the closest ones, or with `savings` the ones that save the most).
    Idle vehicles are seeded with the most expensive outside request (or the best saving with `savings`).
    Each matrix keeps the feasible arcs among the cluster nodes and the arcs of the current route; the rest is -1.
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


# Adaptive search

@dataclass
class _Budget:
    ''' Neighborhood size and solver time per call, adapted to what the solver can handle. '''

    n: int
    iteration_time: float
    max_iteration_time: float
    size: int
    cap: int  # largest size worth trying (lowered by timeouts, relaxed by improvements)

    def more_time(self) -> bool:
        if self.iteration_time >= self.max_iteration_time:
            return False

        self.iteration_time = min(2 * self.iteration_time, self.max_iteration_time)
        self.cap = self.n

        return True

    def grow(self):
        self.size = min(self.size + 1, self.cap)

    def on_timeout(self):
        if self.size > MIN_SIZE:
            self.cap = self.size = self.size - 1
        else:
            self.more_time()

    def on_improved(self, spent: float):
        self.cap = min(self.cap + 1, self.n)

        if spent < FAST * self.iteration_time:
            self.grow()

    def on_stalled(self) -> str | None:
        ''' Every type is exhausted at this size: grow, else give the solver more time, else stop. '''

        if self.size < self.cap:
            self.size += 1
            return 'grow'

        return 'more time' if self.more_time() else None


def lns(
    problem: Problem,
    solution: Solution,
    *,
    time_limit: float = 600.0,
    max_integrated: int | None = None,
    workers: int = 0,
    seed: int = 0,
    solver_params: dict | None = None,
    deterministic: bool = False,
    log: bool = False,
) -> Solution:
    ''' Large neighborhood search from a feasible `solution`: repeats `expand` -> `solve` (CP-SAT).

    Each call re-optimizes only a subset of routes: a seed route, its `size - 1` most related routes, the
    carrier / bus requests that look most promising and a couple of idle vehicles. The other routes stay fixed
    (their cost is a constant and their requests are `fixed_requests` of the model). Hence:
      - `size` is the number of routes in the model (it adapts to what the solver can handle);
      - a neighborhood type is drawn with probability proportional to its weight (grows with its improvements);
      - a neighborhood is a (type, seed route) pair; exhausted ones (nothing cheaper, proved) are not solved
        again for this solution;
      - improved: the solution is replaced; size grows too if the solver was fast;
      - timeout: this neighborhood is hard: size shrinks and the failed size is capped; at the smallest size,
        the call time doubles;
      - all neighborhoods exhausted: size grows, then call time doubles (up to `time_limit / 4`), then it stops.

    The call time starts at `time_limit / 60`, clamped to [1, 30] s. With `deterministic=True` the solver
    limit is deterministic time (reproducible, slower).
    '''

    rng = random.Random(seed)
    n = problem.requests
    start = time.perf_counter()

    current = solution
    names = list(NEIGHBORHOODS)
    weights = dict.fromkeys(names, 1.0)
    exhausted: set[bytes] = set()  # neighborhoods without improvement for `current`

    # Everything below is derived from `current` and dropped whenever it changes
    built: dict[tuple[str, int | None, int], tuple] = {}
    distances: dict[float, np.ndarray] = {}
    savings_of: dict[tuple[int, ...], dict[int, float]] = {}

    def used_routes() -> list[Route]:
        return [route for route in current.routes if route.nodes]

    iteration_time = min(max(time_limit / 60, 1.0), 30.0)
    budget = _Budget(
        n=max(1, len(used_routes())),
        iteration_time=iteration_time,
        max_iteration_time=max(time_limit / 4, iteration_time),
        size=min(START_SIZE, max(1, len(used_routes()))),
        cap=max(1, len(used_routes())),
    )
    iteration = 0

    def saving(route: Route) -> dict[int, float]:
        key = tuple(route.nodes)

        if key not in savings_of:
            savings_of[key] = route_savings(problem, route, current.outside_requests, current.carrier_requests)

        return savings_of[key]

    def seeds_for(size: int, routes: list[Route]) -> list[int | None]:
        if not routes:
            return [None]  # only idle vehicles

        return [0] if size >= len(routes) else list(range(len(routes)))

    def neighborhood(name: str, seed_route: int | None, size: int) -> tuple:
        ''' (matrices, key, sub-solution, fixed routes, fixed requests) '''

        cache_key = (name, seed_route, size)

        if cache_key in built:
            return built[cache_key]

        time_weight, by_savings = NEIGHBORHOODS[name]
        routes = used_routes()

        if seed_route is None:
            chosen = []
        else:
            if time_weight not in distances:
                distances[time_weight] = route_distances(relatedness(problem, time_weight), routes, n)

            chosen = sorted(np.argsort(distances[time_weight][seed_route], kind='stable')[:size].tolist())

        selected = [routes[i] for i in chosen]
        fixed_routes = [route for i, route in enumerate(routes) if i not in set(chosen)]
        fixed_requests = {i for route in fixed_routes for i in route.nodes if i <= n}

        sub = Solution(problem, selected, current.carrier_requests, current.integrated_requests)
        route_savings_list = [saving(route) for route in [*selected, Route(problem, [])]] if by_savings else None

        matrices = expand(
            problem, sub, K_NEAREST, K_OUTSIDE, time_weight,
            savings=route_savings_list, max_idle=MAX_IDLE, reserved=len(fixed_routes),
        )

        built[cache_key] = (matrices, fingerprint(matrices), sub, fixed_routes, fixed_requests)

        return built[cache_key]

    def report(name: str, outcome: str, extra: str = ''):
        if log:
            print(
                f'[{time.perf_counter() - start:7.1f}s] it {iteration:3d} {name:<16} routes {budget.size:2d}/{budget.n:<2d} '
                f'budget {budget.iteration_time:5.1f}s {outcome:<9} cost {current.cost:.4f}{extra}'
            )

    while (remaining := time_limit - (time.perf_counter() - start)) > MIN_REMAINING:
        iteration += 1
        first = rng.choices(names, [weights[x] for x in names])[0]
        chosen = None
        size = min(budget.size, max(1, len(used_routes())))

        for name in [first] + [x for x in names if x != first]:
            seeds = seeds_for(size, used_routes())
            rng.shuffle(seeds)

            for seed_route in seeds:
                found = neighborhood(name, seed_route, size)

                if found[1] not in exhausted:
                    chosen = (name, found)
                    break

            if chosen is not None:
                break

        if chosen is None:
            action = budget.on_stalled()
            report('-', action or 'done')

            if action is None:
                break

            continue

        name, (matrices, key, sub, fixed_routes, fixed_requests) = chosen
        tic = time.perf_counter()
        info = f' (fixed {len(fixed_routes)})'

        try:
            candidate = solve(
                problem, sub, matrices,
                time_limit=min(budget.iteration_time, remaining),
                max_integrated=max_integrated,
                workers=workers,
                seed=seed + iteration,
                params=solver_params,
                deterministic=deterministic,
                strict=True,
                fixed_requests=fixed_requests,
            )
        except Timeout:
            outcome = 'timeout'
            budget.on_timeout()
        else:
            full = None

            if candidate is not None and candidate.cost < sub.cost - EPSILON:
                full = Solution(
                    problem,
                    [*fixed_routes, *candidate.routes],
                    candidate.carrier_requests,
                    candidate.integrated_requests,
                )

            if full is not None and full.cost < current.cost - EPSILON:
                try:
                    full.validate()
                except AssertionError:
                    # The model accepted something the problem rejects: do not abort the search
                    outcome = 'invalid'
                    exhausted.add(key)
                    budget.grow()
                else:
                    outcome = 'improved'
                    current = full
                    exhausted.clear()
                    built.clear()
                    distances.clear()
                    savings_of.clear()
                    budget.n = max(1, len(used_routes()))
                    budget.cap = min(budget.cap, budget.n)
                    budget.on_improved(time.perf_counter() - tic)
            else:
                outcome = 'exhausted'
                exhausted.add(key)
                budget.grow()

        reward = 1.0 if outcome == 'improved' else 0.0
        weights[name] = max(MIN_WEIGHT, (1 - LEARNING_RATE) * weights[name] + LEARNING_RATE * reward)
        report(name, outcome, info)

    return current