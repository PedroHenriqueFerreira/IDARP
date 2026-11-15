from __future__ import annotations

import random
import time
from typing import NamedTuple

import numpy as np

from src.lns.budget import Budget
from src.lns.neighborhoods import expand, fingerprint, relatedness, route_distances, route_savings
from src.lns.params import (
    EPSILON, K_NEAREST, K_OUTSIDE, LEARNING_RATE, MAX_IDLE, MIN_REMAINING, MIN_WEIGHT, NEIGHBORHOODS,
    RR_SHARE, START_SIZE,
)
from src.lns.ruin_recreate import ruin_recreate
from src.problem import Problem
from src.route import Route
from src.solution import Solution
from src.solver import Timeout, solve


class Neighborhood(NamedTuple):
    matrices: list[np.ndarray]
    key: bytes
    sub: Solution  # routes the model may change + carrier / bus requests
    fixed_routes: list[Route]
    fixed_requests: set[int]


class _Search:
    ''' CP-SAT phase: each call re-optimizes a subset of routes while the others stay fixed. '''

    def __init__(self, problem, solution, time_limit, max_integrated, workers, seed, solver_params, deterministic, log):
        self.problem, self.current = problem, solution
        self.time_limit, self.max_integrated, self.workers = time_limit, max_integrated, workers
        self.seed, self.solver_params, self.deterministic, self.log = seed, solver_params, deterministic, log

        self.rng = random.Random(seed)
        self.names = list(NEIGHBORHOODS)
        self.weights = dict.fromkeys(self.names, 1.0)
        self.exhausted: set[bytes] = set()  # neighborhoods without improvement for `current`

        # Derived from `current` and dropped whenever it changes
        self.built: dict[tuple[str, int | None, int], Neighborhood] = {}
        self.distances: dict[float, np.ndarray] = {}
        self.savings: dict[tuple[int, ...], dict[int, float]] = {}

        routes = max(1, len(self.used_routes()))
        call_time = min(max(time_limit / 60, 1.0), 30.0)
        self.budget = Budget(
            n=routes,
            iteration_time=call_time,
            max_iteration_time=max(time_limit / 4, call_time),
            size=min(START_SIZE, routes),
            cap=routes,
        )
        self.iteration = 0
        self.start = time.perf_counter()

    def used_routes(self) -> list[Route]:
        return [route for route in self.current.routes if route.nodes]

    def saving(self, route: Route) -> dict[int, float]:
        key = tuple(route.nodes)

        if key not in self.savings:
            self.savings[key] = route_savings(
                self.problem, route, self.current.outside_requests, self.current.carrier_requests
            )

        return self.savings[key]

    def neighborhood(self, name: str, seed_route: int | None, size: int) -> Neighborhood:
        ''' The `size` routes closest to `seed_route` (none: only idle vehicles) and the arcs they may use. '''

        cache_key = (name, seed_route, size)

        if cache_key in self.built:
            return self.built[cache_key]

        problem, n = self.problem, self.problem.requests
        time_weight, by_savings = NEIGHBORHOODS[name]
        routes = self.used_routes()

        if seed_route is None:
            chosen = []
        else:
            if time_weight not in self.distances:
                self.distances[time_weight] = route_distances(relatedness(problem, time_weight), routes, n)

            chosen = sorted(np.argsort(self.distances[time_weight][seed_route], kind='stable')[:size].tolist())

        selected = [routes[i] for i in chosen]
        fixed_routes = [route for i, route in enumerate(routes) if i not in set(chosen)]
        fixed_requests = {i for route in fixed_routes for i in route.nodes if i <= n}

        sub = Solution(problem, selected, self.current.carrier_requests, self.current.integrated_requests)
        savings = [self.saving(route) for route in [*selected, Route(problem, [])]] if by_savings else None

        matrices = expand(
            problem, sub, K_NEAREST, K_OUTSIDE, time_weight,
            savings=savings, max_idle=MAX_IDLE, reserved=len(fixed_routes),
        )

        self.built[cache_key] = Neighborhood(matrices, fingerprint(matrices), sub, fixed_routes, fixed_requests)

        return self.built[cache_key]

    def pick(self) -> tuple[str, Neighborhood] | None:
        ''' A not exhausted neighborhood: type drawn by weight, then the others; None if all are exhausted. '''

        first = self.rng.choices(self.names, [self.weights[x] for x in self.names])[0]
        routes = self.used_routes()
        size = min(self.budget.size, max(1, len(routes)))

        for name in [first] + [x for x in self.names if x != first]:
            if not routes:
                seeds = [None]  # only idle vehicles
            else:
                seeds = [0] if size >= len(routes) else list(range(len(routes)))

            self.rng.shuffle(seeds)

            for seed_route in seeds:
                found = self.neighborhood(name, seed_route, size)

                if found.key not in self.exhausted:
                    return name, found

        return None

    def attempt(self, found: Neighborhood, remaining: float) -> str:
        ''' Solves one neighborhood and updates the solution and the budget. Returns the outcome. '''

        tic = time.perf_counter()

        try:
            candidate = solve(
                self.problem, found.sub, found.matrices,
                time_limit=min(self.budget.iteration_time, remaining),
                max_integrated=self.max_integrated,
                workers=self.workers,
                seed=self.seed + self.iteration,
                params=self.solver_params,
                deterministic=self.deterministic,
                strict=True,
                fixed_requests=found.fixed_requests,
            )
        except Timeout:
            self.budget.on_timeout()
            return 'timeout'

        full = None

        if candidate is not None and candidate.cost < found.sub.cost - EPSILON:
            full = Solution(
                self.problem,
                [*found.fixed_routes, *candidate.routes],
                candidate.carrier_requests,
                candidate.integrated_requests,
            )

        if full is None or full.cost >= self.current.cost - EPSILON:
            self.exhausted.add(found.key)
            self.budget.grow()
            return 'exhausted'

        try:
            full.validate()
        except AssertionError:
            # The model accepted something the problem rejects: do not abort the search
            self.exhausted.add(found.key)
            self.budget.grow()
            return 'invalid'

        self.current = full
        self.exhausted.clear()
        self.built.clear()
        self.distances.clear()
        self.savings.clear()
        self.budget.n = max(1, len(self.used_routes()))
        self.budget.cap = min(self.budget.cap, self.budget.n)
        self.budget.on_improved(time.perf_counter() - tic)

        return 'improved'

    def report(self, name: str, outcome: str, extra: str = ''):
        if self.log:
            b = self.budget
            print(
                f'[{time.perf_counter() - self.start:7.1f}s] it {self.iteration:3d} {name:<16} '
                f'routes {b.size:2d}/{b.n:<2d} budget {b.iteration_time:5.1f}s {outcome:<9} '
                f'cost {self.current.cost:.4f}{extra}'
            )

    def run(self) -> Solution:
        while (remaining := self.time_limit - (time.perf_counter() - self.start)) > MIN_REMAINING:
            self.iteration += 1
            picked = self.pick()

            if picked is None:
                action = self.budget.on_stalled()
                self.report('-', action or 'done')

                if action is None:
                    break

                continue

            name, found = picked
            outcome = self.attempt(found, remaining)
            reward = 1.0 if outcome == 'improved' else 0.0
            self.weights[name] = max(MIN_WEIGHT, (1 - LEARNING_RATE) * self.weights[name] + LEARNING_RATE * reward)
            self.report(name, outcome, f' (fixed {len(found.fixed_routes)})')

        return self.current


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
    rr_share: float = RR_SHARE,
    log: bool = False,
) -> Solution:
    ''' Large neighborhood search from a feasible `solution`.

    1. `ruin_recreate` for a fraction `rr_share` of `time_limit` (0 = skip it);
    2. CP-SAT with the time left: each call re-optimizes a subset of routes (a seed route and its most related
       ones, the most promising carrier / bus requests and a few idle vehicles); the others stay fixed.

    The neighborhood type is drawn by weight (grows with its improvements); a (type, seed route) pair proved
    exhausted is not solved again for the same solution. Size and call time adapt (see `Budget`).
    '''

    if rr_share > 0 and problem.requests > 1:
        tic = time.perf_counter()
        solution = ruin_recreate(
            problem, solution, time_limit=rr_share * time_limit, max_integrated=max_integrated, seed=seed, log=log
        )
        time_limit -= time.perf_counter() - tic

    return _Search(
        problem, solution, time_limit, max_integrated, workers, seed, solver_params, deterministic, log
    ).run()
