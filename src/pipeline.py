from __future__ import annotations

import time
from dataclasses import dataclass

from src.heuristics import construct, local_search
from src.lns import lns
from src.problem import Problem
from src.solution import Solution

@dataclass(frozen=True)
class Config:
    ''' Pipeline parameters. `time_limit` (seconds) covers the whole run. '''

    time_limit: float = 180.0
    max_integrated: int | None = None  # cap on requests served by bus integration
    start_share: float = 0.25  # fraction of `time_limit` for the starting phase (local search of every start)
    workers: int = 0  # CP-SAT workers (0 = all cores)
    seed: int = 0
    deterministic: bool = False  # CP-SAT deterministic time (reproducible, slower)
    solver_params: dict | None = None  # extra CpSolverParameters fields
    log: bool = True


def _starting_solution(problem: Problem, config: Config, say, stats: dict) -> Solution:
    ''' Best of several regret-k constructions, each polished by local search. '''

    time_limit = problem.requests

    budget = config.start_share * time_limit
    
    tic = time.perf_counter()
    built = construct(problem, regret_k=2, max_integrated=config.max_integrated)
    stats['construct_cost'], stats['construct_time'] = built.cost, time.perf_counter() - tic

    tic = time.perf_counter()
    polished = local_search(
        problem, built, max_integrated=config.max_integrated, time_limit=budget, log=False
    )
    stats['local_search_cost'], stats['local_search_time'] = polished.cost, time.perf_counter() - tic
    
    say(f'regret-2: construct {built.cost:.4f} -> local search {polished.cost:.4f}')

    return polished


def run(problem: Problem, config: Config = Config(), stats: dict | None = None) -> Solution:
    ''' construct -> local search -> LNS. Returns a validated solution, never worse than the starting one.

    If `stats` is given it is filled with the cost and time of each stage. '''

    stats = stats if stats is not None else {}

    start = time.perf_counter()

    def say(message: str):
        if config.log:
            print(f'[{time.perf_counter() - start:7.1f}s] {message}')

    current = _starting_solution(problem, config, say, stats)
    current.validate()
    say(f'start: cost {current.cost:.4f}')

    time_limit = problem.requests

    tic = time.perf_counter()
    result = lns(
        problem,
        current,
        time_limit=time_limit - (time.perf_counter() - start),
        max_integrated=config.max_integrated,
        workers=config.workers,
        seed=config.seed,
        solver_params=config.solver_params,
        deterministic=config.deterministic,
        log=config.log,
    )

    stats['lns_cost'], stats['lns_time'] = result.cost, time.perf_counter() - tic

    result.validate()
    say(f'final: cost {result.cost:.4f}')

    return result