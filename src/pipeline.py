from __future__ import annotations

import time

from src.config import Config
from src.heuristics import construct, local_search
from src.lns import lns
from src.problem import Problem
from src.solution import Solution

START_SHARE = 0.25  # fraction of `time_limit` for the local search of the starting solution


def _starting_solution(problem: Problem, config: Config, say, stats: dict) -> Solution:
    ''' Regret-2 construction polished by local search. '''

    tic = time.perf_counter()
    built = construct(problem, regret_k=2, max_integrated=config.max_integrated)
    stats['construct_cost'], stats['construct_time'] = built.cost, time.perf_counter() - tic

    tic = time.perf_counter()
    polished = local_search(
        problem, built, max_integrated=config.max_integrated, time_limit=START_SHARE * problem.requests, log=False
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

    tic = time.perf_counter()
    result = lns(
        problem,
        current,
        time_limit=problem.requests - (time.perf_counter() - start),
        max_integrated=config.max_integrated,
        workers=config.workers,
        seed=config.seed,
        log=config.log,
    )
    stats['lns_cost'], stats['lns_time'] = result.cost, time.perf_counter() - tic

    result.validate()
    say(f'final: cost {result.cost:.4f}')

    return result
