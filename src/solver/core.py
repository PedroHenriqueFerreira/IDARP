from __future__ import annotations

import os
from typing import Collection

import numpy as np
from ortools.sat.python import cp_model

from src.problem import Problem
from src.route import Route
from src.solution import Solution
from src.solver.context import context
from src.solver.model import IdarpModel
from src.solver.scope import build_scope


class Timeout(RuntimeError):
    ''' The search stopped at the time limit without a solution and without proving that none exists. '''


def solve(
    problem: Problem,
    initial_solution: Solution,
    matrices: list[np.ndarray],
    time_limit: float = 120.0,
    max_integrated: int | None = None,
    workers: int = 0,
    gap: float = 0.0,
    log: bool = False,
    upper_bound_cut: bool = True,
    params: dict | None = None,
    seed: int = 0,
    deterministic: bool = True,
    strict: bool = False,
    hint: bool = True,
    fixed_requests: Collection[int] = (),
) -> Solution | None:
    ''' CP-SAT model of the IDARP, warm-started with `initial_solution`.

    Vehicle v can only use the arcs allowed in `matrices[v]` (see `lns.expand`); the first matrices must be
    the routes of `initial_solution`. `fixed_requests` are served by routes outside the model (untouched
    by a route-subset LNS): they get no variables and `initial_solution` must not contain them.

    - `workers`: 0 = all cores; `gap`: relative gap to stop early; `params`: any CpSolverParameters field;
    - `upper_bound_cut`: adds "cost <= initial cost";
    - `deterministic`: reproducible search with the limit in deterministic time (may differ from `time_limit`).

    With `strict=True` it looks for a solution cheaper than `initial_solution` and returns:
      - the cheaper solution;
      - `None` if it is proved that nothing is cheaper inside these arcs (neighborhood exhausted);
      - raises `Timeout` if the limit is reached without finding anything cheaper.
    '''

    fixed = frozenset(fixed_requests)
    ctx = context(problem)
    scope = build_scope(problem, ctx, matrices, fixed)
    built = IdarpModel(problem, ctx, scope, fixed, max_integrated)

    plan = built.plan(initial_solution, len(matrices))

    if hint:
        built.add_hint(initial_solution, plan)

    initial_cost = built.add_upper_bound(initial_solution, plan, strict) if upper_bound_cut or strict else None

    solver = _configure(time_limit, workers, gap, log, params, seed, deterministic)
    status = solver.Solve(built.model)

    if status == cp_model.INFEASIBLE and strict:
        return None  # nothing strictly better than the initial solution inside these arcs

    if status == cp_model.UNKNOWN:
        raise Timeout('Time limit reached without a solution')

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError('No solution found (the model is infeasible)')

    if strict and initial_cost is not None and round(solver.ObjectiveValue()) >= initial_cost:
        if status == cp_model.OPTIMAL:
            return None  # proved: nothing cheaper than the initial solution inside these arcs

        raise Timeout('Time limit reached without improving the initial solution')

    return _extract(problem, built, solver)


def _configure(time_limit, workers, gap, log, params, seed, deterministic) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = workers or os.cpu_count() or 8

    if deterministic:
        # Wall-clock limits make the stopping point depend on machine load: the limit is deterministic
        # time and the wall-clock limit is only a safety net
        solver.parameters.interleave_search = True
        solver.parameters.max_deterministic_time = time_limit
        solver.parameters.max_time_in_seconds = 10 * time_limit
    else:
        solver.parameters.max_time_in_seconds = time_limit

    solver.parameters.relative_gap_limit = gap
    solver.parameters.log_search_progress = log
    # `repair_hint` is off: it can abort the process when the repair model is unsatisfiable
    solver.parameters.hint_conflict_limit = 100
    solver.parameters.linearization_level = 2

    for name, value in (params or {}).items():
        setattr(solver.parameters, name, value)

    return solver


def _extract(problem: Problem, built: IdarpModel, solver: cp_model.CpSolver) -> Solution:
    routes: list[Route] = []

    for v in built.scope.vehicles:
        successor = {i: j for i, j, var in built.arcs_of[v] if solver.BooleanValue(var)}

        if 0 not in successor:
            continue

        nodes, current = [], successor[0]

        while current != 0:
            nodes.append(current)
            current = successor[current]

        routes.append(Route(problem, nodes))

    return Solution(
        problem,
        routes,
        {k for k, var in built.carrier.items() if solver.BooleanValue(var)},
        {k for k, var in built.bus.items() if solver.BooleanValue(var)},
    )
