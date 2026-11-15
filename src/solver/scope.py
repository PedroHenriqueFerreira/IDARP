from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Collection

import numpy as np

from src.problem import Problem
from src.solver.arcs import restrict_requests, tighten_windows


@dataclass
class Scope:
    ''' What the model may use: node time domains, vehicles (grouped by identical arcs) and requests. '''

    lo: list[int]  # arrival time domain of each node
    hi: list[int]
    group_of: list[int]  # group of each matrix (vehicle)
    group_members: list[list[int]]
    kept_members: list[list[int]]  # members kept per group (surplus vehicles are dropped)
    vehicles: list[int]
    mask_of: dict[int, np.ndarray]  # allowed arcs per vehicle
    eligible: dict[int, list[int]]  # requests each vehicle may serve
    active: list[int]  # requests some vehicle can serve
    active_nodes: list[int]


def build_scope(
    problem: Problem, ctx: SimpleNamespace, matrices: list[np.ndarray], fixed: Collection[int]
) -> Scope:
    ''' Restricts arcs, time windows and requests to a fixpoint and groups interchangeable vehicles. '''

    n = problem.requests
    size = 2 * n + 1
    data, chain, static = ctx.data, ctx.chain, ctx.static
    ready, due, service, T = data.ready, data.due, data.service, data.times

    # Allowed arcs per vehicle (distinct matrices are processed once)
    raw_index: dict[bytes, int] = {}
    raw_masks: list[np.ndarray] = []
    raw_of: list[int] = []

    for matrix in matrices:
        mask = matrix[:size, :size] >= 0
        key = mask.tobytes()

        if key not in raw_index:
            raw_index[key] = len(raw_masks)
            raw_masks.append(mask)

        raw_of.append(raw_index[key])

    def restrict_all(lb, ub, alive):
        arcs = static & (lb[:, None] + service[:, None] + T <= ub[None, :])
        return [restrict_requests(mask & arcs, n, alive) for mask in raw_masks]

    # Fixpoint: arcs -> tighter time windows -> fewer arcs / requests -> ...
    lb, ub, alive = ready.copy(), due.copy(), data.request_ok.copy()

    for k in fixed:
        alive[k - 1] = False  # served by an untouched route: never part of the model

    for _ in range(10):
        results = restrict_all(lb, ub, alive)
        union = np.zeros((size, size), dtype=bool)

        for sub, _eligible in results:
            union |= sub

        new_lb, new_ub = tighten_windows(ready, due, service, T, union, n, data.max_ride, chain)
        new_alive = alive & (new_lb[1:n + 1] <= new_ub[1:n + 1]) & (new_lb[n + 1:] <= new_ub[n + 1:])
        stable = np.array_equal(new_alive, alive) and np.array_equal(new_lb, lb) and np.array_equal(new_ub, ub)
        lb, ub, alive = new_lb, new_ub, new_alive

        if stable:
            break

    results = restrict_all(lb, ub, alive)

    lo, hi = ready.tolist(), due.tolist()
    for i in range(1, size):
        if alive[(i if i <= n else i - n) - 1]:
            lo[i], hi[i] = int(lb[i]), int(ub[i])

    # Groups of interchangeable vehicles (same allowed arcs); surplus vehicles of a group are dropped,
    # since a group never uses more vehicles than it has eligible requests
    group_index: dict[bytes, int] = {}
    group_of: list[int] = []
    group_members: list[list[int]] = []
    group_eligible: list[list[int]] = []
    sub_keys = [sub.tobytes() for sub, _eligible in results]

    for v in range(len(matrices)):
        u = raw_of[v]

        if sub_keys[u] not in group_index:
            group_index[sub_keys[u]] = len(group_members)
            group_members.append([])
            group_eligible.append(results[u][1])

        group_of.append(group_index[sub_keys[u]])
        group_members[group_of[-1]].append(v)

    kept = [members[:len(group_eligible[g])] for g, members in enumerate(group_members)]
    vehicles = sorted(v for members in kept for v in members)

    # Only requests that some vehicle can serve need time / load variables
    eligible = {v: group_eligible[group_of[v]] for v in vehicles}
    active = sorted({r for v in vehicles for r in eligible[v]})

    return Scope(
        lo=lo,
        hi=hi,
        group_of=group_of,
        group_members=group_members,
        kept_members=kept,
        vehicles=vehicles,
        mask_of={v: results[raw_of[v]][0] for v in vehicles},
        eligible=eligible,
        active=active,
        active_nodes=[i for r in active for i in (r, r + n)],
    )
