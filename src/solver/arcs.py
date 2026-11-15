from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.problem import Problem

BIG = 1 << 60


def min_chain_times(times: np.ndarray, service: np.ndarray) -> np.ndarray:
    ''' chain[i, j]: lower bound of (service start at j) - (service start at i) when j is visited after i.

    Shortest path over weights service[i] + times[i, j]; valid without the triangle inequality.
    '''

    chain = service[:, None] + times
    np.fill_diagonal(chain, 0)

    for m in range(chain.shape[0]):
        np.minimum(chain, chain[:, m][:, None] + chain[m, :][None, :], out=chain)

    return chain


def static_arcs(
    n: int,
    service: np.ndarray,
    demand: np.ndarray,
    times: np.ndarray,
    capacity: int,
    max_ride: int,
    chain: np.ndarray,
    need_load: bool = True,
) -> np.ndarray:
    ''' Arcs that no feasible route can use for structural, load or ride-time reasons (no time windows). '''

    size = 2 * n + 1
    ok = np.ones((size, size), dtype=bool)
    ks = np.arange(1, n + 1)

    # Structure: no depot -> delivery, pickup -> depot, delivery of k -> pickup of k
    ok[0, n + 1:] = False
    ok[1:n + 1, 0] = False
    ok[ks + n, ks] = False

    # Load: load[j] = load[i] + demand[j] must fit the domain of load[j]
    if need_load:
        lo = np.maximum(0, demand)
        hi = np.minimum(capacity, capacity + demand)
        lo[0] = hi[0] = 0
        load_ok = (lo[:, None] + demand[None, :] <= hi[None, :]) & (hi[:, None] + demand[None, :] >= lo[None, :])
        load_ok[:, 0] = True  # no load constraint towards the depot
        ok &= load_ok

    # Ride time: a node right after pickup k (or right before delivery k + n) makes k ride at least this long
    after = times[ks, :] + chain[:, ks + n].T  # arc k -> j
    before = chain[ks, :] - service[ks][:, None] + service[None, :] + times[:, ks + n].T  # arc j -> k + n
    ok[ks, :] &= after <= max_ride
    ok[:, ks + n] &= (before <= max_ride).T

    np.fill_diagonal(ok, False)

    return ok


@lru_cache(maxsize=None)
def feasible_arcs(problem: Problem) -> np.ndarray:
    ''' Arcs that can belong to some feasible route of the instance (necessary conditions). '''

    size = 2 * problem.requests + 1
    times = problem.times[:size, :size]
    service = np.array([node.service_time for node in problem.nodes])
    ready = np.array([node.ready_time for node in problem.nodes])
    due = np.array([node.due_time for node in problem.nodes])
    demand = np.array([node.demand for node in problem.nodes])
    chain = min_chain_times(times, service)

    arcs = static_arcs(problem.requests, service, demand, times, problem.capacity, problem.max_ride_time, chain)
    arcs &= ready[:, None] + service[:, None] + times <= due[None, :]  # time windows

    # Route duration: a route using (i, j) lasts at least depot -> i -> j -> depot
    duration = chain[0, :, None] + (service[:, None] + times) + chain[:, 0][None, :]
    arcs &= duration <= problem.max_duration_route

    return arcs


def restrict_requests(mask: np.ndarray, n: int, request_ok: np.ndarray) -> tuple[np.ndarray, list[int]]:
    ''' Drops requests whose nodes lose all incoming or outgoing arcs, until a fixpoint.
    Returns the restricted arcs and the remaining (eligible) requests. '''

    alive = request_ok.copy()

    while True:
        keep = np.concatenate(([True], alive, alive))
        sub = mask & keep[:, None] & keep[None, :]
        reachable = sub.any(axis=0) & sub.any(axis=1)
        new_alive = alive & reachable[1:n + 1] & reachable[n + 1:]

        if np.array_equal(new_alive, alive):
            return sub, [int(r) + 1 for r in np.flatnonzero(alive)]

        alive = new_alive


def tighten_windows(
    ready: np.ndarray,
    due: np.ndarray,
    service: np.ndarray,
    times: np.ndarray,
    arcs: np.ndarray,
    n: int,
    max_ride: int,
    chain: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ''' Bound propagation over the allowed arcs: earliest / latest arrival at every node.

    A node served by the private fleet has a predecessor and a successor through allowed arcs, so its
    arrival is bounded by the best of them. Precedence and ride time between pickup and delivery tighten
    it further.
    '''

    ks = np.arange(1, n + 1)
    ds = ks + n
    lb, ub = ready.copy(), due.copy()

    for _ in range(100):
        previous_lb, previous_ub = lb.copy(), ub.copy()

        forward = np.where(arcs, lb[:, None] + service[:, None] + times, BIG)
        lb[1:] = np.maximum(lb[1:], forward[:, 1:].min(axis=0))

        backward = np.where(arcs, ub[None, :] - service[:, None] - times, -BIG)
        ub[1:] = np.minimum(ub[1:], backward[1:, :].max(axis=1))

        lb[ds] = np.maximum(lb[ds], lb[ks] + chain[ks, ds])
        ub[ks] = np.minimum(ub[ks], ub[ds] - chain[ks, ds])
        lb[ks] = np.maximum(lb[ks], lb[ds] - service[ks] - max_ride)
        ub[ds] = np.minimum(ub[ds], ub[ks] + service[ks] + max_ride)

        np.minimum(lb, BIG, out=lb)
        np.maximum(ub, -BIG, out=ub)

        if np.array_equal(lb, previous_lb) and np.array_equal(ub, previous_ub):
            break

    return lb, ub
