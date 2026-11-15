from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from src.problem import Problem
from src.solver.arcs import min_chain_times, static_arcs


def prepare(problem: Problem) -> SimpleNamespace:
    ''' Numpy view of the instance restricted to the depot, pickups and deliveries.

    All time data is divided by the greatest common divisor of every time value: the schedule can be
    expressed in multiples of it, and smaller domains make the search faster.
    '''

    n = problem.requests
    size = 2 * n + 1
    nodes = problem.nodes[:size]

    # The int64 conversion below truncates: the data must already be integral (see `loader`)
    assert np.array_equal(problem.times, np.floor(problem.times)), 'times must be integral'
    assert all(
        float(node.ready_time).is_integer() and float(node.due_time).is_integer()
        and float(node.service_time).is_integer()
        for node in nodes
    ), 'time windows and service times must be integral'

    ready = np.array([node.ready_time for node in nodes], dtype=np.int64)
    due_raw = np.array([node.due_time for node in nodes], dtype=np.int64)
    due = np.maximum(ready, due_raw)
    window_ok = ready <= due_raw
    service = np.array([node.service_time for node in nodes], dtype=np.int64)
    demand = np.array([node.demand for node in nodes], dtype=np.int64)
    demand[0] = 0

    times = np.asarray(problem.times, dtype=np.int64)[:size, :size].copy()
    max_ride = int(problem.max_ride_time)
    max_duration = int(problem.max_duration_route)

    g = int(np.gcd.reduce(np.concatenate([ready, due, service, times.ravel()])))

    if g > 1:
        ready //= g
        due //= g
        service //= g
        times //= g
        max_ride //= g
        max_duration //= g

    return SimpleNamespace(
        ready=ready,
        due=due,
        service=service,
        demand=demand,
        times=times,
        max_ride=max_ride,
        max_duration=max_duration,
        request_ok=window_ok[1:n + 1] & window_ok[n + 1:],
    )


@lru_cache(maxsize=4)
def context(problem: Problem) -> SimpleNamespace:
    ''' Everything `solve` derives from the instance alone (constant across LNS calls).

    The arrays are shared between calls: `solve` must not modify them in place.
    '''

    n = problem.requests
    data = prepare(problem)
    need_load = int(np.maximum(0, data.demand[1:n + 1]).sum()) > problem.capacity
    duration_binding = int(data.due[0] - data.ready[0]) > data.max_duration
    chain = min_chain_times(data.times, data.service)
    static = static_arcs(
        n, data.service, data.demand, data.times, problem.capacity, data.max_ride, chain, need_load
    )

    return SimpleNamespace(
        data=data,
        need_load=need_load,
        duration_binding=duration_binding,
        chain=chain,
        static=static,
        Tl=data.times.tolist(),
        Cl=chain.tolist(),
        ready_l=data.ready.tolist(),
        due_l=data.due.tolist(),
        service_l=data.service.tolist(),
        demand_l=data.demand.tolist(),
    )
