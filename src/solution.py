from __future__ import annotations

from typing import Generator

from src.problem import Problem
from src.route import Route

class Solution:
    ''' Class representing an IDARP solution.

    Every request (identified by its pickup id, 1..n) must be served in exactly one way:
      - private fleet: pickup and delivery are in one of the `routes` (at most `problem.vehicles`);
      - common carrier: the request is in `carrier_requests` (direct trip);
      - bus integration: the request is in `integrated_requests` (carrier -> bus -> carrier).
    '''

    def __init__(
        self,
        problem: Problem,
        routes: list[Route],
        carrier_requests: set[int],
        integrated_requests: set[int]
    ):
        self.problem = problem
        self.routes = routes
        self.carrier_requests = carrier_requests
        self.integrated_requests = integrated_requests

        self._cost: float | None = None

    def __repr__(self):
        return (
            f'Solution(routes={self.routes}, '
            f'carrier={self.carrier_requests}, '
            f'integrated={self.integrated_requests})'
        )

    def validate(self):
        ''' Raises an AssertionError if the solution breaks any rule of the problem. '''

        route_nodes = [id for route in self.routes for id in route.nodes]
        all_requests = set(range(1, self.problem.requests + 1))
        private_requests = { id for id in route_nodes if id <= self.problem.requests }
        
        # Private fleet (shared vehicles)
        used_routes = [route for route in self.routes if route.nodes]
                
        assert len(set(route_nodes)) == len(route_nodes)
        assert len(used_routes) <= self.problem.vehicles
        assert all(route.feasible for route in used_routes)
    
        # Common carrier (direct trips)
        for k in self.carrier_requests:
            assert k in all_requests                                
            assert self.problem.carrier_feasible(k)

        # Bus integration (first mile by carrier, bus, last mile by carrier)
        for k in self.integrated_requests:
            assert k in self.problem.integrations
            assert self.problem.integration_feasible(k)

        # Every request served exactly once
        requests = [*private_requests, *self.carrier_requests, *self.integrated_requests]

        assert len(requests) == len(set(requests))
        assert all_requests == set(requests)
    
    @property
    def outside_requests(self) -> set[int]:
        ''' Requests served by the carrier or the bus. '''

        return self.carrier_requests | self.integrated_requests

    @property
    def cost(self) -> float:
        if self._cost is None:
            self._cost = 0.0
            
            self._cost += sum(route.cost for route in self.routes)
            self._cost += sum(self.problem.carrier_cost(k) for k in self.carrier_requests)
            self._cost += sum(self.problem.integration_cost(k) for k in self.integrated_requests)

        return self._cost

def best_insertion(problem: Problem, route: Route, k: int) -> tuple[float, Route] | None:
    ''' Cheapest feasible insertion of request `k` in `route`: (distance increase, new route) or None. '''

    dist = problem.distances
    nodes = route.nodes
    m = len(nodes)
    p, d = k, k + problem.requests

    load = [0] * (m + 1)  # load before each position
    for t, node_id in enumerate(nodes):
        load[t + 1] = load[t] + problem.nodes[node_id].demand

    candidates = []

    for i in range(m + 1):
        if load[i] + 1 > problem.capacity:
            continue

        prev = nodes[i - 1] if i > 0 else 0

        for j in range(i, m + 1):
            if j > i and load[j] + 1 > problem.capacity:
                break

            if j == i:  # pickup and delivery adjacent
                nxt = nodes[i] if i < m else 0
                delta = dist[prev, p] + dist[p, d] + dist[d, nxt] - dist[prev, nxt]
            else:
                after_p, before_d = nodes[i], nodes[j - 1]
                after_d = nodes[j] if j < m else 0
                delta = (
                    dist[prev, p] + dist[p, after_p] - dist[prev, after_p]
                    + dist[before_d, d] + dist[d, after_d] - dist[before_d, after_d]
                )

            candidates.append((delta, i, j))

    for delta, i, j in sorted(candidates):
        new_route = Route(problem, nodes[:i] + [p] + nodes[i:j] + [d] + nodes[j:])

        if new_route.feasible:
            return delta, new_route

    return None


