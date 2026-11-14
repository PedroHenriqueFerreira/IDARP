from __future__ import annotations

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
    def cost(self) -> float:
        if self._cost is None:
            self._cost = 0.0
            
            self._cost += sum(route.cost() for route in self.routes)
            self._cost += sum(self.problem.carrier_cost(k) for k in self.carrier_requests)
            self._cost += sum(self.problem.integration_cost(k) for k in self.integrated_requests)

        return self._cost