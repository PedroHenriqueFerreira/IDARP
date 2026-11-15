from __future__ import annotations

from src.problem import Problem
from src.route import Route


class Solution:
    ''' IDARP solution. Every request (pickup id 1..n) is served in exactly one way:
      - private fleet: pickup and delivery in one of the `routes` (at most `problem.vehicles`);
      - common carrier: in `carrier_requests` (direct trip);
      - bus integration: in `integrated_requests` (carrier -> bus -> carrier).
    '''

    def __init__(
        self,
        problem: Problem,
        routes: list[Route],
        carrier_requests: set[int],
        integrated_requests: set[int],
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
        ''' Raises AssertionError if the solution breaks any rule of the problem. '''

        problem = self.problem
        route_nodes = [id for route in self.routes for id in route.nodes]
        all_requests = set(range(1, problem.requests + 1))
        private_requests = {id for id in route_nodes if id <= problem.requests}
        used_routes = [route for route in self.routes if route.nodes]

        # Private fleet
        assert len(set(route_nodes)) == len(route_nodes)
        assert len(used_routes) <= problem.vehicles
        assert all(route.feasible for route in used_routes)

        # Common carrier
        for k in self.carrier_requests:
            assert k in all_requests
            assert problem.carrier_feasible(k)

        # Bus integration
        for k in self.integrated_requests:
            assert k in problem.integrations
            assert problem.integration_feasible(k)

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
            problem = self.problem
            self._cost = (
                sum(route.cost for route in self.routes)
                + sum(problem.carrier_cost(k) for k in self.carrier_requests)
                + sum(problem.integration_cost(k) for k in self.integrated_requests)
            )

        return self._cost
