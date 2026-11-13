from __future__ import annotations

from dataclasses import dataclass

from src.problem import Problem
from src.route import Route

@dataclass(frozen=True)
class CostScheme:
    ''' Cost parameters of the IDARP.
    
    The defaults are the Québec City values of the paper (Schenekemberg et al. 2025)
    '''
    
    private_per_km: float = 0.4 # Private fleet: cost per km traveled
    carrier_fixed: float = 3.0 # Common carrier: fixed cost f per request
    carrier_per_km: float = 2.0 # Common carrier: variable cost v per km
    bus_ticket: float = 3.0 # Bus ticket p_l per integrated request

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
        carrier_requests: set[int] = set(),
        integrated_requests: set[int] = set(),
        costs: CostScheme = CostScheme(),
    ):
        self.problem = problem
        self.routes = routes
        self.carrier_requests = carrier_requests
        self.integrated_requests = integrated_requests
        self.costs = costs

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
                
        assert len(set(route_nodes)) == len(route_nodes), 'Duplicate nodes found in routes'
        assert len(used_routes) <= self.problem.vehicles, 'Maximum number of vehicles exceeded'
        assert all(route.feasible for route in used_routes), 'One or more routes are not feasible'
    
        # Common carrier (direct trips)
        for k in self.carrier_requests:
            assert k in all_requests, f'Invalid carrier request: {k}'
                                
            pickup, delivery = self.problem.nodes[k], self.problem.nodes[k + self.problem.requests]
            direct = self.problem.times_cc[k, k + self.problem.requests]
    
            assert pickup.ready_time + pickup.service_time + direct <= delivery.due_time, \
                f'Request {k} cannot be served by the common carrier'

        # Bus integration (first mile by carrier, bus, last mile by carrier)
        for k in self.integrated_requests:
            assert k in self.problem.integrations, f'Request {k} has no bus integration option'
                    
            problem = self.problem
            n = problem.requests
            integration = problem.integrations[k]
            pickup, delivery = problem.nodes[k], problem.nodes[k + n]
    
            # (1) First mile
            first_mile = problem.times[k, integration.boarding_stop]
            latest_pickup = min(pickup.due_time, integration.boarding_time - pickup.service_time - first_mile)
            assert pickup.ready_time <= latest_pickup, \
                f'Request {k} cannot be served by the bus integration'
    
            # (2) Last mile
            last_mile = problem.times[integration.landing_stop, k + n]
            arrival = max(integration.landing_time + last_mile, delivery.ready_time)
            assert arrival <= delivery.due_time, \
                f'Request {k} cannot be served by the bus integration'
    
            # (3) Ride time
            assert arrival - latest_pickup - pickup.service_time <= problem.max_ride_time, \
                f'Request {k} cannot be served by the bus integration due to ride time'
            
        # Every request served exactly once
        requests = [*private_requests, *self.carrier_requests, *self.integrated_requests]

        assert len(requests) == len(set(requests)), 'Request served by more than one mode'
        assert all_requests == set(requests), 'Requests served do not match all requests'
    
    @property
    def cost(self) -> float:
        if self._cost is None:
            private_cost = self._private_cost()
            carrier_cost = sum(self._request_carrier_cost(k) for k in self.carrier_requests)
            integration_cost = sum(self._request_integration_cost(k) for k in self.integrated_requests)
            
            self._cost = private_cost + carrier_cost + integration_cost

        return self._cost

    def _private_cost(self) -> float:
        ''' Cost of serving all requests with the private fleet: v * total_distance '''
        total_distance = sum(r.distance for r in self.routes) / 10 ** self.problem.precision
        return self.costs.private_per_km * total_distance

    def _request_carrier_cost(self, k: int) -> float:
        ''' Cost of serving request `k` with the common carrier: f + v * d(k, n+k) '''
        distance = self.problem.distances_cc[k, k + self.problem.requests] / 10 ** self.problem.precision
        return self.costs.carrier_fixed + self.costs.carrier_per_km * distance

    def _request_integration_cost(self, k: int) -> float:
        ''' Cost of serving request `k` with its bus integration:
        - [f + v * d(k, board)] + p_l + [f + v * d(land, n+k)].
        '''
        
        integration = self.problem.integrations[k]
        
        first = self.problem.distances[k, integration.boarding_stop]
        last = self.problem.distances[integration.landing_stop, k + self.problem.requests]

        return (
            2 * self.costs.carrier_fixed + \
            self.costs.carrier_per_km * (first + last) / 10 ** self.problem.precision + \
            self.costs.bus_ticket
        )