from __future__ import annotations

from typing import Generator

from src.problem import Problem, Node

# Alternatives of a request. Tie-break order: private fleet first.
ROUTE, INTEGRATION, CARRIER = 0, 1, 2

class Route:
    ''' Class representing a private fleet vehicle route.

    The route is the ordered list of node ids visited between the depot (start) and the depot (end). 
    The pickup and the delivery of each request must be in the same route.
    '''

    def __init__(self, problem: Problem, nodes: list[int]):
        self.problem = problem
        self.nodes = nodes

        self._cost: float | None = None
        self._feasible: bool | None = None

    def __repr__(self) -> str:
        return f'Route<{", ".join(str(n) for n in self.nodes)}>'

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Generator[Node, None, None]:
        return (self.problem.nodes[node_id] for node_id in self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.problem.nodes[self.nodes[index]]

    def set_feasible(self):
        self._feasible = True
        return self._feasible

    def set_unfeasible(self):
        self._feasible = False
        return self._feasible
    
    @property
    def cost(self) -> float:
        ''' Total cost of the route (km) '''
        if self._cost is None:
            node_ids = [0, *self.nodes, 0]
            self._cost = sum(self.problem.private_cost(a, b) for a, b in zip(node_ids, node_ids[1:]))

        return self._cost

    @property
    def feasible(self) -> bool:
        ''' Validates the route's feasibility using bound propagation for the Simple Temporal Problem (STP). 
        
        This approach mathematically guarantees to find a valid schedule if one exists, distributing waiting times correctly to protect the passenger's ride time. 
        '''

        if self._feasible is not None:
            return self._feasible
        
        n = self.problem.requests
        if any(node < 1 or node > 2 * n for node in self.nodes):
            return self.set_unfeasible()

        pos = { id: i for i, id in enumerate(self.nodes) }

        # Precedence check (pickup before delivery)

        if any(id + n not in self.nodes or pos[id] > pos[id + n] for id in self.nodes if id <= n):
            return self.set_unfeasible()

        if any(id - n not in self.nodes for id in self.nodes if id > n):
            return self.set_unfeasible()

        # Capacity check

        load = 0
        for node in self.nodes:
            load += self.problem.nodes[node].demand
            if load < 0 or load > self.problem.capacity:
                return self.set_unfeasible()

        # Time windows, ride time, and route duration check using bound propagation

        node_ids = [0, *self.nodes, 0]
        nodes = [self.problem.nodes[node] for node in node_ids]
        last = len(nodes) - 1

        # (pickup position, delivery position) pairs for ride time constraints
        pairs = [(i, pos[nodes[i].id + n] + 1) for i in range(1, last) if nodes[i].demand > 0]

        Ei = [n.ready_time for n in nodes]  # earliest start of service
        Li = [n.due_time for n in nodes]  # latest start of service
        Si = [n.service_time for n in nodes]  # service times

        # Safe maximum iteration limit for Bellman-Ford convergence (2 * |V|)
        for _ in range(2 * len(nodes)):
            changed = False

            # Forward pass: travel times and time windows
            for i in range(1, len(nodes)):
                earliest_arrival = Ei[i - 1] + Si[i - 1] + self.problem.times[nodes[i - 1].id, nodes[i].id]

                if earliest_arrival > Ei[i]:
                    Ei[i] = earliest_arrival
                    changed = True

                if Ei[i] > Li[i]:
                    return self.set_unfeasible()

            # Backward pass: travel times and time windows
            for i in range(len(nodes) - 2, -1, -1):
                latest_departure = Li[i + 1] - self.problem.times[nodes[i].id, nodes[i + 1].id] - Si[i]

                if latest_departure < Li[i]:
                    Li[i] = latest_departure
                    changed = True

                if Ei[i] > Li[i]:
                    return self.set_unfeasible()

            # Ride time constraints (B_d - B_p - Si_p <= max_ride_time)
            for p, d in pairs:
                min_p = Ei[d] - Si[p] - self.problem.max_ride_time

                if min_p > Ei[p]:
                    Ei[p] = min_p
                    changed = True

                if Ei[p] > Li[p]:
                    return self.set_unfeasible()

                max_d = Li[p] + Si[p] + self.problem.max_ride_time

                if max_d < Li[d]:
                    Li[d] = max_d
                    changed = True

                if Ei[d] > Li[d]:
                    return self.set_unfeasible()

            # Maximum route duration constraint (B_end - B_start <= max_duration_route)
            # Forward
            min_0 = Ei[last] - self.problem.max_duration_route
            if min_0 > Ei[0]:
                Ei[0] = min_0
                changed = True

            if Ei[0] > Li[0]:
                return self.set_unfeasible()
            # Backward
            max_last = Li[0] + self.problem.max_duration_route
            if max_last < Li[last]:
                Li[last] = max_last
                changed = True

            if Ei[last] > Li[last]:
                return self.set_unfeasible()

            if not changed:
                break
        else:
            # The bounds did not converge within the limit: the constraints cannot be satisfied together
            return self.set_unfeasible()

        return self.set_feasible()

    def best_insertion(self, k: int) -> tuple[float, Route] | None:
        ''' Cheapest feasible insertion of request `k`. '''

        dist = self.problem.distances
        nodes = self.nodes
        m = len(nodes)
        p, d = k, k + self.problem.requests

        load = [0] * (m + 1)  # load before each position
        for t, node_id in enumerate(nodes):
            load[t + 1] = load[t] + self.problem.nodes[node_id].demand

        candidates = []

        for i in range(m + 1):
            if load[i] + 1 > self.problem.capacity:
                continue

            prev = nodes[i - 1] if i > 0 else 0

            for j in range(i, m + 1):
                if j > i and load[j] + 1 > self.problem.capacity:
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
            new_route = Route(self.problem, nodes[:i] + [p] + nodes[i:j] + [d] + nodes[j:])

            if new_route.feasible:
                return delta, new_route

        return None
    
def outside_cost(problem: Problem, k: int, carrier: set[int]) -> float:
    ''' Cost of request `k` served outside the private fleet: carrier if `k` is in `carrier`, else bus. '''

    return problem.carrier_cost(k) if k in carrier else problem.integration_cost(k)

def fixed_options(problem: Problem, k: int) -> list[tuple[float, int]]:
    ''' Alternatives whose cost does not depend on the routes: (cost, tag). '''

    options = []

    if problem.carrier_feasible(k):
        options.append((problem.carrier_cost(k), CARRIER))

    if problem.integration_feasible(k):
        options.append((problem.integration_cost(k), INTEGRATION))

    return options