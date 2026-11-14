from __future__ import annotations

from typing import Generator

from src.problem import Problem, Node

class Route:
    ''' Class representing a private fleet vehicle route.

    The route is the ordered list of node ids visited between the depot (start) and the depot (end). 
    The pickup and the delivery of each request must be in the same route.
    '''

    def __init__(self, problem: Problem, nodes: list[int]):
        self.problem = problem
        self.nodes = nodes

        self._distance: int | None = None
        self._feasible: bool | None = None

    def __repr__(self) -> str:
        return f'Route<{", ".join(str(n) for n in self.nodes)}>'

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Generator[Node, None, None]:
        return (self.problem.nodes[node_id] for node_id in self.nodes)

    def __getitem__(self, index: int) -> Node:
        return self.problem.nodes[self.nodes[index]]

    def cost(self):
        return self.problem.costs.private_per_km * self.distance

    @property
    def distance(self) -> int:
        ''' Total distance of the route (km, scaled by 10 ** precision) '''
        if self._distance is None:
            self._distance = 0
            
            if self.nodes:
                self._distance += self.problem.distances[0, self.nodes[0]]  # depot to first node
                self._distance += sum(self.problem.distances[a, b] for a, b in zip(self.nodes, self.nodes[1:]))
                self._distance += self.problem.distances[self.nodes[-1], 0]  # last node to depot

        return self._distance / self.problem.scale

    def set_feasible(self):
        self._feasible = True
        return self._feasible

    def set_unfeasible(self):
        self._feasible = False
        return self._feasible

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
        for node in self:
            load += node.demand
            if load < 0 or load > self.problem.capacity:
                return self.set_unfeasible()

        # Time windows, ride time, and route duration check using bound propagation

        nodes = [self.problem.nodes[0], *list(self), self.problem.nodes[0]]
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
