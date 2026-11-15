from __future__ import annotations

from typing import Generator

from src.feasibility import is_feasible
from src.problem import Node, Problem


class Route:
    ''' Private fleet route: node ids visited between the depot (start) and the depot (end).
    The pickup and the delivery of each request must be in the same route. '''

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

    def set_feasible(self) -> bool:
        self._feasible = True
        return True

    def set_unfeasible(self) -> bool:
        self._feasible = False
        return False

    @property
    def cost(self) -> float:
        ''' Total cost of the route. '''

        if self._cost is None:
            ids = [0, *self.nodes, 0]
            self._cost = sum(self.problem.private_cost(a, b) for a, b in zip(ids, ids[1:]))

        return self._cost

    @property
    def feasible(self) -> bool:
        if self._feasible is None:
            self._feasible = is_feasible(self.problem, self.nodes)

        return self._feasible

    def best_insertion(self, k: int) -> tuple[float, Route] | None:
        ''' Cheapest feasible insertion of request `k`: (distance increase, new route) or None. '''

        p, d = k, k + self.problem.requests

        for delta, i, j in _insertion_candidates(self.problem, self.nodes, k):
            route = Route(self.problem, self.nodes[:i] + [p] + self.nodes[i:j] + [d] + self.nodes[j:])

            if route.feasible:
                return delta, route

        return None


def _insertion_candidates(problem: Problem, nodes: list[int], k: int) -> list[tuple[float, int, int]]:
    ''' (distance increase, pickup position, delivery position) of every capacity-feasible insertion, sorted. '''

    dist = problem.distances
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

    return sorted(candidates)
