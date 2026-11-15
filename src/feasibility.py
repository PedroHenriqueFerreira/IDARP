from __future__ import annotations

from src.problem import Problem


def is_feasible(problem: Problem, nodes: list[int]) -> bool:
    ''' Feasibility of a route (node ids between the depots): pairing and precedence, capacity and the
    temporal constraints. '''

    n = problem.requests

    if any(node < 1 or node > 2 * n for node in nodes):
        return False

    pos = {id: i for i, id in enumerate(nodes)}

    # Pickup before delivery, both in the same route
    if any(id + n not in pos or pos[id] > pos[id + n] for id in nodes if id <= n):
        return False

    if any(id - n not in pos for id in nodes if id > n):
        return False

    return _capacity_ok(problem, nodes) and _schedule_exists(problem, nodes, pos)


def _capacity_ok(problem: Problem, nodes: list[int]) -> bool:
    load = 0

    for node in nodes:
        load += problem.nodes[node].demand

        if load < 0 or load > problem.capacity:
            return False

    return True


def _schedule_exists(problem: Problem, nodes: list[int], pos: dict[int, int]) -> bool:
    ''' Time windows, ride time and route duration as a Simple Temporal Problem, by bound propagation.

    It finds a valid schedule whenever one exists (waiting is distributed to protect the ride times).
    '''

    n = problem.requests
    times = problem.times
    path = [problem.nodes[node] for node in (0, *nodes, 0)]
    last = len(path) - 1

    # (pickup position, delivery position) pairs for the ride time constraints
    pairs = [(i, pos[path[i].id + n] + 1) for i in range(1, last) if path[i].demand > 0]

    Ei = [node.ready_time for node in path]  # earliest start of service
    Li = [node.due_time for node in path]  # latest start of service
    Si = [node.service_time for node in path]

    # 2 * |V| iterations always suffice for convergence (Bellman-Ford)
    for _ in range(2 * len(path)):
        changed = False

        # Forward: travel times and time windows
        for i in range(1, len(path)):
            arrival = Ei[i - 1] + Si[i - 1] + times[path[i - 1].id, path[i].id]

            if arrival > Ei[i]:
                Ei[i] = arrival
                changed = True

            if Ei[i] > Li[i]:
                return False

        # Backward
        for i in range(len(path) - 2, -1, -1):
            departure = Li[i + 1] - times[path[i].id, path[i + 1].id] - Si[i]

            if departure < Li[i]:
                Li[i] = departure
                changed = True

            if Ei[i] > Li[i]:
                return False

        # Ride time: B_d - B_p - Si_p <= max_ride_time
        for p, d in pairs:
            earliest_p = Ei[d] - Si[p] - problem.max_ride_time

            if earliest_p > Ei[p]:
                Ei[p] = earliest_p
                changed = True

            if Ei[p] > Li[p]:
                return False

            latest_d = Li[p] + Si[p] + problem.max_ride_time

            if latest_d < Li[d]:
                Li[d] = latest_d
                changed = True

            if Ei[d] > Li[d]:
                return False

        # Route duration: B_end - B_start <= max_duration_route
        earliest_start = Ei[last] - problem.max_duration_route

        if earliest_start > Ei[0]:
            Ei[0] = earliest_start
            changed = True

        if Ei[0] > Li[0]:
            return False

        latest_end = Li[0] + problem.max_duration_route

        if latest_end < Li[last]:
            Li[last] = latest_end
            changed = True

        if Ei[last] > Li[last]:
            return False

        if not changed:
            return True

    return False  # did not converge: the constraints cannot hold together
