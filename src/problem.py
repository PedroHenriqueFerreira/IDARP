from __future__ import annotations

from pathlib import Path

from src import loader
from src.models import (  # noqa: F401 (re-exported)
    CARRIER, INTEGRATION, ROUTE, BusLine, BusStop, CostScheme, Integration, Node,
)


class Problem:
    ''' DARP-PFCC instance integrated with public transportation (IDARP).

    An instance is a folder with 9 files `<folder>/<folder>_<Suffix>.txt`:
    Codes, DARP, TimeMatrix, DistanceMatrix, TimeBC, DistBC, BusLines, IntegrationBusStation, Cost.

    Ids 0..2n are the DARP nodes (0 = depot, 1..n pickups, n+1..2n deliveries); ids 2n+1.. are bus
    stops (the ones used in the full matrices and in the bus line / integration files).
    '''

    def __init__(self, folder: str, costs: CostScheme = CostScheme()):
        self.folder = Path(folder).resolve()
        self.costs = costs

        self.codes = loader.load_codes(self.folder)

        config, self.nodes = loader.load_darp(self.folder, self.codes)
        self.vehicles = int(config['Vehicles'])
        self.requests = int(config['Requests'])
        self.max_duration_route = float(config['Max-Duration-Route'])
        self.capacity = int(config['Capacity'])
        self.max_ride_time = float(config['Max-Ride-Time'])
        self.integration_percentage = float(config['Integration-Percentage'])

        # Bus stops are every element after the DARP nodes
        self.bus_stops = [BusStop(i, self.codes[i]) for i in range(len(self.nodes), len(self.codes))]

        # Full matrices (nodes + bus stops): times in seconds, distances in km
        # Common carrier matrices (nodes + end depot 2n+1)
        self.times, self.distances, self.times_cc, self.distances_cc = loader.load_matrices(
            self.folder, self.codes, self.requests
        )

        self.bus_lines: list[BusLine] = loader.load_bus_lines(self.folder, self.codes)
        self.integrations: dict[int, Integration] = loader.load_integrations(self.folder)

        self._tighten_time_windows()

    def __repr__(self):
        return (
            f'Problem::{self.folder.name}'
            f'({self.requests} requests, {self.vehicles} vehicles, {len(self.bus_stops)} bus_stops, '
            f'{len(self.bus_lines)} bus_lines, {len(self.integrations)} integrations)'
        )

    def _tighten_time_windows(self):
        ''' Tightens pickup / delivery windows with the direct travel time and the maximum ride time. '''

        for i in range(1, self.requests + 1):
            pickup, delivery = self.nodes[i], self.nodes[i + self.requests]
            direct = self.times[pickup.id, delivery.id]

            pickup.ready_time = max(
                pickup.ready_time, delivery.ready_time - pickup.service_time - self.max_ride_time
            )
            pickup.due_time = min(pickup.due_time, delivery.due_time - pickup.service_time - direct)
            delivery.ready_time = max(delivery.ready_time, pickup.ready_time + pickup.service_time + direct)
            delivery.due_time = min(
                delivery.due_time, pickup.due_time + pickup.service_time + self.max_ride_time
            )

    # Costs and alternatives

    def private_cost(self, i: int, j: int) -> float:
        ''' Cost of the private fleet going from node i to node j. '''

        return self.costs.private_per_km * self.distances[i, j]

    def carrier_feasible(self, k: int) -> bool:
        ''' Whether request `k` can be served by a direct common carrier trip. '''

        pickup, delivery = self.nodes[k], self.nodes[k + self.requests]
        direct = self.times_cc[k, k + self.requests]

        return pickup.ready_time + pickup.service_time + direct <= delivery.due_time

    def carrier_cost(self, k: int) -> float:
        ''' f + v * d(k, n+k) '''

        distance = self.distances_cc[k, k + self.requests]

        return self.costs.carrier_fixed + self.costs.carrier_per_km * distance

    def integration_feasible(self, k: int) -> bool:
        ''' Whether request `k` can be served by its bus integration. '''

        integration = self.integrations.get(k)

        if integration is None or not self.nodes[k].allowed_to_integrate:
            return False

        pickup, delivery = self.nodes[k], self.nodes[k + self.requests]

        # First mile
        first_mile = self.times[k, integration.boarding_stop]
        latest_pickup = min(pickup.due_time, integration.boarding_time - pickup.service_time - first_mile)

        if pickup.ready_time > latest_pickup:
            return False

        # Last mile
        last_mile = self.times[integration.landing_stop, k + self.requests]
        arrival = max(integration.landing_time + last_mile, delivery.ready_time)

        if arrival > delivery.due_time:
            return False

        # Ride time
        return arrival - latest_pickup - pickup.service_time <= self.max_ride_time

    def integration_cost(self, k: int) -> float:
        ''' [f + v * d(k, board)] + p_l + [f + v * d(land, n+k)] '''

        integration = self.integrations[k]
        first = self.distances[k, integration.boarding_stop]
        last = self.distances[integration.landing_stop, k + self.requests]

        return 2 * self.costs.carrier_fixed + self.costs.carrier_per_km * (first + last) + self.costs.bus_ticket

    def fixed_options(self, k: int) -> list[tuple[float, int]]:
        ''' Alternatives whose cost does not depend on the routes: (cost, tag). '''

        options = []

        if self.carrier_feasible(k):
            options.append((self.carrier_cost(k), CARRIER))

        if self.integration_feasible(k):
            options.append((self.integration_cost(k), INTEGRATION))

        return options

    def outside_cost(self, k: int, carrier: set[int]) -> float:
        ''' Cost of `k` outside the private fleet: carrier if `k` is in `carrier`, else bus. '''

        return self.carrier_cost(k) if k in carrier else self.integration_cost(k)
