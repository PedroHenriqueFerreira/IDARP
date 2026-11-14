from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

@dataclass
class Node:
    ''' DARP node (depot, pickup or delivery). '''
    id: int
    code: str
    x: float # Longitude
    y: float # Latitude
    service_time: int
    demand: int # Passengers (pickup +1, delivery -1)
    ready_time: int 
    due_time: int
    allowed_to_integrate: bool

@dataclass
class BusStop:
    id: int
    code: str

@dataclass
class BusTrip:
    line: int
    bus: int  # Starts at 1
    stop_ids: np.ndarray
    stop_times: np.ndarray  

@dataclass
class BusLine:
    id: int
    route: str 
    name: str
    trips: list[BusTrip] = field(default_factory=list)

@dataclass
class Integration:
    ''' Best bus integration pre-computed for a request. '''
    request: int
    delivery: int
    line: int
    bus: int
    boarding_stop: int 
    landing_stop: int 
    boarding_time: int
    landing_time: int
    cost: float | None = None 
    mode: str | None = None

class Problem:
    ''' Class representing a DARP-PFCC instance integrated with public transportation (IDARP).

    An problem is a folder with 9 files sharing the prefix of the folder name:
      - `<folder>/<folder>_Codes.txt`,
      - `<folder>/<folder>_DARP.txt`,
      - `<folder>/<folder>_TimeMatrix.txt`,
      - `<folder>/<folder>_DistanceMatrix.txt`,
      - `<folder>/<folder>_TimeBC.txt`,
      - `<folder>/<folder>_DistBC.txt`,
      - `<folder>/<folder>_BusLines.txt`,
      - `<folder>/<folder>_IntegrationBusStation.txt`,
      - `<folder>/<folder>_Cost.txt`.

    Index conventions:
      - Ids 0..2n are the DARP nodes (0 = depot, 1..n pickups, n+1..2n deliveries).
      - Ids 2n+1.. are bus stops. These ids are the ones used in the full matrices and in the
        bus line / integration files.

    All time and distance values are stored as integers scaled by 10 ** precision.
    '''

    def __init__(self, folder: str, precision: int = 2, tighten_time_windows: bool = True):
        self.folder = Path(folder).resolve()  # Instance folder
        self.scale: int = 10 ** precision   # Scale factor for time and distance values
        self.tighten_time_windows = tighten_time_windows # Whether to tighten DARP time windows

        # Codes file
        self.codes: list[str] = []  # codes[id] for every element
        
        # DARP file
        self.vehicles = 0  # Number of vehicles
        self.requests = 0  # Number of requests
        self.max_duration_route = 0  # Maximum route duration
        self.capacity = 0  # Each vehicle capacity
        self.max_ride_time = 0  # Maximum ride time
        self.integration_percentage = 0.0  # Percentage of requests allowed to integrate
        self.nodes: list[Node] = []  # Depot, pickups and deliveries
        self.bus_stops: list[BusStop] = [] # Bus stops

        # Full matrices (nodes + bus stops), private fleet / bus network
        self.times: np.ndarray = None  # TimeMatrix file (seconds)
        self.distances: np.ndarray = None  # DistanceMatrix file (km)

        # Common carrier matrices (nodes + end depot 2n+1)
        self.times_cc: np.ndarray = None  # TimeBC file (seconds)
        self.distances_cc: np.ndarray = None  # DistBC file (km)

        # BusLines file: public transportation network (lines, buses and stops)
        self.bus_lines: list[BusLine] = []

        # Integration and Cost files: best bus option for each integrable request
        self.integrations: dict[int, Integration] = {}

        self._load()

    def __repr__(self):
        return (
            f'Problem::{self.folder.name}'
            f'({self.requests} requests, '
            f'{self.vehicles} vehicles, '
            f'{len(self.bus_stops)} bus_stops, '
            f'{len(self.bus_lines)} bus_lines, '
            f'{len(self.integrations)} integrations)'
        )

    # Helpers

    def _scale_value(self, value: str) -> int:
        return int(round(float(value) * self.scale))

    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.rint(matrix * self.scale).astype(np.int64)

    def _read_lines(self, suffix: str) -> list[str]: # OK
        ''' Non-empty, stripped lines of <folder>/<name>_<suffix>.txt '''
        
        path = self.folder / f'{self.folder.name}_{suffix}.txt'
        
        if not path.is_file():
            raise FileNotFoundError(f'IDARP instance file not found: {path}')
        
        with open(path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    # Loading

    def _load(self):
        self._load_codes()
        self._load_darp()
        self._load_matrices()
        self._load_bus_lines()
        self._load_integrations()

        if self.tighten_time_windows:
            self._tighten_time_windows()

    def _load_codes(self): # OK
        ''' Loads the Codes file. '''

        lines = self._read_lines('Codes') 
        count = int(lines[1])

        for i, line in enumerate(lines[3:]):
            id, code = line.split()
            
            assert int(id) == i
            
            self.codes.append(code)

        assert len(self.codes) == count

    def _load_darp(self): # OK
        ''' DARP file: configuration + nodes '''

        lines = self._read_lines('DARP')
        
        cfg = dict(zip(lines[0].split(), lines[1].split()))

        self.vehicles = int(cfg['Vehicles'])
        self.requests = int(cfg['Requests'])
        self.max_duration_route = self._scale_value(cfg['Max-Duration-Route'])
        self.capacity = int(cfg['Capacity'])
        self.max_ride_time = self._scale_value(cfg['Max-Ride-Time'])
        self.integration_percentage = float(cfg['Integration-Percentage'])

        columns = lines[2].split()

        for i, line in enumerate(lines[3:]):
            row = dict(zip(columns, line.split()))
            
            assert int(row['Id']) == i
            assert row['Code'] == self.codes[i]

            # Column 'Lat' holds longitude and column 'Lng' holds latitude.
            self.nodes.append(Node(
                id=int(row['Id']),
                code=row['Code'],
                x=float(row['Lat']),
                y=float(row['Lng']),
                service_time=self._scale_value(row['Service-Time']),
                demand=int(row['Demand']),
                ready_time=self._scale_value(row['Ei']),
                due_time=self._scale_value(row['Li']),
                allowed_to_integrate=bool(int(row['AllowedToIntegrate']))
            ))

        assert len(self.nodes) == 2 * self.requests + 1
        
        # Bus stops are every element after the DARP nodes
        n_nodes, n_codes = len(self.nodes), len(self.codes)
        self.bus_stops = [BusStop(i, self.codes[i]) for i in range(n_nodes, n_codes)]

    def _load_matrices(self):
        self.times = self._scale_matrix(self._read_labeled_matrix('TimeMatrix'))
        self.distances = self._scale_matrix(self._read_labeled_matrix('DistanceMatrix'))
        
        self.times_cc = self._scale_matrix(self._read_square_matrix('TimeBC'))
        self.distances_cc = self._scale_matrix(self._read_square_matrix('DistBC'))

        expected = 2 * self.requests + 2  # nodes + end depot
        
        assert self.times_cc.shape[0] == expected

    def _load_bus_lines(self):
        ''' BusLines file: one row per (line, bus, stop) '''

        lines = self._read_lines('BusLines')
        
        self.bus_lines = [None] * int(lines[1])
        
        buses_per_line = list(map(int, lines[3].split()))
        stops_per_line = list(map(int, lines[5].split()))
        columns = lines[6].split()

        # (line, bus) -> (stop ids, times)
        trips: dict[tuple[int, int], tuple[list[int], list[int]]] = {}  
        
        for line in lines[7:]:
            row = dict(zip(columns, line.split()))
            
            line_id, bus = int(row['Id-Line']), int(row['Bus'])
            stop_id, stop_time = int(row['Id']), self._scale_value(row['Time'])

            assert self.codes[stop_id] == row['Code']
        
            if self.bus_lines[line_id] is None:
                self.bus_lines[line_id] = BusLine(line_id, row['Line'], row['Name'])

            stop_ids, stop_times = trips.setdefault((line_id, bus), ([], []))
            
            stop_ids.append(stop_id)
            stop_times.append(stop_time)

        for (line_id, bus), (stop_ids, stop_times) in trips.items():
            assert len(stop_ids) == stops_per_line[line_id]
                
            stop_ids_np = np.array(stop_ids, dtype=np.int64)
            stop_times_np = np.array(stop_times, dtype=np.int64)
            self.bus_lines[line_id].trips.append(BusTrip(line_id, bus, stop_ids_np, stop_times_np))

        assert all(len(bus_line.trips) == buses_per_line[bus_line.id] for bus_line in self.bus_lines)
            
    def _load_integrations(self):
        ''' IntegrationBusStation file (+ Cost file): best bus option for each integrable request. '''

        lines = self._read_lines('IntegrationBusStation')

        for line in lines[3:]:
            tokens = line.split()
            request, delivery, bus_line, bus, board, land = map(int, tokens[:6])

            assert request not in self.integrations
            
            self.integrations[request] = Integration(
                request=request,
                delivery=delivery,
                line=bus_line,
                bus=bus,
                boarding_stop=board,
                landing_stop=land,
                boarding_time=self._scale_value(tokens[6]),
                landing_time=self._scale_value(tokens[7]),
            )

        assert len(self.integrations) == int(lines[1])

        # Cost file: Req, Cost, Mode
        lines = self._read_lines('Cost')
        rows = lines[3:]

        assert len(rows) == int(lines[1])

        for line in rows:
            _req, _cost, mode = line.split()
            req, cost = int(_req), float(_cost)
            
            assert req in self.integrations
            
            self.integrations[req].cost = cost
            self.integrations[req].mode = mode

    def _read_labeled_matrix(self, suffix: str) -> np.ndarray:
        ''' TimeMatrix / DistanceMatrix: header and first column hold the element codes '''

        lines = self._read_lines(suffix)
        count = int(lines[1])

        assert count == len(self.codes) and lines[2].split()[1:] == self.codes
        
        matrix = np.zeros((count, count), dtype=float)

        for i, line in enumerate(lines[3:]):
            label, *values = line.split()
            
            assert label == self.codes[i] and len(values) == count
            matrix[i] = values

        return matrix

    def _read_square_matrix(self, suffix: str) -> np.ndarray: # OK
        ''' TimeBC / DistBC: first line is the size, then size rows with no labels '''

        lines = self._read_lines(suffix)
        size = int(lines[0])
        
        matrix = np.array([line.split() for line in lines[1:]], dtype=float)
        assert matrix.shape == (size, size)

        return matrix

    # Preprocessing

    def _tighten_time_windows(self): # OK
        ''' Time window tightening, using the time matrix as travel time. '''

        for i in range(1, self.requests + 1):
            pickup, delivery = self.nodes[i], self.nodes[i + self.requests]
            direct: int = self.times[pickup.id, delivery.id]

            # Earliest pickup time considering the maximum ride time constraint
            pickup.ready_time = max(
                pickup.ready_time,
                delivery.ready_time - pickup.service_time - self.max_ride_time
            )

            # Latest pickup time considering the delivery's due time
            pickup.due_time = min(pickup.due_time, delivery.due_time - pickup.service_time - direct)

            # Earliest delivery time considering the pickup's ready time and service time
            delivery.ready_time = max(delivery.ready_time, pickup.ready_time + pickup.service_time + direct)

            # Latest delivery time considering the maximum ride time constraint
            delivery.due_time = min(
                delivery.due_time,
                pickup.due_time + pickup.service_time + self.max_ride_time
            )
