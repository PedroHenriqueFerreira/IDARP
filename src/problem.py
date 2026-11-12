from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

@dataclass
class Node:
    ''' DARP node (depot, pickup or delivery). '''
    id: int
    code: str
    x: float  # Longitude
    y: float  # Latitude
    service_time: float  # Seconds
    demand: int  # Passengers (pickup +1, delivery -1)
    ready_time: float  # Seconds
    due_time: float  # Seconds
    allowed_to_integrate: bool

@dataclass
class BusStop:
    ''' Bus stop. `id` is the index in the full time/distance matrices. '''
    id: int
    code: str

@dataclass
class BusTrip:
    ''' One departure (bus) of a line: the ordered stops and the time the bus is at each one. '''
    line: int  # Id-Line
    bus: int  # Bus number, as in the file (starts at 1)
    stop_ids: np.ndarray  # Bus stop ids (indices in the full matrices), in visiting order
    stop_times: np.ndarray  # Time the bus is at each stop (scaled integers, same order)

@dataclass
class BusLine:
    id: int
    route: str  # Column "Line" of the file (e.g. 'L2')
    name: str  # Column "Name" of the file (e.g. 'Sainte-Foy_centre')
    trips: list[BusTrip] = field(default_factory=list)


@dataclass
class Integration:
    ''' Best bus integration pre-computed for a request (files IntegrationBusStation + Cost). '''
    request: int  # k: pickup node
    delivery: int  # n + k: delivery node
    line: int  # Id-Line
    bus: int  # Bus number inside the line (as in BusLines file)
    boarding_stop: int  # i: bus stop id where the user boards
    landing_stop: int  # j: bus stop id where the user gets off
    boarding_time: int  # Scaled integer
    landing_time: int  # Scaled integer
    cost: float | None = None  # From the Cost file
    mode: int | str | None = None  # From the Cost file (column "Mode", kept as it is)

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
        self.precision = precision  # Number of decimal places kept (values scaled by 10 ** precision)
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

        # Matrices files

        # Full matrices (nodes + bus stops), private fleet / bus network
        self.times: np.ndarray = None  # TimeMatrix (seconds)
        self.distances: np.ndarray = None  # DistanceMatrix (km)

        # Common carrier matrices (nodes + end depot 2n+1)
        self.times_cc: np.ndarray = None  # TimeBC
        self.distances_cc: np.ndarray = None  # DistBC

        # Bus lines file: public transportation network (lines, buses and stops)
        self.bus_lines: list[BusLine] = []

        # Integration and Cost files: best bus option for each integrable request
        self.integrations: dict[int, Integration] = {}

        self._load()

    # Helpers

    def _scale_value(self, value: str) -> int: # OK
        return int(round(float(value) * 10 ** self.precision))

    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray: # OK
        return np.rint(matrix * 10 ** self.precision).astype(np.int64)

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
        ''' Codes file: id -> code for DARP nodes and bus stops '''

        lines = self._read_lines('Codes')  # 'Elements:', count, 'ID Code', rows...
        count = int(lines[1])

        for i, line in enumerate(lines[3:]):
            id_, code = line.split()
            
            assert int(id_) == i, f'Codes file: expected id {i}, found {id_}'
            
            self.codes.append(code)

        if len(self.codes) != count:
            raise ValueError(f'Codes file: declared {count} elements, found {len(self.codes)}')

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
            
            assert int(row['Id']) == i, \
                f'DARP file: expected node id {i}, found {row["Id"]}'
            
            assert row['Code'] == self.codes[i], \
                f'DARP file: expected code {self.codes[i]}, found {row["Code"]}'

            # The headers are swapped: column 'Lat' holds longitude and column 'Lng' holds latitude.
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

        assert len(self.nodes) == 2 * self.requests + 1, \
            f'DARP file: expected {2 * self.requests + 1} nodes, found {len(self.nodes)}'
        
        # Bus stops are every element after the DARP nodes
        self.bus_stops = [BusStop(i, self.codes[i]) for i in range(len(self.nodes), len(self.codes))]

    def _load_matrices(self): # OK
        self.times = self._scale_matrix(self._read_labeled_matrix('TimeMatrix'))
        self.distances = self._scale_matrix(self._read_labeled_matrix('DistanceMatrix'))
        
        self.times_cc = self._scale_matrix(self._read_square_matrix('TimeBC'))
        self.distances_cc = self._scale_matrix(self._read_square_matrix('DistBC'))

        expected = 2 * self.requests + 2  # nodes + end depot
        
        assert self.times_cc.shape[0] == expected, \
            f'Common carrier matrices: expected size {expected}, found {self.times_cc.shape[0]}'

    def _load_bus_lines(self): # OK
        ''' BusLines file: one row per (line, bus, stop) '''

        # Number-Of-Lines / n, Number-Of-Bus-Per-Line / ..., Number-Of-Bus-Station-Per-Line / ..., header, rows
        
        lines = self._read_lines('BusLines')
        
        self.bus_lines = [None] * int(lines[1])
        
        buses_per_line = list(map(int, lines[3].split()))
        stops_per_line = list(map(int, lines[5].split()))
        columns = lines[6].split()

        trips: dict[tuple[int, int], tuple[list[int], list[int]]] = {}  # (line, bus) -> (stop ids, times)
        
        for line in lines[7:]:
            row = dict(zip(columns, line.split()))
            
            line_id, bus = int(row['Id-Line']), int(row['Bus'])
            stop_id, stop_time = int(row['Id']), self._scale_value(row['Time'])

            assert self.codes[stop_id] == row['Code'], \
                f'BusLines file: stop id {stop_id} does not match code {row["Code"]}'
        
            if self.bus_lines[line_id] is None:
                self.bus_lines[line_id] = BusLine(line_id, row['Line'], row['Name'])

            stop_ids, stop_times = trips.setdefault((line_id, bus), ([], []))
            
            stop_ids.append(stop_id)
            stop_times.append(stop_time)

        for (line_id, bus), (stop_ids, stop_times) in trips.items():
            assert len(stop_ids) == stops_per_line[line_id], \
                f'BusLines file: line {line_id} bus {bus} has {len(stop_ids)} stops, expected {stops_per_line[line_id]}'
                
            self.bus_lines[line_id].trips.append(BusTrip(
                line_id, bus, 
                np.array(stop_ids, dtype=np.int64), 
                np.array(stop_times, dtype=np.int64)
            ))

        for bus_line in self.bus_lines:
            assert len(bus_line.trips) == buses_per_line[bus_line.id], \
                f'BusLines file: line {bus_line.id} has {len(bus_line.trips)} buses, ' \
                f'expected {buses_per_line[bus_line.id]}'
            
    def _load_integrations(self): # OK
        ''' IntegrationBusStation file (+ Cost file): best bus option for each integrable request.

        Row format (header: k n+k Line Bus i j BoardingTime LandingTime : cod(k) cod(n+k) cod(i) cod(j)).
        Everything after the ':' is redundant (codes) and ignored.
        '''

        lines = self._read_lines('IntegrationBusStation')
        count = int(lines[1])

        for line in lines[3:]:
            tokens = line.split()
            request, delivery, bus_line, bus, board, land = map(int, tokens[:6])

            assert request not in self.integrations, \
                f'IntegrationBusStation file: request {request} appears more than once'

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

        assert len(self.integrations) == count, \
            f'IntegrationBusStation file: declared {count} integrations, found {len(self.integrations)}'

        # Cost file: Req, Cost, Mode
        lines = self._read_lines('Cost')
        count = int(lines[1])
        rows = lines[3:]

        assert len(rows) == count, \
            f'Cost file: declared {count} integrations, found {len(rows)}'

        for line in rows:
            req, cost, mode = line.split()
            
            req = int(req)
            cost = float(cost)
            
            assert req in self.integrations, \
                f'Cost file: request {req} has no entry in IntegrationBusStation file'
            
            self.integrations[req].cost = cost
            self.integrations[req].mode = mode

    def _read_labeled_matrix(self, suffix: str) -> np.ndarray: # OK
        ''' TimeMatrix / DistanceMatrix: header and first column hold the element codes '''

        lines = self._read_lines(suffix)  # 'Elements:', count, 'Code/Code <codes>', rows...
        count = int(lines[1])

        assert count == len(self.codes) and lines[2].split()[1:] == self.codes, \
            f'{suffix} file: header does not match the Codes file'
        
        matrix = np.zeros((count, count), dtype=float)

        for i, line in enumerate(lines[3:]):
            label, *values = line.split()
            assert label == self.codes[i] and len(values) == count, \
                f'{suffix} file: malformed row {i}'
            matrix[i] = values

        return matrix

    def _read_square_matrix(self, suffix: str) -> np.ndarray: # OK
        ''' TimeBC / DistBC: first line is the size, then size rows with no labels '''

        lines = self._read_lines(suffix)
        size = int(lines[0])
        matrix = np.array([line.split() for line in lines[1:]], dtype=float)

        assert matrix.shape == (size, size), \
            f'{suffix} file: expected {size}x{size}, found {matrix.shape}'

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
            pickup.due_time = min(
                pickup.due_time,
                delivery.due_time - pickup.service_time - direct
            )

            # Earliest delivery time considering the pickup's ready time and service time
            delivery.ready_time = max(
                delivery.ready_time,
                pickup.ready_time + pickup.service_time + direct
            )

            # Latest delivery time considering the maximum ride time constraint
            delivery.due_time = min(
                delivery.due_time,
                pickup.due_time + pickup.service_time + self.max_ride_time
            )

    def __repr__(self):
        return (
            f'Problem_{self.folder.name}'
            f'({self.requests} requests, '
            f'{self.vehicles} vehicles, '
            f'{len(self.bus_stops)} bus stops, '
            f'{len(self.bus_lines)} bus lines, '
            f'{len(self.integrations)} integrations)'
        )