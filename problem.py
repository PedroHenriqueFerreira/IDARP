import os
from dataclasses import dataclass, field

import numpy as np

class IDARPNode:
    ''' DARP node (depot, pickup or delivery) with the IDARP attributes. '''

    def __init__(
        self,
        id: int,
        code: str,
        x: float,
        y: float,
        service_time: float,
        demand: int,
        ready_time: float,
        due_time: float,
        allowed_to_integrate: int,
        precision: int,
    ):
        self.id = int(id)
        self.code = code  # Postal-code-like identifier of the location
        self.pos = np.array([float(x), float(y)])  # (longitude, latitude)
        self.service_time = int(round(service_time * 10 ** precision))
        self.demand = int(demand)
        self.ready_time = int(round(ready_time * 10 ** precision))
        self.due_time = int(round(due_time * 10 ** precision))
        self.allowed_to_integrate = bool(int(allowed_to_integrate))  # Request may use the bus

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def __repr__(self):
        return f'Node({self.id}, {self.code})'


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
    times: np.ndarray  # Time the bus is at each stop (scaled integers, same order)


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

    An instance is a folder with 9 files sharing the prefix of the folder name, e.g.
        '<FOLDER>/<FOLDER>_Codes.txt',
        `<FOLDER>/<FOLDER>_DARP.txt`,
        '<FOLDER>/<FOLDER>_TimeMatrix.txt',
        '<FOLDER>/<FOLDER>_DistanceMatrix.txt',
        '<FOLDER>/<FOLDER>_TimeBC.txt',
        '<FOLDER>/<FOLDER>_DistBC.txt',
        '<FOLDER>/<FOLDER>_BusLines.txt',
        '<FOLDER>/<FOLDER>_IntegrationBusStation.txt',
        '<FOLDER>/<FOLDER>_Cost.txt'.

    Index conventions:
      - Ids 0..2n are the DARP nodes (0 = depot, 1..n pickups, n+1..2n deliveries).
      - Ids 2n+1.. are bus stops. These ids are the ones used in the full matrices and in the
        bus line / integration files.

    All time and distance values are stored as integers scaled by 10 ** precision.
    '''

    def __init__(self, folder: str, precision: int = 3, tighten_time_windows: bool = True):
        self.folder = os.path.normpath(folder)  # Instance folder
        self.name = os.path.basename(self.folder)  # Also the prefix of every file
        self.precision = precision  # Number of decimal places kept (values scaled by 10 ** precision)
        self.tighten_time_windows = tighten_time_windows

        # DARP file
        self.vehicle_number = 0  # Number of vehicles
        self.request_number = 0  # Number of requests
        self.max_vehicle_time = 0  # Maximum route duration
        self.vehicle_capacity = 0  # Each vehicle capacity
        self.max_request_time = 0  # Maximum ride time
        self.integration_percentage = 0.0  # Percentage of requests allowed to integrate
        self.nodes: list[IDARPNode] = []  # Depot, pickups and deliveries

        # Codes file
        self.codes: list[str] = []  # codes[id] for every element (nodes + bus stops)
        self.code_to_id: dict[str, int] = {}
        self.bus_stops: list[BusStop] = []

        # Full matrices (nodes + bus stops), private fleet / bus network
        self.times: np.ndarray = None  # TimeMatrix (seconds)
        self.distances: np.ndarray = None  # DistanceMatrix (km)

        # Common carrier matrices (nodes + end depot 2n+1)
        self.times_cc: np.ndarray = None  # TimeBC
        self.distances_cc: np.ndarray = None  # DistBC

        # Public transportation
        self.bus_lines: list[BusLine] = []

        # Pre-processed integrations, indexed by request (pickup id)
        self.integrations: dict[int, Integration] = {}

        self._load()

    # ------------------------------------------------------------------ helpers

    @property
    def end_depot(self) -> int:
        ''' Index of the end depot (2n+1) in the common carrier matrices. '''
        return 2 * self.request_number + 1

    def _scale(self, value: float) -> int:
        return int(round(float(value) * 10 ** self.precision))

    def _scale_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.rint(matrix * 10 ** self.precision).astype(np.int64)

    def _read_lines(self, suffix: str) -> list[str]:
        ''' Non-empty, stripped lines of <folder>/<name>_<suffix>.txt '''
        path = os.path.join(self.folder, f'{self.name}_{suffix}.txt')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'IDARP instance file not found: {path}')
        with open(path, 'r') as file:
            return [line.strip() for line in file if line.strip()]

    # ------------------------------------------------------------------ loading

    def _load(self):
        self._load_codes()
        self._load_darp()
        self._load_full_matrices()
        self._load_common_carrier_matrices()
        self._load_bus_lines()
        self._load_integrations()

        if self.tighten_time_windows:
            self._tighten_time_windows()

    def _load_codes(self):
        ''' Codes file: id -> code for DARP nodes and bus stops '''

        lines = self._read_lines('Codes')  # 'Elements:', count, 'ID Code', rows...
        count = int(lines[1])

        for i, line in enumerate(lines[3:]):
            id_, code = line.split()
            if int(id_) != i:
                raise ValueError(f'Codes file: expected id {i}, found {id_}')
            self.codes.append(code)

        if len(self.codes) != count:
            raise ValueError(f'Codes file: declared {count} elements, found {len(self.codes)}')

        self.code_to_id = {code: i for i, code in enumerate(self.codes)}

    def _load_darp(self):
        ''' DARP file: configuration + nodes '''

        lines = self._read_lines('DARP')
        cfg = dict(zip(lines[0].split(), lines[1].split()))

        self.vehicle_number = int(cfg['Vehicles'])
        self.request_number = int(cfg['Requests'])
        self.max_vehicle_time = self._scale(cfg['Max-Duration-Route'])
        self.vehicle_capacity = int(cfg['Capacity'])
        self.max_request_time = self._scale(cfg['Max-Ride-Time'])
        self.integration_percentage = float(cfg['Integration-Percentage'])

        columns = lines[2].split()

        for i, line in enumerate(lines[3:]):
            row = dict(zip(columns, line.split()))

            if int(row['Id']) != i:
                raise ValueError(f'DARP file: expected node id {i}, found {row["Id"]}')

            if row['Code'] != self.codes[i]:
                raise ValueError(f'DARP file: node {i} code {row["Code"]} differs from Codes file ({self.codes[i]})')

            # The file headers are swapped: column "Lat" holds the longitude (~ -71)
            # and column "Lng" the latitude (~ 46.8). So x = longitude, y = latitude.
            self.nodes.append(IDARPNode(
                id=int(row['Id']),
                code=row['Code'],
                x=float(row['Lat']),
                y=float(row['Lng']),
                service_time=float(row['Service-Time']),
                demand=int(float(row['Demand'])),
                ready_time=float(row['Ei']),
                due_time=float(row['Li']),
                allowed_to_integrate=int(float(row['AllowedToIntegrate'])),
                precision=self.precision,
            ))

        if len(self.nodes) != 2 * self.request_number + 1:
            raise ValueError(
                f'DARP file: {self.request_number} requests need {2 * self.request_number + 1} nodes, '
                f'found {len(self.nodes)}'
            )

        # Bus stops are every element after the DARP nodes
        self.bus_stops = [BusStop(i, self.codes[i]) for i in range(len(self.nodes), len(self.codes))]

    def _read_labeled_matrix(self, suffix: str) -> np.ndarray:
        ''' TimeMatrix / DistanceMatrix: header and first column hold the element codes '''

        lines = self._read_lines(suffix)  # 'Elements:', count, 'Code/Code <codes>', rows...
        count = int(lines[1])

        if count != len(self.codes) or lines[2].split()[1:] != self.codes:
            raise ValueError(f'{suffix} file: header does not match the Codes file')

        matrix = np.zeros((count, count), dtype=float)

        for i, line in enumerate(lines[3:]):
            label, *values = line.split()
            if label != self.codes[i] or len(values) != count:
                raise ValueError(f'{suffix} file: malformed row {i}')
            matrix[i] = values

        return matrix

    def _load_full_matrices(self):
        self.times = self._scale_matrix(self._read_labeled_matrix('TimeMatrix'))
        self.distances = self._scale_matrix(self._read_labeled_matrix('DistanceMatrix'))

    def _read_square_matrix(self, suffix: str) -> np.ndarray:
        ''' TimeBC / DistBC: first line is the size, then size rows with no labels '''

        lines = self._read_lines(suffix)
        size = int(lines[0])
        matrix = np.array([line.split() for line in lines[1:]], dtype=float)

        if matrix.shape != (size, size):
            raise ValueError(f'{suffix} file: expected {size}x{size}, found {matrix.shape}')

        return matrix

    def _load_common_carrier_matrices(self):
        self.times_cc = self._scale_matrix(self._read_square_matrix('TimeBC'))
        self.distances_cc = self._scale_matrix(self._read_square_matrix('DistBC'))

        expected = 2 * self.request_number + 2  # nodes + end depot
        if self.times_cc.shape[0] != expected:
            raise ValueError(f'Common carrier matrices: expected size {expected}, found {self.times_cc.shape[0]}')

    def _load_bus_lines(self):
        ''' BusLines file: one row per (line, bus, stop) '''

        lines = self._read_lines('BusLines')
        # Number-Of-Lines / n, Number-Of-Bus-Per-Line / ..., Number-Of-Bus-Station-Per-Line / ..., header, rows
        line_number = int(lines[1])
        buses_per_line = list(map(int, lines[3].split()))
        stops_per_line = list(map(int, lines[5].split()))
        columns = lines[6].split()

        self.bus_lines = [None] * line_number
        trips: dict[tuple[int, int], tuple[list[int], list[int]]] = {}  # (line, bus) -> (stop ids, times)

        for line in lines[7:]:
            row = dict(zip(columns, line.split()))
            line_id, bus = int(row['Id-Line']), int(row['Bus'])
            stop_id = int(row['Id'])

            if self.codes[stop_id] != row['Code']:
                raise ValueError(f'BusLines file: stop id {stop_id} does not match code {row["Code"]}')

            if self.bus_lines[line_id] is None:
                self.bus_lines[line_id] = BusLine(line_id, row['Line'], row['Name'])

            stop_ids, times = trips.setdefault((line_id, bus), ([], []))
            stop_ids.append(stop_id)
            times.append(self._scale(row['Time']))

        for (line_id, bus), (stop_ids, times) in trips.items():
            if len(stop_ids) != stops_per_line[line_id]:
                raise ValueError(
                    f'BusLines file: line {line_id} bus {bus} has {len(stop_ids)} stops, '
                    f'expected {stops_per_line[line_id]}'
                )
            self.bus_lines[line_id].trips.append(
                BusTrip(line_id, bus, np.array(stop_ids, dtype=np.int64), np.array(times, dtype=np.int64))
            )

        for bus_line in self.bus_lines:
            if len(bus_line.trips) != buses_per_line[bus_line.id]:
                raise ValueError(
                    f'BusLines file: line {bus_line.id} has {len(bus_line.trips)} buses, '
                    f'expected {buses_per_line[bus_line.id]}'
                )

    def _load_integrations(self):
        ''' IntegrationBusStation file (+ Cost file): best bus option for each integrable request.

        Row format (header: k n+k Line Bus i j BoardingTime LandingTime : cod(k) cod(n+k) cod(i) cod(j)).
        Everything after the ':' is redundant (codes) and ignored.
        '''

        lines = self._read_lines('IntegrationBusStation')
        count = int(lines[1])

        for line in lines[3:]:
            tokens = line.split()
            k, delivery, bus_line, bus, board, land = (int(t) for t in tokens[:6])

            if k in self.integrations:
                raise ValueError(f'IntegrationBusStation file: request {k} appears more than once')

            self.integrations[k] = Integration(
                request=k,
                delivery=delivery,
                line=bus_line,
                bus=bus,
                boarding_stop=board,
                landing_stop=land,
                boarding_time=self._scale(tokens[6]),
                landing_time=self._scale(tokens[7]),
            )

        if len(self.integrations) != count:
            raise ValueError(
                f'IntegrationBusStation file: declared {count} integrations, found {len(self.integrations)}'
            )

        # Cost file: Req, Cost, Mode
        lines = self._read_lines('Cost')
        count = int(lines[1])
        rows = lines[3:]

        if len(rows) != count:
            raise ValueError(f'Cost file: declared {count} integrations, found {len(rows)}')

        for line in rows:
            req, cost, mode = line.split()[:3]
            if int(req) not in self.integrations:
                raise ValueError(f'Cost file: request {req} has no entry in IntegrationBusStation file')

            self.integrations[int(req)].cost = float(cost)
            self.integrations[int(req)].mode = int(mode) if mode.lstrip('-').isdigit() else mode

    # ------------------------------------------------------------------ preprocessing

    def _tighten_time_windows(self):
        ''' Time window tightening (same rules as `Instance`), using the time matrix as travel time. '''

        for i in range(1, self.request_number + 1):
            pickup, delivery = self.nodes[i], self.nodes[i + self.request_number]
            direct = int(self.times[pickup.id, delivery.id])

            # Earliest pickup time considering the maximum ride time constraint
            pickup.ready_time = max(
                pickup.ready_time,
                delivery.ready_time - pickup.service_time - self.max_request_time
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
                pickup.due_time + pickup.service_time + self.max_request_time
            )

    def __repr__(self):
        return (
            f'Problem({self.name}: {self.request_number} requests, {self.vehicle_number} vehicles, '
            f'{len(self.bus_stops)} bus stops, {len(self.bus_lines)} bus lines, '
            f'{len(self.integrations)} integrations)'
        )