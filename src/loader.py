from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.models import BusLine, BusTrip, Integration, Node

# Floating point noise tolerated before rounding times to integers (120.0000001 stays 120)
ROUNDING_TOLERANCE = 1e-6


def read_lines(folder: Path, suffix: str) -> list[str]:
    ''' Non-empty, stripped lines of <folder>/<name>_<suffix>.txt '''

    path = folder / f'{folder.name}_{suffix}.txt'

    if not path.is_file():
        raise FileNotFoundError(f'IDARP instance file not found: {path}')

    with open(path, 'r') as file:
        return [line.strip() for line in file if line.strip()]


def load_codes(folder: Path) -> list[str]:
    lines = read_lines(folder, 'Codes')
    codes = []

    for i, line in enumerate(lines[3:]):
        id, code = line.split()
        assert int(id) == i
        codes.append(code)

    assert len(codes) == int(lines[1])

    return codes


def load_darp(folder: Path, codes: list[str]) -> tuple[dict[str, str], list[Node]]:
    ''' DARP file: configuration + nodes. '''

    lines = read_lines(folder, 'DARP')
    config = dict(zip(lines[0].split(), lines[1].split()))
    columns = lines[2].split()
    nodes = []

    for i, line in enumerate(lines[3:]):
        row = dict(zip(columns, line.split()))

        assert int(row['Id']) == i
        assert row['Code'] == codes[i]

        # Column 'Lat' holds longitude and column 'Lng' holds latitude.
        nodes.append(Node(
            id=int(row['Id']),
            code=row['Code'],
            x=float(row['Lat']),
            y=float(row['Lng']),
            service_time=int(round(float(row['Service-Time']))),
            demand=int(row['Demand']),
            # Conservative rounding (ready up, due down): a schedule valid on the rounded data is
            # valid on the original, and the CP-SAT model and Route.feasible agree.
            ready_time=math.ceil(float(row['Ei']) - ROUNDING_TOLERANCE),
            due_time=math.floor(float(row['Li']) + ROUNDING_TOLERANCE),
            allowed_to_integrate=bool(int(row['AllowedToIntegrate'])),
        ))

    assert len(nodes) == 2 * int(config['Requests']) + 1

    return config, nodes


def _labeled_matrix(folder: Path, suffix: str, codes: list[str]) -> np.ndarray:
    ''' Header and first column hold the element codes. '''

    lines = read_lines(folder, suffix)
    count = int(lines[1])

    assert count == len(codes) and lines[2].split()[1:] == codes

    matrix = np.zeros((count, count), dtype=float)

    for i, line in enumerate(lines[3:]):
        label, *values = line.split()

        assert label == codes[i] and len(values) == count
        matrix[i] = values

    return matrix


def _square_matrix(folder: Path, suffix: str) -> np.ndarray:
    ''' First line is the size, then `size` rows without labels. '''

    lines = read_lines(folder, suffix)
    size = int(lines[0])
    matrix = np.array([line.split() for line in lines[1:]], dtype=float)

    assert matrix.shape == (size, size)

    return matrix


def load_matrices(folder: Path, codes: list[str], requests: int):
    ''' Returns (times, distances, times_cc, distances_cc). Travel times are rounded up to whole seconds. '''

    times = np.ceil(_labeled_matrix(folder, 'TimeMatrix', codes) - ROUNDING_TOLERANCE)
    distances = _labeled_matrix(folder, 'DistanceMatrix', codes)
    times_cc = _square_matrix(folder, 'TimeBC')
    distances_cc = _square_matrix(folder, 'DistBC')

    assert times_cc.shape[0] == 2 * requests + 2  # nodes + end depot

    return times, distances, times_cc, distances_cc


def load_bus_lines(folder: Path, codes: list[str]) -> list[BusLine]:
    ''' BusLines file: one row per (line, bus, stop). '''

    lines = read_lines(folder, 'BusLines')
    bus_lines: list[BusLine | None] = [None] * int(lines[1])
    buses_per_line = list(map(int, lines[3].split()))
    stops_per_line = list(map(int, lines[5].split()))
    columns = lines[6].split()
    trips: dict[tuple[int, int], tuple[list[int], list[int]]] = {}  # (line, bus) -> (stops, times)

    for line in lines[7:]:
        row = dict(zip(columns, line.split()))
        line_id, bus = int(row['Id-Line']), int(row['Bus'])
        stop_id, stop_time = int(row['Id']), float(row['Time'])

        assert codes[stop_id] == row['Code']

        if bus_lines[line_id] is None:
            bus_lines[line_id] = BusLine(line_id, row['Line'], row['Name'])

        stop_ids, stop_times = trips.setdefault((line_id, bus), ([], []))
        stop_ids.append(stop_id)
        stop_times.append(stop_time)

    for (line_id, bus), (stop_ids, stop_times) in trips.items():
        assert len(stop_ids) == stops_per_line[line_id]

        bus_lines[line_id].trips.append(BusTrip(
            line_id, bus, np.array(stop_ids, dtype=np.int64), np.array(stop_times, dtype=np.int64)
        ))

    assert all(len(line.trips) == buses_per_line[line.id] for line in bus_lines)

    return bus_lines


def load_integrations(folder: Path) -> dict[int, Integration]:
    ''' IntegrationBusStation + Cost files: best bus option of each integrable request. '''

    lines = read_lines(folder, 'IntegrationBusStation')
    integrations: dict[int, Integration] = {}

    for line in lines[3:]:
        tokens = line.split()
        request, delivery, bus_line, bus, board, land = map(int, tokens[:6])

        assert request not in integrations

        integrations[request] = Integration(
            request=request,
            delivery=delivery,
            line=bus_line,
            bus=bus,
            boarding_stop=board,
            landing_stop=land,
            boarding_time=float(tokens[6]),
            landing_time=float(tokens[7]),
        )

    assert len(integrations) == int(lines[1])

    lines = read_lines(folder, 'Cost')  # Req, Cost, Mode
    rows = lines[3:]

    assert len(rows) == int(lines[1])

    for line in rows:
        request, cost, mode = line.split()

        assert int(request) in integrations

        integrations[int(request)].cost = float(cost)
        integrations[int(request)].mode = mode

    return integrations
