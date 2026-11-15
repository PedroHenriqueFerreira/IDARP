from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Alternatives of a request (tie-break order: private fleet first)
ROUTE, INTEGRATION, CARRIER = 0, 1, 2


@dataclass
class Node:
    ''' Depot, pickup or delivery. '''

    id: int
    code: str
    x: float  # longitude
    y: float  # latitude
    service_time: int
    demand: int  # pickup +1, delivery -1
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
    bus: int  # starts at 1
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


@dataclass(frozen=True)
class CostScheme:
    ''' Cost parameters (defaults: Québec City, Schenekemberg et al. 2025). '''

    private_per_km: float = 0.4  # private fleet, per km
    carrier_fixed: float = 3.0  # common carrier, fixed cost per request
    carrier_per_km: float = 2.0  # common carrier, per km
    bus_ticket: float = 3.0  # bus ticket per integrated request
