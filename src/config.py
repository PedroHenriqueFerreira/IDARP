from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    ''' Pipeline parameters. '''

    max_integrated: int | None = None  # cap on requests served by bus integration
    workers: int = -1  # CP-SAT workers (0 = all cores)
    seed: int = 0
    log: bool = True
