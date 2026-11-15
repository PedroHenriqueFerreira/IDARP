from __future__ import annotations

from dataclasses import dataclass

from src.lns.params import FAST, MIN_SIZE


@dataclass
class Budget:
    ''' Neighborhood size (routes in the model) and solver time per call, adapted to what the solver handles. '''

    n: int  # routes in the current solution
    iteration_time: float
    max_iteration_time: float
    size: int
    cap: int  # largest size worth trying (lowered by timeouts, relaxed by improvements)

    def more_time(self) -> bool:
        if self.iteration_time >= self.max_iteration_time:
            return False

        self.iteration_time = min(2 * self.iteration_time, self.max_iteration_time)
        self.cap = self.n

        return True

    def grow(self):
        self.size = min(self.size + 1, self.cap)

    def on_timeout(self):
        ''' Hard neighborhood: shrink and cap the failed size; at the smallest size, give more time. '''

        if self.size > MIN_SIZE:
            self.cap = self.size = self.size - 1
        else:
            self.more_time()

    def on_improved(self, spent: float):
        self.cap = min(self.cap + 1, self.n)

        if spent < FAST * self.iteration_time:
            self.grow()

    def on_stalled(self) -> str | None:
        ''' Every type is exhausted at this size: grow, else give the solver more time, else stop. '''

        if self.size < self.cap:
            self.size += 1
            return 'grow'

        return 'more time' if self.more_time() else None
