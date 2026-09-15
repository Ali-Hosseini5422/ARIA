from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SlotMetrics:
    welfare: float
    covered: int
    n_tasks: int
    spend: float
    hires: int
    delayed_frac: float
    ir_violation: float
    worker_util: float
    x: np.ndarray


@dataclass
class EpisodeMetrics:
    slots: List[SlotMetrics] = field(default_factory=list)

    def add(self, m: SlotMetrics) -> None:
        self.slots.append(m)

    def summary(self) -> dict:
        w = np.array([s.welfare for s in self.slots])
        tcr = np.array([s.covered / max(s.n_tasks, 1) for s in self.slots])
        icr = 1.0 - float(np.mean([s.delayed_frac for s in self.slots]))
        irv = float(np.mean([s.ir_violation for s in self.slots]))
        util = float(np.mean([s.worker_util for s in self.slots]))
        xs = np.stack([s.x for s in self.slots], axis=0)
        return {
            "W_P": float(w.sum()),
            "TCR": float(100.0 * tcr.mean()),
            "ICR": float(icr),
            "Ubar": util,
            "IR_v": irv,
            "hires": int(sum(s.hires for s in self.slots)),
            "x_mean": xs.mean(axis=0),
            "x_path": xs,
        }
