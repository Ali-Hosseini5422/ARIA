from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SimConfig:
    """Constants from Table II of the ARIA manuscript."""

    n: int = 500
    t_slots: int = 24
    grid: int = 32
    task_mean: float = 80.0
    b0: float = 220.0
    inflow: float = 90.0
    shock_frac: float = 0.25
    psi: float = 0.15
    kappa_quality: float = 0.05
    cpt_center: Tuple[float, float, float] = (0.88, 2.25, 0.65)
    strategic_frac: float = 0.40
    shade: float = 0.08
    attack_frac: float = 0.20
    cover_phi: float = 0.18
    v_weights: Tuple[float, float, float] = (0.65, 0.25, 0.10)
    eta_band: Tuple[float, float] = (0.45, 0.82)
    expected_budget: float = 220.0
    posted_price: float = 0.85
    gas_per_hire: int = 281_292
    seed: int = 0
    attack: bool = False
    instrument_names: Tuple[str, ...] = ("ref", "cpt", "ax", "del", "rsc")

    def expected_b(self) -> float:
        return self.expected_budget
