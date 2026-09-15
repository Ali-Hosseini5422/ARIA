from __future__ import annotations

from typing import List

import numpy as np

from .config import SimConfig
from .metrics import EpisodeMetrics, SlotMetrics
from .selectors import Selector
from .world import CrowdWorld


class Simulator:
    def __init__(self, cfg: SimConfig, selector: Selector):
        self.cfg = cfg
        self.selector = selector

    def regime(self, B: float, n_tasks: int, abd: float, uncert: float, vol: float) -> np.ndarray:
        """Public r_t. Frozen before current reports / encodings."""
        return np.array(
            [
                B / max(self.cfg.expected_b(), 1e-6),
                n_tasks / max(self.cfg.n, 1),
                abd,
                uncert,
                vol,
            ],
            dtype=float,
        )

    def run(self) -> EpisodeMetrics:
        world = CrowdWorld(self.cfg)
        B = float(self.cfg.b0)
        Z = 0.0
        abd = float(world.abd_rate)
        uncert = float(world.last_uncert)
        ep = EpisodeMetrics()
        shocks: List[float] = []
        for t in range(self.cfg.t_slots):
            slot = world.sample_slot(t)
            vol = (
                float(np.std(shocks[-5:])) / max(self.cfg.expected_b(), 1.0)
                if shocks
                else 0.1
            )
            # r uses lagged abandonment and lagged realized uncertainty only
            r = self.regime(B, slot.task_xy.shape[0], abd, uncert, vol)
            alloc = self.selector.allocate(slot, r, Z, B)
            spend = min(float(alloc.spend), B)
            welfare = alloc.value - spend - self.cfg.psi * alloc.winners.size
            worker_u = 0.0
            if alloc.winners.size:
                worker_u = float((alloc.pay[alloc.winners] - slot.cost[alloc.winners]).mean())
            x = np.asarray(alloc.x, dtype=float)
            ep.add(
                SlotMetrics(
                    welfare=float(welfare),
                    covered=alloc.covered_tasks,
                    n_tasks=alloc.n_tasks,
                    spend=spend,
                    hires=int(alloc.winners.size),
                    delayed_frac=alloc.delayed_frac,
                    ir_violation=alloc.ir_violation,
                    worker_util=worker_u,
                    x=x.copy(),
                )
            )
            world.observe_outcome(alloc.winners, alloc.pay, alloc.quality)
            B = max(B - spend + slot.inflow + slot.shock, 0.0)
            Z = max(Z + spend - slot.inflow - slot.shock - self.cfg.expected_b(), 0.0)
            abd = float(world.abd_rate)
            uncert = float(world.last_uncert)
            shocks.append(slot.shock)
        return ep


def make_selector(name: str, cfg: SimConfig) -> Selector:
    from .selectors import (
        CoverOnly,
        FieldOnly,
        GTDIMInterface,
        LockedVertex,
        PostedPrice,
        PrivateMapARIA,
        PublicARIA,
    )

    table = {
        "posted": PostedPrice,
        "gtdim_if": GTDIMInterface,
        "cover_only": CoverOnly,
        "field_only": FieldOnly,
        "aria_public": PublicARIA,
        "aria_private": PrivateMapARIA,
    }
    if name.startswith("locked_"):
        return LockedVertex(cfg, name.split("_", 1)[1])
    return table[name](cfg)


def run_selector(name: str, cfg: SimConfig) -> EpisodeMetrics:
    return Simulator(cfg, make_selector(name, cfg)).run()
