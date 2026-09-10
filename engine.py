from __future__ import annotations

from dataclasses import replace
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
        abd = 0.10
        ep = EpisodeMetrics()
        shocks: List[float] = []
        for t in range(self.cfg.t_slots):
            slot = world.sample_slot(t)
            uncert = float(slot.encoding[:, 2].std())
            vol = float(np.std(shocks[-5:])) / max(self.cfg.expected_b(), 1.0) if shocks else 0.1
            r = self.regime(B, slot.task_xy.shape[0], abd, uncert, vol)
            alloc = self.selector.allocate(slot, r, Z, B)
            spend = min(alloc.spend, B)
            if alloc.spend > B + 1e-6:
                # last-winner dropping already applied inside selectors; clip residual numerically
                spend = B
            welfare = alloc.value - spend - self.cfg.psi * alloc.winners.size
            worker_u = 0.0
            if alloc.winners.size:
                worker_u = float((alloc.pay[alloc.winners] - slot.cost[alloc.winners]).mean())
            x = getattr(alloc, "x", None)
            if x is None:
                x = np.zeros(5)
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
                    x=np.zeros(5),
                )
            )
            world.observe_outcome(alloc.winners, alloc.pay, alloc.quality)
            B = max(B - spend + slot.inflow + slot.shock, 0.0)
            Z = max(Z + spend - slot.inflow - slot.shock - self.cfg.expected_b(), 0.0)
            abd = world.abd_rate
            shocks.append(slot.shock)
        return ep


def run_selector(name: str, cfg: SimConfig):
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
        sel = LockedVertex(cfg, name.split("_", 1)[1])
    else:
        sel = table[name](cfg)
    return Simulator(cfg, sel).run()
