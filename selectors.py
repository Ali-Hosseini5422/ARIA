from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SimConfig
from .field import InstrumentField, PrivateMap, PublicMap, project_to_field
from .scores import BaseScores
from .settlement import Allocation, CriticalRankDrop, PhaseACover, characteristic_value
from .world import SlotState


def _eta(r: np.ndarray, cfg: SimConfig) -> float:
    lo, hi = cfg.eta_band
    tightness = np.clip(1.0 - float(r[0]), 0.0, 1.0)
    density = np.clip(float(r[1]) / 2.0, 0.0, 1.0)
    return float(np.clip(lo + 0.25 * density + 0.15 * tightness, lo, hi))


def _kappa(r: np.ndarray) -> float:
    return float(np.clip(0.08 + 0.12 * (1.0 - float(r[0])), 0.05, 0.25))


def _finish(
    slot: SlotState,
    cfg: SimConfig,
    winners: np.ndarray,
    pay: np.ndarray,
    phi: np.ndarray,
    x: np.ndarray,
    delayed_frac: float,
) -> Allocation:
    q = np.zeros(slot.cost.size)
    if winners.size:
        q[winners] = slot.quality[winners]
    value, covered = characteristic_value(winners, slot, cfg, phi)
    spend = float(pay[winners].sum()) if winners.size else 0.0
    welfare = value - spend - cfg.psi * winners.size
    ir = 0.0
    if winners.size:
        slip = np.maximum(slot.cost[winners] - pay[winners], 0.0)
        ir = float((slip > 1e-8).mean())
    util = float((pay[winners] - slot.cost[winners]).mean()) if winners.size else 0.0
    return Allocation(
        winners=winners,
        pay=pay,
        quality=q,
        spend=spend,
        phase_a=np.array([], dtype=int),
        phase_b=winners,
        delayed_frac=delayed_frac,
        ir_violation=ir,
        covered_tasks=covered,
        n_tasks=slot.task_xy.shape[0],
        value=value,
        # welfare stored via value-spend-psi by engine
    )


class Selector:
    name = "base"

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        raise NotImplementedError


class PostedPrice(Selector):
    name = "posted"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        from .scores import coverage_kernel

        phi = coverage_kernel(slot.worker_xy, slot.task_xy)
        price = self.cfg.posted_price
        order = np.argsort(-phi.max(axis=1))
        winners = []
        pay = np.zeros(slot.cost.size)
        spend = 0.0
        hire_cap = max(6, int(0.18 * slot.cost.size))
        for i in order:
            if len(winners) >= hire_cap:
                break
            if spend + price > B:
                break
            winners.append(i)
            pay[i] = price
            spend += price
        w = np.array(winners, dtype=int)
        alloc = _finish(slot, self.cfg, w, pay, phi, np.zeros(5), 0.0)
        return alloc


class GTDIMInterface(Selector):
    """Protocol-matched two-stage CPS + PMI interface, not vendor binary."""

    name = "gtdim_if"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        from .scores import coverage_kernel

        phi = coverage_kernel(slot.worker_xy, slot.task_xy)
        shortage = (phi > self.cfg.cover_phi).sum(axis=0)
        bonus = (phi * (1.0 / np.maximum(shortage, 1.0))[None, :]).sum(axis=1)
        pmi = 0.55 * slot.quality + 0.30 * phi.max(axis=1) + 0.15 * bonus / (1.0 + bonus)
        pmi = pmi / (1.0 + slot.bid)
        order = np.argsort(-pmi)
        winners = []
        pay = np.zeros(slot.cost.size)
        spend = 0.0
        for i in order:
            price = max(slot.cost[i], slot.cost[i] + 0.15 * bonus[i])
            if spend + price > B:
                continue
            winners.append(i)
            pay[i] = price
            spend += price
            if len(winners) >= max(8, min(slot.task_xy.shape[0] + 4, int(0.2 * slot.cost.size))):
                break
        w = np.array(winners, dtype=int)
        return _finish(slot, self.cfg, w, pay, phi, np.zeros(5), 0.0)


class CoverOnly(Selector):
    name = "cover_only"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.phase_a = PhaseACover(cfg)

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        from .scores import coverage_kernel

        phi = coverage_kernel(slot.worker_xy, slot.task_xy)
        winners, pay, _ = self.phase_a.run(slot, B, phi)
        return _finish(slot, self.cfg, winners, pay, phi, np.zeros(5), 0.0)


class _FieldMixin:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.scores = BaseScores(cfg)
        self.phase_a = PhaseACover(cfg)
        self.phase_b = CriticalRankDrop(cfg)
        self.public = PublicMap()
        self.private = PrivateMap()

    def _settle(
        self,
        slot: SlotState,
        x: np.ndarray,
        r: np.ndarray,
        B: float,
        use_cover: bool,
    ) -> Allocation:
        phi = self.scores.phi(slot)
        s = self.scores.compute(slot)
        pay = np.zeros(slot.cost.size)
        blocked = np.array([], dtype=int)
        spend_a = 0.0
        wa = np.array([], dtype=int)
        if use_cover:
            cap = _eta(r, self.cfg) * B
            wa, pay_a, spend_a = self.phase_a.run(slot, cap, phi)
            pay[wa] = pay_a[wa]
            blocked = wa
        sigma = s @ x + _kappa(r) * self.scores.public_residual(slot, phi)
        wb, pay_b, spend_b = self.phase_b.run(slot, sigma, B - spend_a, blocked)
        pay[wb] = pay_b[wb]
        winners = np.unique(np.concatenate([wa, wb])).astype(int)
        delayed = float(x[3])
        alloc = _finish(slot, self.cfg, winners, pay, phi, x, delayed)
        alloc.phase_a = wa
        alloc.phase_b = wb
        return alloc


class LockedVertex(_FieldMixin, Selector):
    def __init__(self, cfg: SimConfig, name: str):
        _FieldMixin.__init__(self, cfg)
        self.vname = name
        self.name = f"locked_{name}"
        self.x = InstrumentField.vertex(name)

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        x = self.x.copy()
        if self.vname == "rsc" and slot.quit.mean() < 0.05:
            x = project_to_field(x * 0.3)
        return self._settle(slot, x, r, B, use_cover=True)


class FieldOnly(_FieldMixin, Selector):
    name = "field_only"

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        x = self.public(r, z, self.cfg.expected_b())
        return self._settle(slot, x, r, B, use_cover=False)


class PublicARIA(_FieldMixin, Selector):
    name = "aria_public"

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        x = self.public(r, z, self.cfg.expected_b())
        return self._settle(slot, x, r, B, use_cover=True)


class PrivateMapARIA(_FieldMixin, Selector):
    name = "aria_private"

    def allocate(self, slot: SlotState, r: np.ndarray, z: float, B: float) -> Allocation:
        x = self.private(r, z, slot.encoding, self.cfg.expected_b())
        return self._settle(slot, x, r, B, use_cover=True)
