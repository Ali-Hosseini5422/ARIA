from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SimConfig
from .scores import coverage_kernel
from .world import SlotState


@dataclass
class Allocation:
    winners: np.ndarray
    pay: np.ndarray
    quality: np.ndarray
    spend: float
    phase_a: np.ndarray
    phase_b: np.ndarray
    delayed_frac: float
    ir_violation: float
    covered_tasks: int
    n_tasks: int
    value: float


def characteristic_value(
    winners: np.ndarray,
    slot: SlotState,
    cfg: SimConfig,
    phi: np.ndarray | None = None,
) -> Tuple[float, int]:
    if phi is None:
        phi = coverage_kernel(slot.worker_xy, slot.task_xy)
    n_tasks = slot.task_xy.shape[0]
    if winners.size == 0:
        return 0.0, 0
    cov = phi[winners].max(axis=0) if winners.size else np.zeros(n_tasks)
    covered = int((cov > cfg.cover_phi).sum())
    quality = float(slot.quality[winners].mean()) if winners.size else 0.0
    if winners.size >= 2:
        xy = slot.worker_xy[winners]
        d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        overlap = float(np.exp(-d / 4.0).mean())
    else:
        overlap = 0.0
    a, b, w = cfg.v_weights
    unit = 18.0
    val = unit * (
        a * float((cov * slot.task_w).sum())
        + b * quality * min(winners.size, covered + 2)
        - w * overlap * winners.size
    )
    return float(val), covered


class PhaseACover:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def run(self, slot: SlotState, budget: float, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        n = slot.cost.size
        pay = np.zeros(n)
        chosen = []
        covered = np.zeros(phi.shape[1], dtype=bool)
        remaining = budget
        order = np.argsort(slot.cost)
        for i in order:
            if remaining < slot.cost[i]:
                continue
            useful = phi[i] > self.cfg.cover_phi
            if not np.any(useful & ~covered):
                continue
            chosen.append(i)
            pay[i] = slot.cost[i]
            remaining -= slot.cost[i]
            covered |= useful
        return np.array(chosen, dtype=int), pay, budget - remaining


class CriticalRankDrop:
    """Critical-value payments on a ranked prefix + last-winner dropping."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def run(
        self,
        slot: SlotState,
        scores: np.ndarray,
        budget: float,
        blocked: np.ndarray,
        delta: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = scores.size
        pay = np.zeros(n)
        eligible = np.ones(n, dtype=bool)
        eligible[blocked] = False
        if delta is None:
            delta = np.zeros(n)
        order = np.argsort(-scores)
        order = order[eligible[order]]
        if order.size == 0 or budget <= 0:
            return np.array([], dtype=int), pay, 0.0
        # Approximate critical value: payment that keeps i above next eligible score.
        crit = np.zeros(n)
        for k, i in enumerate(order):
            nxt = scores[order[k + 1]] if k + 1 < order.size else 0.0
            # monotone transform: higher score -> not lower payment floor at cost
            gap = max(scores[i] - nxt, 0.0)
            crit[i] = max(slot.cost[i], slot.cost[i] + 0.35 * gap * slot.quality[i])
        prefix = []
        spend = 0.0
        hire_cap = max(6, int(0.18 * n))
        for i in order:
            if len(prefix) >= hire_cap:
                break
            price = crit[i] + delta[i]
            if spend + price <= budget + 1e-9:
                prefix.append(i)
                pay[i] = price
                spend += price
            else:
                break
        winners = np.array(prefix, dtype=int)
        return winners, pay, spend
