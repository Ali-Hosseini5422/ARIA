from __future__ import annotations

from dataclasses import dataclass, field

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
    x: np.ndarray = field(default_factory=lambda: np.zeros(5))


def characteristic_value(
    winners: np.ndarray,
    slot: SlotState,
    cfg: SimConfig,
    phi: np.ndarray | None = None,
) -> tuple[float, int]:
    if phi is None:
        phi = coverage_kernel(slot.worker_xy, slot.task_xy)
    n_tasks = slot.task_xy.shape[0]
    if winners.size == 0:
        return 0.0, 0
    cov = phi[winners].max(axis=0)
    covered = int((cov > cfg.cover_phi).sum())
    quality = float(slot.quality[winners].mean())
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
    """IR-tight greedy cover of yet-uncovered tasks. Pay equals cost."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def run(
        self, slot: SlotState, budget: float, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = slot.cost.size
        pay = np.zeros(n)
        chosen: list[int] = []
        covered = np.zeros(phi.shape[1], dtype=bool)
        remaining = float(budget)
        # cheapest feasible cover first (IR-tight)
        order = np.argsort(slot.cost)
        for i in order:
            if remaining < slot.cost[i]:
                continue
            useful = phi[i] > self.cfg.cover_phi
            if not np.any(useful & ~covered):
                continue
            chosen.append(int(i))
            pay[i] = slot.cost[i]
            remaining -= slot.cost[i]
            covered |= useful
        return np.asarray(chosen, dtype=int), pay, budget - remaining


class CriticalRankDrop:
    """Monotone critical-value payments on a ranked prefix + last-winner drop.

    The payment is a cost-floor critical value on the score ranking, not a
    Euclidean projection onto the budget simplex (which could cut below cost).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def run(
        self,
        slot: SlotState,
        scores: np.ndarray,
        budget: float,
        blocked: np.ndarray,
        delta: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = scores.size
        pay = np.zeros(n)
        eligible = np.ones(n, dtype=bool)
        if blocked.size:
            eligible[blocked] = False
        if delta is None:
            delta = np.zeros(n)
        order = np.argsort(-scores)
        order = order[eligible[order]]
        if order.size == 0 or budget <= 0:
            return np.asarray([], dtype=int), pay, 0.0

        crit = np.zeros(n)
        for k, i in enumerate(order):
            nxt = scores[order[k + 1]] if k + 1 < order.size else 0.0
            gap = max(float(scores[i] - nxt), 0.0)
            crit[i] = max(slot.cost[i], slot.cost[i] + 0.35 * gap * slot.quality[i])

        # longest feasible prefix (last-winner dropping)
        prices = crit[order] + delta[order]
        csum = np.cumsum(prices)
        keep = int(np.searchsorted(csum, budget + 1e-9, side="right"))
        prefix = order[:keep]
        if keep:
            pay[prefix] = prices[:keep]
            spend = float(prices[:keep].sum())
        else:
            spend = 0.0
        return np.asarray(prefix, dtype=int), pay, spend
