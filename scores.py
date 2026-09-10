from __future__ import annotations

import numpy as np

from .config import SimConfig
from .world import SlotState


def coverage_kernel(worker_xy: np.ndarray, task_xy: np.ndarray, scale: float = 3.2) -> np.ndarray:
    """phi[i, j] in [0, 1], Gaussian proximity."""
    d2 = ((worker_xy[:, None, :] - task_xy[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * scale**2))


class BaseScores:
    """Closed-form five coordinates. No sequence model."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def phi(self, slot: SlotState) -> np.ndarray:
        return coverage_kernel(slot.worker_xy, slot.task_xy)

    def public_residual(self, slot: SlotState, phi: np.ndarray) -> np.ndarray:
        rare = (phi > self.cfg.cover_phi).sum(axis=0)
        rarity = 1.0 / np.maximum(rare, 1.0)
        return (phi * rarity[None, :] * slot.task_w[None, :]).sum(axis=1)

    def compute(self, slot: SlotState) -> np.ndarray:
        phi = self.phi(slot)
        cover = (phi * slot.task_w[None, :]).max(axis=1)
        energy = slot.energy
        q = slot.quality
        cost = np.maximum(slot.bid, 1e-6)
        surplus = np.maximum(cover * q - cost, 0.0)
        s_ref = surplus / (1.0 + cost)
        s_cpt = cover * q * energy / cost
        s_ax = cover * q * (1.0 - 0.3 * slot.idle.astype(float))
        s_del = energy * q / cost
        s_rsc = slot.quit.astype(float) * cover * q
        s = np.stack([s_ref, s_cpt, s_ax, s_del, s_rsc], axis=1)
        s = np.maximum(s, 0.0)
        denom = s.max(axis=0, keepdims=True)
        denom = np.where(denom < 1e-9, 1.0, denom)
        return s / denom
