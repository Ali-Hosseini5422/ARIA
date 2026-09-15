from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig


@dataclass
class SlotState:
    t: int
    worker_xy: np.ndarray
    energy: np.ndarray
    task_xy: np.ndarray
    task_w: np.ndarray
    cost: np.ndarray
    bid: np.ndarray
    quality: np.ndarray
    idle: np.ndarray
    quit: np.ndarray
    attacker: np.ndarray
    encoding: np.ndarray
    inflow: float
    shock: float


class CrowdWorld:
    """Paired synthetic city. One seed yields one physical world for every selector.

    Attacker bits are always drawn. Extreme encodings are applied only when
    ``cfg.attack`` is True, after the physical RNG draws, so positions, tasks,
    shocks, and attacker membership stay paired.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        n = cfg.n
        self.home = self.rng.uniform(0, cfg.grid, size=(n, 2))
        self.speed = self.rng.uniform(0.4, 1.8, size=n)
        alpha0, lam0, gam0 = cfg.cpt_center
        self.alpha = np.clip(self.rng.normal(alpha0, 0.06, n), 0.4, 1.0)
        self.lam = np.clip(self.rng.normal(lam0, 0.25, n), 1.0, 3.5)
        self.gamma = np.clip(self.rng.normal(gam0, 0.08, n), 0.3, 1.0)
        self.beta_h = np.clip(self.rng.normal(0.75, 0.12, n), 0.3, 1.0)
        self.true_cost = self.rng.uniform(0.35, 1.40, size=n)
        self.qmax = self.rng.uniform(0.55, 1.0, size=n)
        self.strategic = self.rng.random(n) < cfg.strategic_frac
        self.attacker = self.rng.random(n) < cfg.attack_frac
        self.xy = self.home.copy()
        self.energy = self.rng.uniform(0.4, 1.0, size=n)
        self.last_pay = np.zeros(n)
        self.last_q = np.zeros(n)
        self.idle = np.ones(n, dtype=bool)
        self.abd_rate = 0.08
        self.last_uncert = 0.10

    def _move(self) -> None:
        drift = self.rng.normal(0.0, 0.35, size=self.xy.shape)
        self.xy = np.clip(self.xy + drift * self.speed[:, None], 0, self.cfg.grid)
        self.energy = np.clip(self.energy - 0.03 + 0.02 * self.rng.random(self.cfg.n), 0.05, 1.0)

    def sample_slot(self, t: int) -> SlotState:
        self._move()
        n_tasks = int(self.rng.poisson(self.cfg.task_mean))
        n_tasks = max(8, n_tasks)
        task_xy = self.rng.uniform(0, self.cfg.grid, size=(n_tasks, 2))
        task_w = self.rng.uniform(0.6, 1.4, size=n_tasks)
        bid = self.true_cost.copy()
        bid[self.strategic] *= 1.0 + self.cfg.shade
        encoding = np.stack(
            [self.alpha, self.lam, self.gamma, self.beta_h, self.energy],
            axis=1,
        )
        if self.cfg.attack:
            encoding[self.attacker] = np.array([0.2, 3.4, 0.2, 0.2, 1.0])
        shock_scale = self.cfg.shock_frac * self.cfg.expected_b()
        shock = float(self.rng.uniform(-shock_scale, shock_scale))
        quit = (self.rng.random(self.cfg.n) < 0.07) | (self.energy < 0.12)
        return SlotState(
            t=t,
            worker_xy=self.xy.copy(),
            energy=self.energy.copy(),
            task_xy=task_xy,
            task_w=task_w,
            cost=self.true_cost.copy(),
            bid=bid,
            quality=self.qmax.copy(),
            idle=self.idle.copy(),
            quit=quit,
            attacker=self.attacker.copy(),
            encoding=encoding,
            inflow=self.cfg.inflow,
            shock=shock,
        )

    def observe_outcome(self, winners: np.ndarray, pay: np.ndarray, q: np.ndarray) -> None:
        self.last_pay[:] = 0.0
        self.last_q[:] = 0.0
        if winners.size:
            self.last_pay[winners] = pay[winners]
            self.last_q[winners] = q[winners]
            self.idle[:] = True
            self.idle[winners] = False
            self.last_uncert = float(np.std(q[winners]))
        else:
            self.idle[:] = True
            self.last_uncert = 0.5 * self.last_uncert + 0.5 * 0.20
        hired = winners.size / max(self.cfg.n, 1)
        self.abd_rate = 0.7 * self.abd_rate + 0.3 * (1.0 - hired)
