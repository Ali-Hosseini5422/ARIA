from __future__ import annotations

import numpy as np


def project_to_field(x: np.ndarray, allow_rescue: bool = True) -> np.ndarray:
    """Project onto X: x >= 0, ||x||_1 <= 1, ref * cpt = 0, rescue optional."""
    x = np.maximum(np.asarray(x, dtype=float).copy(), 0.0)
    if x[0] * x[1] > 0:
        if x[0] >= x[1]:
            x[1] = 0.0
        else:
            x[0] = 0.0
    if not allow_rescue:
        x[4] = 0.0
    s = float(x.sum())
    if s > 1.0:
        x = x / s
    return x


class InstrumentField:
    names = ("ref", "cpt", "ax", "del", "rsc")

    @staticmethod
    def vertex(name: str) -> np.ndarray:
        x = np.zeros(5)
        idx = {"ref": 0, "cpt": 1, "ax": 2, "del": 3, "rsc": 4}[name]
        x[idx] = 1.0
        return x

    @staticmethod
    def origin() -> np.ndarray:
        return np.zeros(5)


class PublicMap:
    """Closed-form Lipschitz map pi(r, Z). Does not read encodings or reports."""

    def __init__(self):
        self.W = np.array(
            [
                [0.55, -0.10, 0.05, 0.05, 0.20, 0.10],
                [-0.15, 0.45, -0.05, 0.15, -0.05, -0.05],
                [0.05, 0.10, 0.05, 0.25, 0.05, 0.00],
                [0.10, 0.05, 0.15, 0.05, 0.10, 0.15],
                [-0.05, 0.05, 0.55, 0.05, 0.05, 0.05],
            ]
        )
        self.b = np.array([0.08, 0.18, 0.08, 0.10, 0.04])

    def __call__(self, r: np.ndarray, z: float, expected_b: float = 220.0) -> np.ndarray:
        feat = np.concatenate([np.asarray(r, dtype=float), [z / max(expected_b, 1e-6)]])
        raw = self.W @ feat + self.b
        allow_rescue = float(r[2]) > 0.12
        x = project_to_field(raw, allow_rescue=allow_rescue)
        tightness = 1.0 - float(r[0])
        if tightness > 0.25 and x[0] > 0 and x[1] > 0:
            x[1] = 0.0
            x = project_to_field(x, allow_rescue=allow_rescue)
        return x


class PrivateMap:
    """Ablation: pi also reads the pool-mean encoding. Movable by a coalition."""

    def __init__(self):
        self.public = PublicMap()

    def __call__(
        self,
        r: np.ndarray,
        z: float,
        encodings: np.ndarray,
        expected_b: float = 220.0,
    ) -> np.ndarray:
        x = self.public(r, z, expected_b)
        zbar = np.asarray(encodings, dtype=float).mean(axis=0)
        pull = np.array(
            [
                zbar[1] / 4.0,
                1.0 - zbar[0],
                zbar[2],
                1.0 - zbar[3],
                zbar[1] / 5.0,
            ]
        )
        return project_to_field(0.7 * x + 0.3 * pull)
