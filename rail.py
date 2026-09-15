from __future__ import annotations

from .config import SimConfig


def gas_per_hire(cfg: SimConfig) -> int:
    """Groth16-BN254 verify + pay + calldata accounting only. Not a chain send."""
    return int(cfg.gas_per_hire)


def gas_for_hires(hires: int, cfg: SimConfig) -> int:
    return int(hires) * gas_per_hire(cfg)
