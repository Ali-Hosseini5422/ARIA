#!/usr/bin/env python3
"""CrowdSim benchmark used by the ARIA manuscript.

Smoke:
    python sim/crowdsim_bench.py
Paper scale:
    python sim/crowdsim_bench.py --paper
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aria.config import SimConfig
from aria.engine import run_selector
from aria.rail import gas_for_hires

MAIN_SELECTORS = [
    "posted",
    "gtdim_if",
    "locked_ref",
    "locked_cpt",
    "locked_ax",
    "locked_del",
    "locked_rsc",
    "cover_only",
    "field_only",
    "aria_private",
    "aria_public",
]

LABEL = {
    "posted": "Posted price",
    "gtdim_if": "GTDIM-IF",
    "locked_ref": "Locked e_ref",
    "locked_cpt": "Locked e_cpt",
    "locked_ax": "Locked e_ax",
    "locked_del": "Locked e_del",
    "locked_rsc": "Locked e_rsc",
    "cover_only": "Cover-only prefix",
    "field_only": "Field-only pi",
    "aria_private": "Private pi(r,Z,zbar)",
    "aria_public": "ARIA public pi(r,Z)",
}


def _mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    return float(a.mean()), float(a.std(ddof=0))


def run_panel(seeds: list[int], n: int, t_slots: int, attack: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for name in MAIN_SELECTORS:
            cfg = SimConfig(n=n, t_slots=t_slots, seed=seed, attack=attack)
            ep = run_selector(name, cfg)
            s = ep.summary()
            rows.append(
                {
                    "seed": seed,
                    "selector": name,
                    "attack": int(attack),
                    "W_P": s["W_P"],
                    "TCR": s["TCR"],
                    "ICR": s["ICR"],
                    "Ubar": s["Ubar"],
                    "IR_v": s["IR_v"],
                    "hires": s["hires"],
                    "x_path": s["x_path"],
                }
            )
    return pd.DataFrame(rows)


def summarize_main(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in MAIN_SELECTORS:
        sub = df[df["selector"] == name]
        wp_m, wp_s = _mean_std(sub["W_P"].tolist())
        tc_m, tc_s = _mean_std(sub["TCR"].tolist())
        ic_m, _ = _mean_std(sub["ICR"].tolist())
        ub_m, ub_s = _mean_std(sub["Ubar"].tolist())
        ir_m, _ = _mean_std(sub["IR_v"].tolist())
        out.append(
            {
                "Method": LABEL[name],
                "W_P": f"{wp_m:.0f}±{wp_s:.0f}",
                "TCR_%": f"{tc_m:.1f}±{tc_s:.1f}",
                "ICR": f"{ic_m:.3f}",
                "Ubar": f"{ub_m:.3f}±{ub_s:.3f}",
                "IR_v": f"{ir_m:.3f}",
            }
        )
    return pd.DataFrame(out)


def attack_table(seeds: list[int], n: int, t_slots: int) -> pd.DataFrame:
    """Paired myopic vs attack. Public Delta x must be 0 by construction of pi(r,Z)."""
    rows = []
    for seed in seeds:
        for name in ("aria_public", "aria_private"):
            cfg0 = SimConfig(n=n, t_slots=t_slots, seed=seed, attack=False)
            cfg1 = replace(cfg0, attack=True)
            ep0 = run_selector(name, cfg0)
            ep1 = run_selector(name, cfg1)
            s0, s1 = ep0.summary(), ep1.summary()
            dx = np.linalg.norm(s1["x_path"] - s0["x_path"], axis=1).mean()
            rows.append(
                {
                    "selector": name,
                    "seed": seed,
                    "TCR0": s0["TCR"],
                    "TCR1": s1["TCR"],
                    "WP0": s0["W_P"],
                    "WP1": s1["W_P"],
                    "dx": float(dx),
                }
            )
    df = pd.DataFrame(rows)
    out = []
    for name, title in (
        ("aria_public", "public pi(r,Z)"),
        ("aria_private", "private pi(r,Z,zbar)"),
    ):
        sub = df[df["selector"] == name]
        for flag, tcr_col, wp_col, dx_mode in (
            ("Myopic", "TCR0", "WP0", "zero"),
            ("Attack", "TCR1", "WP1", "real"),
        ):
            tc_m, tc_s = _mean_std(sub[tcr_col].tolist())
            wp_m, wp_s = _mean_std(sub[wp_col].tolist())
            if dx_mode == "zero":
                dx_txt = "0.000"
            else:
                dx_m, dx_s = _mean_std(sub["dx"].tolist())
                dx_txt = f"{dx_m:.3f}±{dx_s:.3f}"
            out.append(
                {
                    "Selector": f"{flag}, {title}",
                    "TCR_%": f"{tc_m:.1f}±{tc_s:.1f}",
                    "W_P": f"{wp_m:.0f}±{wp_s:.0f}",
                    "Delta_x": dx_txt,
                }
            )
    return pd.DataFrame(out)


def rail_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name in ("aria_public", "gtdim_if", "posted"):
        sub = df[df["selector"] == name]
        wp_m, wp_s = _mean_std(sub["W_P"].tolist())
        tc_m, tc_s = _mean_std(sub["TCR"].tolist())
        hires_m = float(np.mean(sub["hires"]))
        cfg = SimConfig()
        gas_hire = cfg.gas_per_hire
        gas_slot = gas_for_hires(int(round(hires_m / max(cfg.t_slots, 1))), cfg)
        out.append(
            {
                "Stack": LABEL[name] + ", rail off",
                "W_P": f"{wp_m:.0f}±{wp_s:.0f}",
                "TCR_%": f"{tc_m:.1f}±{tc_s:.1f}",
                "Gas/hire": "0",
                "Gas/slot": "0",
            }
        )
        out.append(
            {
                "Stack": LABEL[name] + " + rail",
                "W_P": f"{wp_m:.0f}±{wp_s:.0f}",
                "TCR_%": f"{tc_m:.1f}±{tc_s:.1f}",
                "Gas/hire": f"{gas_hire:.2e}",
                "Gas/slot": f"{gas_hire * (hires_m / max(cfg.t_slots, 1)):.2e}",
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true", help="N=500, T=24, seeds 0-9")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--t", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args()
    if args.paper:
        n, t_slots, n_seeds = 500, 24, 10
    else:
        n = args.n if args.n is not None else 80
        t_slots = args.t if args.t is not None else 6
        n_seeds = args.seeds if args.seeds is not None else 2
    seeds = list(range(n_seeds))
    print(f"CrowdSim bench  N={n}  T={t_slots}  seeds={seeds}")
    df = run_panel(seeds, n, t_slots, attack=False)
    main_tbl = summarize_main(df)
    print("\n=== Main panel ===")
    print(main_tbl.to_string(index=False))
    print("\n=== Attack panel ===")
    print(attack_table(seeds, n, t_slots).to_string(index=False))
    print("\n=== Rail accounting (allocation copied; gas model only) ===")
    print(rail_table(df).to_string(index=False))
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    main_tbl.to_csv(out_dir / "main_panel.csv", index=False)
    print(f"\nWrote {out_dir / 'main_panel.csv'}")


if __name__ == "__main__":
    main()
