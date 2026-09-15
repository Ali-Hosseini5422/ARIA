# ARIA: A Public Instrument Field for Incentive Design under Strategic Anticipation in Mobile CrowdSensing

Closed-form CrowdSim artifact for the manuscript.

Authors: Seyed Ali Hosseini, Omid Sojodishijani (corresponding), Vahid Khajehvand

Repository: https://github.com/Ali-Hosseini5422/ARIA

## Layout

```
ARIA_complete/
  paper/ARIA.tex
  paper/aria_refs.bib
  aria/                    # library
    config.py              # SimConfig (manuscript table)
    world.py               # paired CrowdWorld
    field.py               # X, pi(r,Z), private ablation
    scores.py              # closed-form base scores
    settlement.py          # Phase A cover + critical drop
    selectors.py           # ARIA, GTDIM-IF, locked vertices, posted
    engine.py              # paired-seed simulator
    metrics.py
    rail.py                # Groth16 gas accounting only
  sim/crowdsim_bench.py    # manuscript entry point
  notebooks/ARIA_CrowdSim_Colab.ipynb
  requirements.txt
```

## Protocol notes (aligned with the manuscript)

- `pi(r,Z)` reads only the public regime and the virtual queue. It does not read encodings or reports.
- `r_t` uses lagged abandonment and lagged realized quality-uncertainty from slot `t-1`. Current-slot encodings do not enter `r_t`.
- Locked vertices reuse ARIA scores and critical values **without** the Phase A cover prefix.
- Field-only is `pi(r,Z)` with `eta = 0`. Cover-only spends the slot budget on Phase A and does not call `pi`.
- Public ARIA = Phase A prefix + field-weighted critical settlement.
- Attacker bits are always drawn from the seed. Extreme encodings are applied only when `--` attack is on inside the bench attack panel.
- No SSM is trained. No GeoLife / T-Drive / Roma replay. No chain submission.

## Run locally

```bash
pip install -r requirements.txt
python sim/crowdsim_bench.py              # smoke: N=80, T=6, 2 seeds
python sim/crowdsim_bench.py --paper      # manuscript scale
```

From the repo root, `sim/crowdsim_bench.py` puts the project root on `sys.path` and imports the `aria` package.

## Reproducibility

This artifact implements the closed-form protocol in the paper. A paper-scale run emits tables with the same columns as the manuscript. Absolute `W_P` / TCR values can differ from the printed means because those printed cells were produced by the authors' original RNG stream. The identification (cover vs field vs public `pi`, and `Delta x = 0` on public attack) is a property of the interface and should hold on this code.

## Colab

Open `notebooks/ARIA_CrowdSim_Colab.ipynb` after copying this repository into the runtime.
