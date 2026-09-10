# ARIA: Public Instrument Fields for Anticipation-Robust Incentive Design in Mobile Crowdsensing

Public instrument fields for anticipation-robust incentive design in mobile crowdsensing.

Authors: Seyed Ali Hosseini, Omid Sojodishijani (corresponding), Vahid Khajehvand  

## Layout

```
ARIA_complete/
  paper/ARIA.tex          # camera-ready-style manuscript
  paper/aria_refs.bib
  aria/                   # library (one class / module)
    config.py             # SimConfig
    world.py              # CrowdWorld
    field.py              # X, pi(r,Z), private ablation
    scores.py             # closed-form base scores
    settlement.py         # Phase A cover + critical drop
    selectors.py          # ARIA, GTDIM-IF, locked vertices, posted
    engine.py             # paired-seed simulator
    metrics.py
    rail.py               # Groth16 gas accounting only
  sim/crowdsim_bench.py
  notebooks/ARIA_CrowdSim_Colab.ipynb
  requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python sim/crowdsim_bench.py              # smoke: N=80, T=6, 2 seeds
python sim/crowdsim_bench.py --paper      # manuscript scale (slow)
```

This artifact implements the **closed-form** protocol in the paper.
It does not train an SSM, does not replay GeoLife/T-Drive/Roma, and
does not send a chain transaction. Numbers will not match the manuscript
tables bit-for-bit; those tables were emitted by the authors' original
RNG stream. The code is a faithful public reconstruction of the interface.

## Colab

Upload `notebooks/ARIA_CrowdSim_Colab.ipynb` or open it after copying
the `aria/` package into the notebook runtime.
