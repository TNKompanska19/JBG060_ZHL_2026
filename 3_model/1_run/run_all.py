"""Run the explanation pipeline: build missing inputs, all stages, tables. Finished stages are skipped.

    python 3_model/1_run/run_all.py
    python 3_model/1_run/run_all.py --no-build --stages s1 as1     (only these stages; keys below)

Stage keys: s/y/f + 1/2/3 = raw-input models, strict/year-only/forward validation, 1/2/3 months ahead (s1 y2 f3 ...);
an extra a in front (as1 ay2 af3 ...) = anomaly-input models; susc = susceptibility; map = harvest map 2022.
Outputs: results/run_all/ (results.md, scores.csv, ablation.csv, preds_*.parquet, run_log.txt)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "2_analysis")]
import explain_models as em  # noqa: E402
import summarise  # noqa: E402

FULL = dict(per_month=2000, horizons=(1, 2, 3), sample_seed=0, model_seed=0, susc_n=60000)
NEED = {"terrain": "slope", "soil": "clay", "grace": "grace", "sm": "sm1", "heavy": "r20", "crop": "crop_pct",
        "hydroweb": "hw", "atmos": "t2m", "rainx": "wetd", "hetero": "hnd_sd"}


# every validation scheme x lead month: raw-input ladder (s1 y1 f1 ...) and anomaly-input models (as1 ay1 af1 ...)
PASSES = {}
for cv in ("strict", "year", "forward"):
    for h in (1, 2, 3):
        PASSES[f"{cv[0]}{h}"] = (cv, h, "")
        PASSES[f"a{cv[0]}{h}"] = (cv, h, "anomaly")


class Tee:
    """Print to the screen and to run_log.txt."""
    def __init__(self, path):
        self.f, self.out = open(path, "a", encoding="utf8"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def build_missing(res):
    p = em.ROOT / "processed_data" / f"panel_{res:.2f}"
    have = set(np.load(p / "inputs.npz").files)
    cols = set(pd.read_csv(p / "shared.csv", nrows=0).columns)
    steps = [s for s, k in NEED.items() if k not in have] + ([] if "nino34" in cols else ["indices"])
    if steps:
        em.log(f"building missing inputs: {steps}")
        subprocess.run([sys.executable, str(em.ROOT / "1_data_processing" / "3_build_dataset" / "build_inputs.py"), "--res", str(res),
                        "--add", *steps], check=True, cwd=em.ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--stages", nargs="+", choices=[*PASSES, "susc", "map"])
    ap.add_argument("--min-years", type=int, default=None, help="scope: flooded in at least this many years (sensitivity)")
    ap.add_argument("--out", default="run_all", help="output folder name under results/")
    a = ap.parse_args()
    out = em.ROOT / "results" / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.min_years:
        em.MIN_FLOOD_YEARS = a.min_years
    sys.stdout = Tee(out / "run_log.txt")
    if not a.no_build:
        build_missing(round(a.res, 2))
    P = em.Panel(round(a.res, 2))
    grp, excl = em.blocks(P, out)
    stages = {k: (lambda cv=cv, h=h, v=v: em.run_pass(P, FULL, out, h, cv, grp, excl, v)) for k, (cv, h, v) in PASSES.items()}
    stages.update({"susc": lambda: em.susceptibility(P, FULL, out, grp, excl),
                   "map": lambda: em.season_map(P, FULL, out, 2022)})
    for key, run in stages.items():
        if not a.stages or key in a.stages:
            em.log(f"=== {key}")
            run()
    summarise.main(out)


if __name__ == "__main__":
    main()
