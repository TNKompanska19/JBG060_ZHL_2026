"""Run the whole explanation pipeline unattended: build missing inputs, smoke test, full run, tables.

    python flood_pred_modelling/3_model/run_all.py                    (smoke test, then the full run)
    python flood_pred_modelling/3_model/run_all.py --hours 6          (stop starting new stages after 6 h; tables are still written)
    python flood_pred_modelling/3_model/run_all.py --skip-smoke       (resume the full run; finished stages are skipped)
    python flood_pred_modelling/3_model/run_all.py --smoke-only
    python flood_pred_modelling/3_model/run_all.py --skip-smoke --no-build --stages s1     (one chunk; keys below)

Stage keys: s1 s3 s2 = strict h=1/3/2; p1 = strict h=1 with M1-physical inputs; y1 y3 = year-only; f1 f3 = forward;
ebm; ebmp = EBM on M1-physical (needs p1); susc = susceptibility; map.

Order of the full run (most important first, so a cut-off still leaves the core results):
  strict h=1 (ladder, ablation, noise, sensitivity) -> strict h=3 -> strict h=2 (ladder) -> year-only h=1, h=3
  -> forward h=1, h=3 -> EBM -> susceptibility -> harvest map 2022 -> summarise (always).
Outputs: flood_pred_modelling/3_model/output/main/ (results.md, scores.csv, ablation.csv, preds_*.parquet, ebm_*, run_log.txt)
Smoke test: same code on ~120 squares per month and 2 test years, in 3_model/output/smoke/ (wiped each time).
The full run starts only if the smoke test finished without any error.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "analysis")]
import explain_models as em  # noqa: E402
import summarise  # noqa: E402

FULL = dict(per_month=2000, horizons=(1, 2, 3), sample_seed=0, model_seed=0, ebm_rows=60000, susc_n=60000, years=None)
SMOKE = dict(per_month=120, horizons=(1, 2, 3), sample_seed=0, model_seed=0, ebm_rows=3000, susc_n=4000,
             years=[2020, 2022])
NEED = {"terrain": "slope", "soil": "clay", "grace": "grace", "sm": "sm1", "heavy": "r20", "crop": "crop_pct",
        "hydroweb": "hw", "atmos": "t2m", "rainx": "wetd", "hetero": "hnd_sd"}


class Tee:
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
        subprocess.run([sys.executable, str(HERE.parent / "2_build_dataset" / "build_inputs.py"), "--res", str(res), "--add", *steps],
                       check=True, cwd=em.ROOT)
    else:
        em.log("panel has all inputs")


def run(P, cfg, out, hours, only=None):
    out.mkdir(parents=True, exist_ok=True)
    grp, excl = em.blocks(P, out)
    y = cfg["years"]
    stages = [("s1", "strict h=1", lambda: em.run_pass(P, cfg, out, 1, "strict", grp, excl, y)),
              ("p1", "strict h=1, M1-physical", lambda: em.run_pass(P, cfg, out, 1, "strict", grp, excl, y, "physical")),
              ("s3", "strict h=3", lambda: em.run_pass(P, cfg, out, 3, "strict", grp, excl, y)),
              ("s2", "strict h=2", lambda: em.run_pass(P, cfg, out, 2, "strict", grp, excl, y)),
              ("y1", "year-only h=1", lambda: em.run_pass(P, cfg, out, 1, "year", grp, excl, y)),
              ("y3", "year-only h=3", lambda: em.run_pass(P, cfg, out, 3, "year", grp, excl, y)),
              ("f1", "forward h=1", lambda: em.run_pass(P, cfg, out, 1, "forward", grp, excl, y)),
              ("f3", "forward h=3", lambda: em.run_pass(P, cfg, out, 3, "forward", grp, excl, y)),
              ("ebm", "EBM", lambda: em.run_ebm(P, cfg, out, grp, excl)),
              ("ebmp", "EBM on M1-physical", lambda: em.run_ebm(P, cfg, out, grp, excl, "physical")),
              ("susc", "susceptibility", lambda: em.susceptibility(P, cfg, out, grp, excl)),
              ("map", "harvest map 2022", lambda: em.season_map(P, cfg, out, 2022))]
    t0 = time.time()
    for key, name, f in stages:
        if only and key not in only:
            continue
        if hours and (time.time() - t0) / 3600 > hours:
            em.log(f"=== {name}: SKIPPED, time budget of {hours} h used (rerun with --skip-smoke to continue)")
            continue
        em.log(f"=== {name}")
        em.safe(f)
        em.log(f"=== {name} done, {(time.time() - t0) / 60:.0f} min total")
    em.safe(summarise.main, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--hours", type=float, default=0, help="time budget for the full run (0 = none)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--stages", nargs="+", choices=["s1", "p1", "s3", "s2", "y1", "y3", "f1", "f3", "ebm", "ebmp", "susc", "map"],
                    help="full run: only these stages (finished ones are skipped anyway)")
    ap.add_argument("--min-years", type=int, default=None, help="scope: flooded in at least this many years (sensitivity)")
    ap.add_argument("--out", default="main", help="output folder name under flood_pred_modelling/3_model/output")
    a = ap.parse_args()
    out = HERE / "output" / a.out
    if a.min_years:
        em.MIN_FLOOD_YEARS = a.min_years
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")
    em.log(f"run_all {' '.join(sys.argv[1:])}")
    if not a.no_build:
        build_missing(round(a.res, 2))
    P = em.Panel(round(a.res, 2))

    if not a.skip_smoke:
        smoke = HERE / "output" / "smoke"
        shutil.rmtree(smoke, ignore_errors=True)
        em.log("===== SMOKE TEST")
        run(P, SMOKE, smoke, 0)
        if em.ERRORS or not (smoke / "results.md").exists():
            em.log(f"===== SMOKE TEST FAILED ({len(em.ERRORS)} errors, first: {em.ERRORS[:3]}). Full run not started.")
            sys.exit(1)
        em.log("===== SMOKE TEST PASSED")
        if a.smoke_only:
            return
    em.ERRORS.clear()
    em.log("===== FULL RUN")
    run(P, FULL, out, a.hours, a.stages)
    em.log(f"===== FINISHED, {len(em.ERRORS)} errors" + (f": {em.ERRORS[:5]}" if em.ERRORS else ""))


if __name__ == "__main__":
    main()
