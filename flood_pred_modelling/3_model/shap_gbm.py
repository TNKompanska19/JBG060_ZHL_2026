"""TreeSHAP for the M1-physical GBM (strict validation, 1 month ahead): how the evaluated GBM uses each input.

Per test year it refits that fold's GBM exactly as run_pass did, stops if its predictions differ from the stored
ones in preds_strict_h1_physical.parquet (so we explain the models that were scored, not new ones), then computes
exact SHAP values on that fold's held-out rows. SHAP unit: log-odds of flooding; values per row add up exactly to
the model's output (checked). Family SHAP = sum of a family's inputs per row, so correlated inputs inside a family
don't split its credit.

    python flood_pred_modelling/3_model/shap_gbm.py
Outputs in 3_model/output/main/: shap_features.csv, shap_families.csv, shap_families.png,
shap_occurrence.png, shap_persistence.png, shap_values_<target>.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "analysis")]
import explain_models as em  # noqa: E402
from run_all import FULL as CFG  # noqa: E402   same sample and seeds as the main run

OUT = HERE / "output" / "main"
H, CV, COL = 1, "strict", "p|M1_physical|gbm"
KEYS = ["test_year", "target", "issue", "row", "col"]
TARGETS = {"occurrence": "Occurrence (dry -> flooded)", "persistence": "Persistence (flooded -> still flooded)"}
FAMILY_OF = {g: f for f, gs in em.FAMILIES.items() for g in gs}


def collect(P):
    grp, excl = em.blocks(P, OUT)                                   # reads blocks.json of the main run
    s = em.sample(P, CFG["per_month"], max(CFG["horizons"]), CFG["sample_seed"])
    lab = P.y[s[:, 0] + H, s[:, 1], s[:, 2]] >= 0.5
    state = P.y[s[:, 0] - 1, s[:, 1], s[:, 2]] >= 0.5
    stored = pd.read_parquet(OUT / "preds_strict_h1_physical.parquet", columns=KEYS + ["cv", "h", COL])
    stored = stored[(stored.cv == CV) & (stored.h == H)].drop(columns=["cv", "h"])
    acc, feat = {t: [] for t in TARGETS}, None
    for Y in em.TEST_YEARS:
        X, names = em.features(P, s, H, em.normals(P, Y, sd=True), physical=True)
        groups, cols = em.physical_sets(names, P)
        fn = [names[i] for i in cols]
        if feat is None:
            feat = (fn, {names[i]: g for g, gc in groups.items() for i in gc})
            missing = sorted({g for g in groups if g not in FAMILY_OF})
            if missing:
                raise SystemExit(f"groups without a family: {missing}")
        elif fn != feat[0]:
            raise SystemExit(f"input list differs in fold {Y}")
        train, test, _ = em.fold_masks(P, s, H, Y, CV, grp, excl)
        for target, sel in (("occurrence", ~state), ("persistence", state)):
            tr, te = np.flatnonzero(train & sel), np.flatnonzero(test & sel)
            if te.size < 20 or tr.size < 200:                      # same rule as run_pass
                continue
            m = HistGradientBoostingClassifier(**{**em.GBM_BASE, "random_state": CFG["model_seed"]})
            m.fit(X[np.ix_(tr, cols)], lab[tr])
            Xte = X[np.ix_(te, cols)]
            p = m.predict_proba(Xte)[:, 1]
            key = pd.DataFrame({"test_year": Y, "target": target, "issue": P.months[s[te, 0]].astype(str),
                                "row": s[te, 1], "col": s[te, 2]})
            ref = key.merge(stored, on=KEYS, how="left")[COL].to_numpy()
            diff = np.nanmax(np.abs(p - ref)) if np.isfinite(ref).all() else np.inf
            if diff > 1e-4:
                raise SystemExit(f"{Y} {target}: refit differs from the stored GBM (max diff {diff:.2g}); stopped")
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(Xte)
            sv = sv[-1] if isinstance(sv, list) else sv
            add = np.abs(sv.sum(1) + np.ravel(ex.expected_value)[-1] - m.decision_function(Xte)).max()
            if add > 1e-4:
                raise SystemExit(f"{Y} {target}: SHAP values don't add up to the model output ({add:.2g}); stopped")
            acc[target].append((Xte, sv.astype(np.float32)))
            em.log(f"{Y} {target}: {te.size:,} held-out rows, refit matches stored (max diff {diff:.1e}), "
                   f"SHAP additivity error {add:.1e}")
    return acc, feat


def tables(acc, feat):
    fn, group = feat
    frows, famrows = [], []
    for target, parts in acc.items():
        X = np.concatenate([a for a, _ in parts])
        S = np.concatenate([b for _, b in parts])
        np.savez_compressed(OUT / f"shap_values_{target}.npz", X=X, shap=S, names=np.array(fn))
        for j, n in enumerate(fn):
            ok = np.isfinite(X[:, j])
            rho = spearmanr(X[ok, j], S[ok, j])[0] if ok.sum() > 10 else np.nan
            frows.append(dict(target=target, feature=n, label=em.pretty(n), group=group[n],
                              family=FAMILY_OF[group[n]], mean_abs_shap=np.abs(S[:, j]).mean(),
                              mean_shap=S[:, j].mean(), direction_spearman=rho, rows=len(X)))
        for fam in em.FAMILIES:
            idx = [j for j, n in enumerate(fn) if FAMILY_OF[group[n]] == fam]
            if idx:
                famrows.append(dict(target=target, family=fam, inputs=len(idx),
                                    mean_abs_shap=np.abs(S[:, idx].sum(1)).mean(), rows=len(X)))
    f = pd.DataFrame(frows).sort_values(["target", "mean_abs_shap"], ascending=[True, False])
    fam = pd.DataFrame(famrows).sort_values(["target", "mean_abs_shap"], ascending=[True, False])
    f.to_csv(OUT / "shap_features.csv", index=False)
    fam.to_csv(OUT / "shap_families.csv", index=False)
    return f, fam


def plots(acc, feat, f, fam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fn = feat[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for a, (target, title) in zip(ax, TARGETS.items()):
        d = fam[fam.target == target].iloc[::-1]
        a.barh(d.family.str.replace("_", " "), d.mean_abs_shap)
        a.set_title(title)
        a.set_xlabel("Mean absolute effect on log-odds of flooding (SHAP)")
    ax[0].set_ylabel("Input family")
    fig.suptitle("GBM (M1-physical), strict validation, 1 month ahead: held-out rows")
    fig.tight_layout()
    fig.savefig(OUT / "shap_families.png", dpi=150)
    plt.close(fig)
    for target, title in TARGETS.items():
        X = np.concatenate([a for a, _ in acc[target]])
        S = np.concatenate([b for _, b in acc[target]])
        top = f[f.target == target].feature.head(8).tolist()
        fig, ax = plt.subplots(2, 4, figsize=(16, 7))
        for a, n in zip(ax.ravel(), top):
            j = fn.index(n)
            ok = np.isfinite(X[:, j])
            x, y = X[ok, j], S[ok, j]
            a.scatter(x, y, s=2, alpha=0.15)
            q = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
            if q.size > 2:
                b = np.clip(np.digitize(x, q[1:-1]), 0, q.size - 2)
                mid = [x[b == k].mean() for k in range(q.size - 1) if (b == k).any()]
                avg = [y[b == k].mean() for k in range(q.size - 1) if (b == k).any()]
                a.plot(mid, avg, color="k", lw=2, label="mean per decile")
            a.axhline(0, color="grey", lw=0.8)
            a.set_xlabel(em.pretty(n), fontsize=8)
            a.set_ylabel("Effect on log-odds of flooding (SHAP)", fontsize=8)
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"{title}: GBM (M1-physical), 8 inputs with the largest mean |SHAP|, held-out rows "
                     f"(dots = rows, 0 = no effect)")
        fig.tight_layout()
        fig.savefig(OUT / f"shap_{target}.png", dpi=150)
        plt.close(fig)


def main():
    P = em.Panel(0.02)
    acc, feat = collect(P)
    f, fam = tables(acc, feat)
    plots(acc, feat, f, fam)
    pd.set_option("display.width", 200)
    print(fam.round(3).to_string(index=False))
    for t in TARGETS:
        print(f"\n{t}: top 10 inputs")
        print(f[f.target == t].head(10)[["label", "family", "mean_abs_shap", "direction_spearman"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
