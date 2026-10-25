"""Appendix A table (all 79 counties) and Figure 1 (coverage thinness vs skill) for the SLE essay.

    python 3_model/2_analysis/county_appendix.py      (from the repo root; run county_performance.py first)

Skill  = PR-AUC gain of p|M3_anomaly|gbm over the climatology baseline, occurrence, 1 month ahead (county_performance.md).
Coverage thinness = mean percentile rank of three indicators (1 = thinnest): distance from county centroid to
Dartmouth station 100205 (9.6N, 31.6E), share of the county's squares outside the flood-prone scope, and
public health facilities per 1000 km2 (inverted).
Red error bars (Renk, Melut, Maban): 95% CI of skill, block bootstrap over issue months (skill_ci.md).
Writes results/county_appendix/{appendix_a.tex, appendix_a.md, skill_ci.md, figure1_coverage_skill.png/.pdf}.
"""
import json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, shape
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_data"
PERF = ROOT / "results" / "county_performance" / "county_performance.md"
ADMIN = RAW / "Administrative boundaries" / "ssd_admin2.geojson"
FAC = RAW / "health facilities" / "Sub-Saharan_public_health_facilities.geojson"
IPC = RAW / "IPC" / "IPC_SS_Acute_Food_Insecurity_September_2025_2026-06-11.xlsx"
OUT = ROOT / "results" / "county_appendix"
PRED = ROOT / "results" / "run_all" / "preds_strict_h1_anomaly.parquet"
PANEL = ROOT / "processed_data" / "panel_0.02"
MODEL = "p|M3_anomaly|gbm"             # same model as county_performance.py
N_BOOT = 2000
GAUGE = (9.6, 31.6)                      # Dartmouth station 100205 (lat, lon)
MARK = ["Renk", "Melut", "Maban"]


def md_tables(path):
    """Parse the '## ' sections of county_performance.md into DataFrames."""
    out, cur = {}, None
    for ln in path.read_text(encoding="utf8").splitlines():
        if ln.startswith("## "):
            cur = ln[3:]; out[cur] = []
        elif ln.startswith("|") and cur:
            out[cur].append([c.strip() for c in ln.strip("|").split("|")])
    res = {}
    for k, r in out.items():
        d = pd.DataFrame(r[2:], columns=r[0])
        for c in d.columns.drop(["state", "county"], errors="ignore"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        res[k] = d
    return res


def km(lat, lon, lat0, lon0):
    r = np.pi / 180
    a = np.sin((lat - lat0) * r / 2) ** 2 + np.cos(lat * r) * np.cos(lat0 * r) * np.sin((lon - lon0) * r / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def build():
    T = md_tables(PERF)
    occ = T["Skill per county, target occurrence"][["county", "pr_auc_gain_vs_clim"]]
    cov = T["Target coverage per county (test period): share of squares in scope, target missing %"][["county", "in_scope_pct"]]

    feats = json.loads(ADMIN.read_text(encoding="utf8"))["features"]
    C = pd.DataFrame([dict(county=f["properties"]["adm2_name"], state=f["properties"]["adm1_name"],
                           area=f["properties"]["area_sqkm"], geom=shape(f["geometry"])) for f in feats])
    C["dist_gauge_km"] = km(C.geom.map(lambda s: s.centroid.y), C.geom.map(lambda s: s.centroid.x), *GAUGE)

    ss = [f for f in json.loads(FAC.read_text(encoding="utf8"))["features"] if f["properties"]["Country"] == "South Sudan"]
    pts = [Point(f["geometry"]["coordinates"][:2]) if f["geometry"] else Point(f["properties"]["Long"], f["properties"]["Lat"])
           for f in ss if f["geometry"] or (f["properties"].get("Lat") is not None and f["properties"].get("Long") is not None)]
    C["fac_per_1000km2"] = [sum(g.contains(p) for p in pts) / a * 1000 for g, a in zip(C.geom, C.area)]

    # IPC Phase 3+ share, Sept 2025; names normalised (Wau matches its "Rural Only" row; Abyei has no row)
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r" county$", "", str(s).strip().lower().replace("/", "-")))
    x = pd.read_excel(IPC, sheet_name="Population Table")
    x = pd.DataFrame({"key": x["Area Name"].map(norm).str.replace(r" \(rural only\)", "", regex=True),
                      "ipc3p": x["Current - Phase 3+ %"]})
    C["key"] = C.county.map(norm)
    C = C.merge(x, on="key", how="left").merge(cov, on="county", how="left").merge(occ, on="county", how="left")

    C["thin"] = pd.concat([C.dist_gauge_km, -C.in_scope_pct, -C.fac_per_1000km2], axis=1).rank(pct=True).mean(axis=1)
    C["scored"] = C.pr_auc_gain_vs_clim.notna()
    return C.drop(columns=["geom", "key", "area"])


def table(C):
    s = C[C.scored].sort_values("pr_auc_gain_vs_clim", ascending=False)
    u = C[~C.scored].sort_values(["state", "county"])
    rows = [[r.county, r.state, f"{r.thin:.2f}", f"{r.pr_auc_gain_vs_clim:.3f}" if r.scored else "Not assessed",
             "No IPC data" if pd.isna(r.ipc3p) else f"{r.ipc3p * 100:.0f}%", "Scored" if r.scored else "Not assessed"]
            for _, r in pd.concat([s, u]).iterrows()]
    head = ["County", "State", "Coverage thinness", "Skill", "IPC Phase 3+ share", "Status"]
    note = ("Skill: gain in PR-AUC of the model over the climatology baseline, flood occurrence one month ahead "
            "(above 0 = better than climatology). Coverage thinness: mean percentile rank of distance to the river "
            "gauge, share of squares outside the flood-prone scope and health facilities per 1000 km2 "
            "(1 = thinnest). Counties with fewer than 20 flood or fewer than 20 non-flood square-months are not assessed.")
    md = "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])
    (OUT / "appendix_a.md").write_text(md + "\n\n" + note + "\n", encoding="utf8")
    esc = lambda t: t.replace("%", r"\%").replace("&", r"\&")
    tex = [r"\begin{longtable}{llrrrl}",
           r"\caption{County results, all 79 counties: scored counties by skill, then counties not assessed.}\label{tab:appA}\\",
           r"\hline " + " & ".join(head) + r" \\ \hline", r"\endfirsthead",
           r"\hline " + " & ".join(head) + r" \\ \hline", r"\endhead",
           r"\hline \multicolumn{6}{p{0.95\linewidth}}{\footnotesize " + esc(note) + r"} \\", r"\endlastfoot"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\end{longtable}"]
    (OUT / "appendix_a.tex").write_text("\n".join(tex) + "\n", encoding="utf8")
    return len(s), len(u)


def skill_ci(counties):
    """95% CI of skill (PR-AUC gain over climatology, occurrence, h=1) per county.
    Block bootstrap over issue months: all of a county's rows from one month are drawn together, so floods that
    co-occur in space within a month are not treated as independent. Model and baseline use the same resample."""
    d = pd.read_parquet(PRED)
    d = d[d.target == "occurrence"]
    z = np.load(PANEL / "inputs.npz")
    lat, lon = z["lat"], z["lon"]
    res = round(float(lon[1] - lon[0]), 6)
    feats = json.loads(ADMIN.read_text(encoding="utf8"))["features"]
    names = [f["properties"]["adm2_name"] for f in feats]
    cid = rasterize([(f["geometry"], i + 1) for i, f in enumerate(feats)], out_shape=(lat.size, lon.size),
                    transform=from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res), fill=0, dtype="int32")
    k = cid[d.row.to_numpy(), d.col.to_numpy()]
    d["county"] = [names[i - 1] if i else "outside" for i in k]
    gain = lambda g: average_precision_score(g.y, g[MODEL]) - average_precision_score(g.y, g.p_clim)
    rng, out = np.random.default_rng(0), {}
    for c in counties:
        g = d[d.county == c].reset_index(drop=True)
        blocks = list(g.groupby("issue").indices.values())
        bs = []
        for _ in range(N_BOOT):
            r = g.iloc[np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])]
            if 0 < r.y.sum() < len(r):
                bs.append(gain(r))
        bs = np.array(bs)
        out[c] = dict(skill=gain(g), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                      share_below_0=(bs < 0).mean(), months=len(blocks), floods=int(g.y.sum()))
    t = pd.DataFrame(out).T.rename_axis("county").reset_index()
    rows = [[r.county, f"{r.skill:.3f}", f"[{r.lo:.3f}, {r.hi:.3f}]", f"{r.share_below_0:.0%}", str(int(r.months)), str(int(r.floods))]
            for r in t.itertuples()]
    head = ["County", "Skill", "95% CI", "Resamples below 0", "Issue months", "Floods"]
    (OUT / "skill_ci.md").write_text("\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
                                               + ["| " + " | ".join(r) + " |" for r in rows])
                                     + f"\n\nBlock bootstrap over issue months, {N_BOOT} resamples, seed 0.\n", encoding="utf8")
    return t.set_index("county")


def figure(C, ci):
    """Single-column figure: thinness (x) vs skill (y), coloured by IPC Phase 3+ share; strip for counties not assessed."""
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 7})
    s, u = C[C.scored], C[~C.scored]
    rho = spearmanr(s.thin, s.pr_auc_gain_vs_clim)[0]
    rng = np.random.default_rng(0)
    bs = [spearmanr(*s[["thin", "pr_auc_gain_vs_clim"]].to_numpy()[rng.integers(0, len(s), len(s))].T)[0] for _ in range(2000)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    cmap, norm = plt.get_cmap("viridis_r"), plt.Normalize(0, 0.7)
    col = lambda v: [cmap(norm(x)) if pd.notna(x) else "white" for x in v]

    fig, (ax, st) = plt.subplots(2, 1, figsize=(3.4, 3.4), sharex=True, gridspec_kw={"height_ratios": [5, 1.1], "hspace": 0.3})
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.scatter(s.thin, s.pr_auc_gain_vs_clim, s=14, c=col(s.ipc3p), edgecolors="black", linewidths=0.3)
    m = s[s.county.isin(MARK)]
    e = ci.loc[m.county]
    ax.errorbar(m.thin, m.pr_auc_gain_vs_clim, yerr=[m.pr_auc_gain_vs_clim.to_numpy() - e.lo.to_numpy(),
                e.hi.to_numpy() - m.pr_auc_gain_vs_clim.to_numpy()], fmt="none", ecolor="red", elinewidth=0.8, capsize=2, zorder=1)
    ax.scatter(m.thin, m.pr_auc_gain_vs_clim, s=40, facecolors="none", edgecolors="red", linewidths=0.9)
    off = {"Renk": (6, -3), "Melut": (-6, -3), "Maban": (-8, 1)}
    for _, r in m.iterrows():
        ax.annotate(r.county, (r.thin, r.pr_auc_gain_vs_clim), xytext=off[r.county], textcoords="offset points",
                    ha="right" if off[r.county][0] < 0 else "left", color="red")
    ax.text(0.98, 0.97, f"Spearman rho = {rho:.2f}\n95% CI [{lo:.2f}, {hi:.2f}], n = {len(s)}",
            transform=ax.transAxes, ha="right", va="top")
    ax.text(0.01, 0.004, "baseline", transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="left", va="bottom", color="grey", fontsize=6)
    ax.set_ylabel("Skill (gain over climatology,\n1 month ahead)")
    ax.set_xlim(0, 1)

    st.scatter(u.thin, np.zeros(len(u)), s=14, c=col(u.ipc3p), edgecolors="black", linewidths=0.3)
    st.set_yticks([]); st.set_ylim(-1, 1)
    st.set_title(f"Not assessed ({len(u)} counties, no skill)", loc="left", fontsize=6.5)
    st.set_xlabel("Coverage thinness (0 = best covered, 1 = thinnest)")

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax, st], fraction=0.05, pad=0.03)
    cb.set_label("IPC Phase 3+ share (white = no IPC data)")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    fig.savefig(OUT / "figure1_coverage_skill.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figure1_coverage_skill.pdf", bbox_inches="tight")
    return rho, lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = build()
    n_s, n_u = table(C)
    ci = skill_ci(MARK)
    rho, lo, hi = figure(C, ci)
    print(ci.round(3).to_string())
    u = C[~C.scored]
    print(f"scored {n_s}, not assessed {n_u} ({u.state.str.contains('Equatoria').sum()} in the Equatoria states)")
    print(f"Spearman thinness vs skill: {rho:.3f} [{lo:.2f}, {hi:.2f}]")
    print(C[C.county.isin(MARK + ['Rubkona'])][["county", "thin", "pr_auc_gain_vs_clim", "ipc3p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
