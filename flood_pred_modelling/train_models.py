"""Train and test flood models on the panel from build_inputs.py.

    python flood_pred_modelling/train_models.py --res 0.02 --model gbm
    python flood_pred_modelling/train_models.py --res 0.02 --model nn --horizons 1   (pip install torch)

One sample = one square x one issue month t (we stand at the end of month t).
TARGET: is the square water at month t+h (h = 1, 2, 3)? At 0.01 deg this is the
CYGNSS label itself; for coarser squares, 'share of its 1 km cells labelled water
>= 0.5'.
INPUTS, all available at the end of month t:
  history (last K months up to t): the square's rainfall, runoff and ET; three lake
      levels; Dartmouth stations; rainfall over three upstream areas
  flood state: the square at t-1, t-2, t-3 and its ~5 km neighbourhood at t-1
      (a CYGNSS map needs the following month, so month t is not yet available)
  the square's normal: how often it was water in the target's calendar month in
      TRAINING years only; latitude; longitude; target month (sin, cos)
TESTING: leave one year out (by target month). Training drops every sample whose
target or input window touches the test year. --cv region also holds out one of
four regions per fold, so neither that year nor that place was seen in training.
BASELINES: the square's normal (climatology) and its latest map (persistence).
SCORES per group, by how often the square is water over the whole record:
always (>= --permanent, one table per value given), seasonal, rarely (< --rare).
Groups are for reporting only, never inputs.
Output: flood_pred_modelling/output/models_<res>/<model>_scores.csv, _preds.parquet
"""
import argparse
import copy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
TEST_YEARS = range(2019, 2026)
IDX = {"tp": "era5_idx", "ro": "era5_idx", "et": "et_idx", "q": "q_idx"}
STATIC = ("q_dist", "q_size", "hnd_mean", "hnd_min", "low_2m", "upa_max")   # fixed per square, if built
NN_SEEDS = 1
# Input groups for --drop (ablation): train without a group to measure what it contributes.
GROUPS = {
    "glofas_nearest": lambda n: n in ("q", "q_dist", "q_size"),
    "glofas_weighted": lambda n: n.startswith("qk"),
    "glofas_points": lambda n: n.startswith("q_") and n not in ("q_dist", "q_size"),
    "merit": lambda n: n in ("hnd_mean", "hnd_min", "low_2m", "upa_max"),
    "rain": lambda n: n in ("tp", "ro", "et") or n.startswith("rain_"),
    "lakes": lambda n: n.startswith("lake_"),
    "dfo": lambda n: n.startswith("dfo_"),
}
GROUPS["glofas"] = lambda n: any(GROUPS[g](n) for g in ("glofas_nearest", "glofas_weighted", "glofas_points"))
warnings.filterwarnings("ignore", message="Mean of empty slice")   # squares outside the country


# ------------------------------------------------------------------ samples

def load(res, drop=()):
    p = ROOT / "processed_data" / f"panel_{res:.2f}"
    z = np.load(p / "inputs.npz")
    d = {k: z[k] for k in z.files}
    d["y"] = d["y"].astype(np.float32)
    d["nbr"] = d["nbr"].astype(np.float32)
    shared = pd.read_csv(p / "shared.csv", index_col=0)
    d["shared"] = shared.to_numpy(np.float32)
    d["shared_names"] = list(shared.columns)
    d["months"] = pd.PeriodIndex(d["months"], freq="M")
    d["local"] = [k for k in IDX if k in d]
    d["cubes"] = [k for k in d if k.startswith("qk")]       # weighted river flow, per square
    for k in d["cubes"]:
        d[k] = d[k].astype(np.float32)

    gone = lambda n: any(GROUPS[g](n) for g in drop)                 # noqa: E731
    dropped = [n for n in d["local"] + d["cubes"] + d["shared_names"] + list(STATIC)
               if gone(n) and (n in d or n in d["shared_names"])]
    d["local"] = [n for n in d["local"] if not gone(n)]            # arrays stay: normals still need them
    d["cubes"] = [n for n in d["cubes"] if not gone(n)]
    keep = [i for i, n in enumerate(d["shared_names"]) if not gone(n)]
    d["shared"], d["shared_names"] = d["shared"][:, keep], [d["shared_names"][i] for i in keep]
    for k in STATIC:
        if gone(k):
            d.pop(k, None)
    if drop:
        print(f"dropped ({', '.join(drop)}): {dropped}")
    return d


def sample(d, k, h_max, per_month, rng):
    """Random (issue month, row, col) with a valid target at t+1..t+h_max and maps t-1..t-3."""
    y, T = d["y"], len(d["months"])
    era_ok = ~np.isnan(d["tp"]).all(1)
    rows = []
    for t in range(max(k - 1, 3), T - h_max):
        if not era_ok[t]:
            continue
        need = np.stack([y[t + h] for h in range(1, h_max + 1)] + [y[t - 1], y[t - 2], y[t - 3]])
        ok = np.flatnonzero(~np.isnan(need).any(0).ravel())
        if ok.size == 0:
            continue
        pick = rng.choice(ok, size=min(per_month, ok.size), replace=False)
        rows.append(np.column_stack([np.full(pick.size, t), *np.unravel_index(pick, y.shape[1:])]))
    s = np.concatenate(rows)
    print(f"{len(s):,} samples, issue months {d['months'][s[:, 0].min()]} to {d['months'][s[:, 0].max()]}")
    return s


def features(d, s, k):
    """Sequence block [n, k, F] and static block [n, 6] (normal added per fold)."""
    t, r, c = s[:, 0], s[:, 1], s[:, 2]
    tt = t[:, None] + np.arange(-k + 1, 1)[None, :]                     # months t-k+1 .. t
    seq = [d[v][tt, d[IDX[v]][r, c][:, None]] for v in d["local"]]
    seq += [d[v][tt, r[:, None], c[:, None]] for v in d["cubes"]]
    seq = np.concatenate([np.stack(seq, -1), d["shared"][tt]], -1)
    stat = np.column_stack([d["y"][t - 1, r, c], d["y"][t - 2, r, c], d["y"][t - 3, r, c],
                            d["nbr"][t - 1, r, c], d["lat"][r], d["lon"][c],
                            *[d[k][r, c] for k in STATIC if k in d]])
    names_seq = d["local"] + d["cubes"] + d["shared_names"]
    return seq.astype(np.float32), stat.astype(np.float32), names_seq


def normal(d, year):
    """[12, row, col]: share of training-year months (all years except `year`) the square was water."""
    m = d["months"]
    out = np.full((12,) + d["y"].shape[1:], np.nan, np.float32)
    for cm in range(1, 13):
        idx = np.flatnonzero((m.month == cm) & (m.year != year))
        out[cm - 1] = np.nanmean(d["y"][idx], 0)
    return out


def discharge_normals(d, year):
    """Per calendar month, the mean of every discharge series in training years only."""
    m = d["months"]
    out = {"q": np.full((12, d["q"].shape[1]), np.nan, np.float32),
           "shared": np.full((12, d["shared"].shape[1]), np.nan, np.float32)}
    out.update({v: np.full((12,) + d[v].shape[1:], np.nan, np.float32) for v in d["cubes"]})
    for cm in range(1, 13):
        idx = np.flatnonzero((m.month == cm) & (m.year != year))
        out["q"][cm - 1] = np.nanmean(d["q"][idx], 0)
        out["shared"][cm - 1] = np.nanmean(d["shared"][idx], 0)
        for v in d["cubes"]:
            out[v][cm - 1] = np.nanmean(d[v][idx], 0)
    return out


def to_anomaly(seq, d, names, norm, cal_tt, qcell, s):
    """Discharge inputs become 'flow minus that river's normal for that calendar month'.
    Raw flow mostly repeats the season, which the model already has; the anomaly is
    what should signal unusual flooding."""
    seq = seq.copy()
    n_front = len(d["local"]) + len(d["cubes"])
    r, c = s[:, 1][:, None], s[:, 2][:, None]
    for j, name in enumerate(names):
        if name == "q":
            seq[:, :, j] -= norm["q"][cal_tt, qcell[:, None]]
        elif name in d["cubes"]:
            seq[:, :, j] -= norm[name][cal_tt, r, c]
        elif name.startswith("q_"):
            seq[:, :, j] -= norm["shared"][cal_tt, j - n_front]
    return seq


def accumulated(seq_f, names, windows=(3, 6, 12)):
    """Summed discharge anomaly over the last 3, 6 and 12 months, per river series.
    The Sudd behaves like a storage basin: it fills from months of excess inflow, so
    the running total matters more than any single month."""
    k = seq_f.shape[1]
    return [np.nansum(seq_f[:, -min(w, k):, j], 1)
            for j, n in enumerate(names) if n == "q" or n.startswith(("q_", "qk")) for w in windows]


# ------------------------------------------------------------------ models

def flat(seq, stat):
    return np.concatenate([seq.reshape(len(seq), -1), stat], 1)


def fit_predict(model, tr, te, seed, val=None):
    (sq_tr, st_tr, y_tr), (sq_te, st_te) = tr, te
    if model == "gbm":
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
        m.fit(flat(sq_tr, st_tr), y_tr)
        return m.predict_proba(flat(sq_te, st_te))[:, 1], None, None
    # logreg and nn: fill gaps with training means, standardise with training spread
    X_tr, X_te = flat(sq_tr, st_tr), flat(sq_te, st_te)
    mu, sd = np.nanmean(X_tr, 0), np.nanstd(X_tr, 0) + 1e-6
    z = lambda X: np.nan_to_num((X - mu) / sd)                               # noqa: E731
    if model == "logreg":
        m = LogisticRegression(max_iter=2000).fit(z(X_tr), y_tr)
        return m.predict_proba(z(X_te))[:, 1], None, None
    k, f = sq_tr.shape[1:]
    split = lambda X: (X[:, :k * f].reshape(-1, k, f), X[:, k * f:])       # noqa: E731
    outs = [nn_fit_predict(split(z(X_tr)), y_tr, split(z(X_te)), seed + i, val) for i in range(NN_SEEDS)]
    return tuple(np.mean([o[j] for o in outs], 0) for j in range(3))     # average over seeds


def nn_fit_predict(tr, y_tr, te, seed, val, epochs=40, batch=2048, patience=4):
    """Temporal-Fusion-style network (after Lim et al. 2021, simplified).

    1. every input gets its own small embedding, so each variable can act nonlinearly;
    2. a variable-selection block, conditioned on the square's fixed inputs, weighs the
       inputs in every month (these softmax weights are the explanation output);
    3. a GRU reads the history, starting from a state built from the fixed inputs;
    4. attention weighs the months; gated residual blocks (ELU, GLU gate, skip
       connection, LayerNorm) combine everything.
    Dropout, weight decay, learning rate halved when the held-out-year loss stalls,
    early stopping on that held-out training year.
    """
    import torch
    import torch.nn.functional as F
    from torch import nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class GRN(nn.Module):
        """Gated residual block."""
        def __init__(self, d_in, d, d_ctx=0, drop=0.1):
            super().__init__()
            self.fc1 = nn.Linear(d_in + d_ctx, d)
            self.fc2 = nn.Linear(d, 2 * d)
            self.skip = nn.Linear(d_in, d) if d_in != d else nn.Identity()
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(drop)

        def forward(self, x, ctx=None):
            h = x if ctx is None else torch.cat([x, ctx], -1)
            h = F.glu(self.drop(self.fc2(F.elu(self.fc1(h)))), -1)
            return self.norm(self.skip(x) + h)

    class Net(nn.Module):
        def __init__(self, f_seq, f_stat, d_emb=8, d=64, drop=0.1):
            super().__init__()
            self.emb_w = nn.Parameter(torch.randn(f_seq, d_emb) * 0.1)
            self.emb_b = nn.Parameter(torch.zeros(f_seq, d_emb))
            self.static = GRN(f_stat, d, drop=drop)
            self.select = GRN(f_seq * d_emb, f_seq, d_ctx=d, drop=drop)
            self.proj = GRN(d_emb, d, drop=drop)
            self.h0 = nn.Linear(d, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.att = nn.Linear(d, 1)
            self.head = GRN(2 * d, d, drop=drop)
            self.out = nn.Linear(d, 1)

        def forward(self, seq, stat):
            n, k, f = seq.shape
            c = self.static(stat)                                          # [n, d]
            e = seq.unsqueeze(-1) * self.emb_w + self.emb_b                # [n, k, f, d_emb]
            w = torch.softmax(self.select(e.reshape(n, k, -1), c.unsqueeze(1).expand(n, k, -1)), -1)
            x = self.proj((w.unsqueeze(-1) * e).sum(2))                    # [n, k, d]
            h, _ = self.gru(x, torch.tanh(self.h0(c)).unsqueeze(0).contiguous())
            a = torch.softmax(self.att(h).squeeze(-1), 1)                  # weight per month
            ctx = (a.unsqueeze(-1) * h).sum(1)
            return self.out(self.head(torch.cat([ctx, c], -1))).squeeze(-1), a, w.mean(1)

    T = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)        # noqa: E731
    sq, st, yy = T(tr[0]), T(tr[1]), T(y_tr.astype(np.float32))
    fit_i, val_i = np.flatnonzero(~val), np.flatnonzero(val)
    net = Net(sq.shape[2], st.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = fit_i[np.random.default_rng(seed + ep).permutation(len(fit_i))]
        for i in range(0, len(perm), batch):
            b = torch.as_tensor(perm[i:i + batch], device=dev)
            opt.zero_grad()
            loss_fn(net(sq[b], st[b])[0], yy[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = sum(loss_fn(net(sq[b], st[b])[0], yy[b]).item() * len(b)
                     for b in (torch.as_tensor(val_i[i:i + 8192], device=dev) for i in range(0, len(val_i), 8192))
                     ) / len(val_i)
        sched.step(vl)
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    print(f"    nn seed {seed} ({dev}): {ep + 1} epochs, best held-out-year loss {best:.4f}", flush=True)
    with torch.no_grad():
        out = [net(T(te[0][i:i + 8192]), T(te[1][i:i + 8192])) for i in range(0, len(te[0]), 8192)]
    logit, att, gates = (torch.cat([o[j] for o in out]) for j in range(3))
    return torch.sigmoid(logit).cpu().numpy(), att.cpu().numpy(), gates.cpu().numpy()


# ------------------------------------------------------------------ evaluation

def score(df, prob):
    if df.y.nunique() < 2:
        return np.nan, np.nan
    return average_precision_score(df.y, df[prob]), brier_score_loss(df.y, df[prob].clip(0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.02)
    ap.add_argument("--model", choices=["logreg", "gbm", "nn"], default="gbm")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--k", type=int, default=12, help="months of history")
    ap.add_argument("--per-month", type=int, default=3000, help="squares sampled per issue month")
    ap.add_argument("--cv", choices=["year", "region"], default="year")
    ap.add_argument("--permanent", type=float, nargs="+", default=[0.8, 0.9, 0.95, 1.0])
    ap.add_argument("--rare", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _qanom")
    ap.add_argument("--nn-seeds", type=int, default=1, help="NN only: average this many training runs")
    ap.add_argument("--drop", nargs="+", default=[], choices=sorted(GROUPS),
                    help="train without these input groups (ablation)")
    a = ap.parse_args()
    global NN_SEEDS
    NN_SEEDS = a.nn_seeds
    res = round(a.res, 2)

    d = load(res, a.drop)
    rng = np.random.default_rng(a.seed)
    s = sample(d, a.k, max(a.horizons), a.per_month, rng)
    seq, stat, names = features(d, s, a.k)
    m = d["months"]
    freq_all = np.nanmean(d["y"], 0)                                     # reporting groups only
    region = (d["lat"][s[:, 1]] >= 7.5) * 2 + (d["lon"][s[:, 2]] >= 30.5)
    issue_lo, issue_hi = s[:, 0] - a.k + 1, s[:, 0]
    tt = s[:, 0][:, None] + np.arange(-a.k + 1, 1)[None, :]
    cal_tt = m.month.to_numpy()[tt] - 1
    qcell = d["q_idx"][s[:, 1], s[:, 2]] if "q" in d else None
    print(f"inputs: history {names}, flood state t-1..t-3, neighbourhood t-1, lat, lon, "
          f"{'river distance and size, ' if 'q_dist' in d else ''}normal, target month. "
          f"Discharge as anomaly: {'q' in d}")

    out_dir = Path(__file__).resolve().parent / "output" / f"models_{res:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, normals, qnorm = [], {}, {}
    for h in a.horizons:
        tgt = s[:, 0] + h
        y_true = d["y"][tgt, s[:, 1], s[:, 2]]
        cal = m[tgt].month.to_numpy()
        tgt_year = m[tgt].year.to_numpy()
        for i, Y in enumerate(TEST_YEARS):
            if Y not in normals:
                normals[Y] = normal(d, Y)
                if "q" in d:
                    qnorm[Y] = discharge_normals(d, Y)
            seq_f = to_anomaly(seq, d, names, qnorm[Y], cal_tt, qcell, s) if "q" in d else seq
            nm = normals[Y][cal - 1, s[:, 1], s[:, 2]]
            st = np.column_stack([stat, nm, np.sin(2 * np.pi * cal / 12), np.cos(2 * np.pi * cal / 12),
                                  *accumulated(seq_f, names)])
            touch = (m[issue_lo].year.to_numpy() <= Y) & (m[issue_hi].year.to_numpy() >= Y)
            test = tgt_year == Y
            train = (tgt_year != Y) & ~touch
            if a.cv == "region":
                test &= region == i % 4
                train &= region != i % 4
            if test.sum() == 0:
                continue
            yb = (y_true >= 0.5).astype(int)
            tr_years = np.unique(tgt_year[train])
            val = tgt_year[train] == tr_years[np.argmin(np.abs(tr_years - Y))]   # nearest training year
            p, att, gates = fit_predict(a.model, (seq_f[train], st[train], yb[train]),
                                        (seq_f[test], st[test]), a.seed, val)
            df = pd.DataFrame({
                "horizon": h, "test_year": Y, "issue": m[s[test, 0]].astype(str),
                "row": s[test, 1], "col": s[test, 2], "y": yb[test], "share": y_true[test],
                "p_model": p, "p_climatology": np.nan_to_num(nm[test]),
                "p_persistence": stat[test, 0], "freq_all": freq_all[s[test, 1], s[test, 2]]})
            if att is not None:
                for j in range(a.k):
                    df[f"att_lag{a.k - 1 - j}"] = att[:, j]
            if gates is not None:
                for j, nm_ in enumerate(names):
                    df[f"gate_{nm_}"] = gates[:, j]
            preds.append(df)
            print(f"  h={h} test {Y}: train {train.sum():,} test {test.sum():,} "
                  f"PR-AUC {score(df, 'p_model')[0]:.3f}", flush=True)
    p = pd.concat(preds, ignore_index=True)
    p.to_parquet(out_dir / f"{a.model}_{a.cv}{a.tag}_preds.parquet", index=False)

    rows = []
    for thr in a.permanent:
        grp = np.where(p.freq_all >= thr, "always", np.where(p.freq_all < a.rare, "rarely", "seasonal"))
        for (h, g), df in p.assign(group=grp).groupby(["horizon", "group"]):
            for who in ("model", "climatology", "persistence"):
                ap_, br = score(df, f"p_{who}")
                rows.append({"permanent": thr, "horizon": h, "group": g, "n": len(df),
                             "base_rate": df.y.mean(), "who": who, "PR-AUC": ap_, "Brier": br})
    sc = pd.DataFrame(rows)
    sc.to_csv(out_dir / f"{a.model}_{a.cv}{a.tag}_scores.csv", index=False)
    main_thr = 0.9 if 0.9 in a.permanent else a.permanent[0]
    show = sc[sc.permanent == main_thr].pivot_table(
        index=["horizon", "group", "n", "base_rate"], columns="who", values=["PR-AUC", "Brier"])
    print(f"\n{a.model}, cv={a.cv}, groups with 'always' = water in >= {main_thr:.0%} of months")
    print("PR-AUC: higher is better (chance = base rate). Brier: lower is better.")
    print(show.round(3).to_string())
    sens = sc[(sc.who == "model") & (sc.group != "always")].groupby(["permanent", "horizon"]) \
        .apply(lambda g: np.average(g["PR-AUC"], weights=g.n), include_groups=False)
    print("\nSensitivity to the permanent-water threshold (model PR-AUC on non-always squares):")
    print(sens.unstack().round(3).to_string())


if __name__ == "__main__":
    main()
