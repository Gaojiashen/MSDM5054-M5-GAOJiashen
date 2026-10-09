"""Test local weekday and cross-store information borrowing on M5.

All forecasts use only observations available at each rolling origin. The
mixture has three reliability bins and three simplex weights per bin, shared
by all M5 series; hence nine values are reported and six are free.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
EVAL = ROOT / "outputs" / "evaluation"
CAL_DIR = EVAL / "calendar_effect"
RHO_DIR = EVAL / "rho_selection"
OUT = EVAL / "borrow_information"
FIG = OUT / "figures"
ORIGINS = tuple(range(121, 1886, 28))
FOLDS = ((32, 8), (40, 8), (48, 8), (56, 8))
EVENTS = ("Pesach End", "OrthodoxEaster", "Cinco De Mayo", "Mother's day")
GROUPS = ("state_id", "cat_id")
H = 28
KAPPAS = (5.0, 20.0, 80.0)
Q_SMOOTH_DAYS = 28.0
MAX_SHARE_ODDS = 20.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_matrix(cal: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        cal.event_name_1.eq(e).to_numpy() | cal.event_name_2.eq(e).to_numpy()
        for e in EVENTS
    ]).astype(float)


def setup(meta: pd.DataFrame):
    codes, keys = pd.factorize(pd.MultiIndex.from_frame(meta[list(GROUPS)]), sort=True)
    labels = [" × ".join(map(str, k)) for k in keys]
    rows = [np.flatnonzero(codes == i) for i in range(len(keys))]
    state = [meta.iloc[r[0]].state_id for r in rows]
    item_codes, items = pd.factorize(meta.item_id, sort=True)
    store_codes, stores = pd.factorize(meta.store_id, sort=True)
    cat_codes, cats = pd.factorize(meta.cat_id, sort=True)
    return codes, labels, rows, state, item_codes, items, store_codes, stores, cat_codes, cats


def params_for(coef: pd.DataFrame, fold: int, labels: list[str]):
    part = coef[(coef.fold == fold) & (coef.model == "both")]
    beta = np.array([float(part.loc[part.parameter == f"beta:{lab}", "log_multiplier"].iloc[0])
                     for lab in labels])
    gamma = np.array([float(part.loc[part.parameter == f"gamma:{e}", "log_multiplier"].iloc[0])
                      for e in EVENTS])
    return beta, gamma


def q_and_prior(train, meta, cal, release, origin, beta, group_codes):
    """Past-only store/category weekday multipliers and store share prior."""
    start = max(1, origin - 363)
    days = np.arange(start, origin + 1)
    sales = np.asarray(train[:, days - 1], dtype=float)
    usable = days[None, :] >= release[:, None]
    snap_col = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    # Align state snapshots through a per-leaf vector; reverse SNAP and omit
    # event-marked dates before estimating the weekday shape.
    hist_snap = np.empty_like(sales)
    for g in np.unique(group_codes):
        idx = np.flatnonzero(group_codes == g)
        hist_snap[idx] = cal.loc[days - 1, snap_col[meta.iloc[idx[0]].state_id]].to_numpy(float)
    holiday = (cal.loc[days - 1, "event_name_1"].fillna("").ne("").to_numpy() |
               cal.loc[days - 1, "event_name_2"].fillna("").ne("").to_numpy())
    de_calendar = sales * np.exp(-beta[group_codes, None] * hist_snap)
    de_calendar[:, holiday] = 0.0
    weekday = cal.loc[days - 1, "date"].pipe(pd.to_datetime).dt.dayofweek.to_numpy()
    store_codes, stores = pd.factorize(meta.store_id, sort=True)
    cat_codes, cats = pd.factorize(meta.cat_id, sort=True)
    key_code = store_codes * len(cats) + cat_codes
    q = np.ones((len(stores) * len(cats), 7), dtype=float)
    prior = np.zeros((len(cats), len(stores)), dtype=float)
    for k in np.unique(key_code):
        leaf = np.flatnonzero(key_code == k)
        vals = np.where(usable[leaf], de_calendar[leaf], 0.0).sum(axis=0)
        wd_sum = np.array([vals[weekday == w].sum() for w in range(7)])
        wd_n = np.array([np.sum((weekday == w) & ~holiday) for w in range(7)])
        mean = wd_sum.sum() / max(wd_n.sum(), 1)
        raw = (wd_sum + Q_SMOOTH_DAYS * mean / 7) / (wd_n + Q_SMOOTH_DAYS / 7)
        avg = raw.mean()
        q[k] = raw / avg if avg > 0 else 1.0
        store, cat = divmod(int(k), len(cats))
        prior[cat, store] = vals.sum()
    prior = prior / np.maximum(prior.sum(axis=1, keepdims=True), 1e-12)
    prior = np.where(prior > 0, prior, 1 / len(stores))
    prior /= prior.sum(axis=1, keepdims=True)
    return q, prior, store_codes, stores, cats, key_code


def reliability_bins(train, release, origin, cal):
    """Target weekday bin from positive counts among past eight weekdays."""
    out = np.zeros((len(release), 7), dtype=np.int8)
    for wd in range(7):
        dow = cal.loc[origin - 1, "date"]
        origin_wd = pd.Timestamp(dow).dayofweek
        last = origin - ((origin_wd - wd) % 7)
        days = last - 7 * np.arange(8)
        days = days[days >= 1]
        if len(days):
            y = np.asarray(train[:, days - 1], dtype=float)
            n = ((days[None, :] >= release[:, None]) & (y > 0)).sum(axis=1)
            out[:, wd] = np.where(n == 0, 0, np.where(n <= 2, 1, 2))
    return out


def local_experts(train, meta, cal, release, origin, rho, beta, gamma, rows,
                  group_codes, q, prior, store_codes, n_stores, n_cats, key_code,
                  item_codes, kappas=KAPPAS):
    """Return B, L-perp and X leaf forecasts for one origin."""
    b = np.zeros((len(meta), H)); l = np.zeros_like(b); x = np.zeros_like(b)
    days_out = np.arange(origin + 1, origin + H + 1)
    ev_out = event_matrix(cal.loc[days_out - 1])
    snap_col = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    group_state = [meta.iloc[r[0]].state_id for r in rows]
    target_snap = np.stack([cal.loc[days_out - 1, snap_col[s]].to_numpy(float) for s in group_state])
    wd_out = cal.loc[days_out - 1, "date"].pipe(pd.to_datetime).dt.dayofweek.to_numpy()
    bins_by_weekday = reliability_bins(train, release, origin, cal)
    start = max(1, origin - 55); all_days = np.arange(start, origin + 1)
    holiday_hist = (cal.loc[all_days - 1, "event_name_1"].fillna("").ne("").to_numpy() |
                    cal.loc[all_days - 1, "event_name_2"].fillna("").ne("").to_numpy())
    hist_wd = cal.loc[all_days - 1, "date"].pipe(pd.to_datetime).dt.dayofweek.to_numpy()
    hist_values = np.asarray(train[:, all_days - 1], dtype=float)
    hist_snap = np.empty_like(hist_values)
    for g, rr in enumerate(rows):
        hist_snap[rr] = cal.loc[all_days - 1, snap_col[group_state[g]]].to_numpy(float)
    # q shape is store × category × actual calendar weekday.
    leaf_cat = key_code % n_cats
    qhist = q[key_code[:, None], hist_wd[None, :]]
    hist_ev = event_matrix(cal.loc[all_days - 1])
    de = hist_values * np.exp(-beta[group_codes, None] * hist_snap - hist_ev @ gamma)
    de = de / np.maximum(qhist, .05)
    available56 = all_days[None, :] >= release[:, None]
    available56 &= ~holiday_hist[None, :]
    hist_decay = np.power(rho[group_codes, None], (origin - all_days[None, :]) / 7.0)

    for g, rr in enumerate(rows):
        state = group_state[g]
        snap_hist = cal.loc[all_days - 1, snap_col[state]].to_numpy(float)
        weights_all = np.power(float(rho[g]), (origin - all_days) / 7.0)
        for phase in range(7):
            ix = np.arange(phase, H, 7)
            target_wd = int(wd_out[phase])
            # Same weekday observations; remove every historically event-marked day.
            last = origin - 6 + phase
            hist_days = last - 7 * np.arange(8)
            valid = hist_days >= 1
            hist_days = hist_days[valid]
            if not len(hist_days):
                continue
            pos = hist_days - start
            if np.any((pos < 0) | (pos >= len(all_days))):
                continue
            # Match B's same-weekday profile: newest available weekday gets
            # exponent zero, then one exponent per seven-day step.
            w = np.power(float(rho[g]), np.arange(len(hist_days), dtype=float))
            hol = (cal.loc[hist_days - 1, "event_name_1"].fillna("").ne("").to_numpy() |
                   cal.loc[hist_days - 1, "event_name_2"].fillna("").ne("").to_numpy())
            snaps = cal.loc[hist_days - 1, snap_col[state]].to_numpy(float)
            vals = hist_values[np.ix_(rr, pos)]
            available8 = hist_days[None, :] >= release[rr, None]
            keep = available8 & (~hol)[None, :]
            corrected = vals * np.exp(-beta[g] * snaps)[None, :]
            den = (keep * w[None, :]).sum(axis=1)
            profile = np.divide((corrected * keep * w[None, :]).sum(axis=1), den,
                                out=np.zeros(len(rr)), where=den > 0)
            b[rr[:, None], ix[None, :]] = profile[:, None] * np.exp(
                beta[g] * target_snap[g, ix] + ev_out[ix] @ gamma)[None, :]
            cat_codes = leaf_cat[rr]
            qtarget = q[store_codes[rr] * n_cats + cat_codes, target_wd]
            factor = np.exp(beta[g] * target_snap[g, ix] + ev_out[ix] @ gamma)
            # L_perp uses all released non-event weekdays except the target
            # weekday to estimate a de-weekdayed level, then restores q(target).
            other_weekday = hist_wd != target_wd
            lw = hist_decay[rr] * available56[rr] * other_weekday[None, :]
            lden = lw.sum(axis=1)
            level = np.divide((de[rr] * lw).sum(axis=1), lden,
                              out=np.full(len(rr), np.nan), where=lden > 0)
            lprof = level * qtarget
            lprof = np.where(np.isfinite(lprof), lprof, profile)
            l[rr[:, None], ix[None, :]] = lprof[:, None] * factor[None, :]

    # Cross-store rates and H shares are vectorized over all item-store rows.
    # Ignore all event-marked dates for H and cross-store rates, including
    # holidays that do not have one of the four fitted gamma coefficients.
    raw_available = (all_days[None, :] >= release[:, None]) & ~holiday_hist[None, :]
    hist_weight = np.power(rho[group_codes, None], (origin - all_days[None, :]) / 7.0)
    own_h = (np.where(raw_available, de, 0.0)).sum(axis=1)
    den_rate = (raw_available * hist_weight).sum(axis=1)
    own_rate = np.divide((np.where(raw_available, de, 0.0) * hist_weight).sum(axis=1), den_rate,
                         out=np.zeros(len(meta)), where=den_rate > 0)
    item_h = np.bincount(item_codes, weights=own_h)
    item_rate = np.bincount(item_codes, weights=own_rate)
    item_days = np.bincount(item_codes, weights=raw_available.sum(axis=1).astype(float))
    own_days = raw_available.sum(axis=1)
    other_rate = np.maximum(item_rate[item_codes] - own_rate, 0.0)
    has_other = item_days[item_codes] > own_days
    p0 = prior[leaf_cat, store_codes]
    target_snap_leaf = np.empty((len(meta), H), dtype=float)
    for g, rr in enumerate(rows):
        target_snap_leaf[rr] = target_snap[g]
    target_factor = np.exp(beta[group_codes, None] * target_snap_leaf + ev_out @ gamma)
    qtarget = q[store_codes[:, None] * n_cats + leaf_cat[:, None], wd_out[None, :]]
    supported = has_other & (release <= origin)
    x_by_kappa = {}
    for kappa in kappas:
        p = (own_h + kappa * p0) / np.maximum(item_h[item_codes] + kappa, 1e-12)
        odds = np.minimum(p / np.maximum(1.0 - p, 1e-8), MAX_SHARE_ODDS)
        x_pred = other_rate[:, None] * odds[:, None] * target_factor * qtarget
        x_by_kappa[float(kappa)] = np.where(supported[:, None], x_pred, b)
    # Reliability labels are looked up by the actual weekday of every target date.
    bins = bins_by_weekday[:, wd_out]
    return b, l, x_by_kappa, bins


def score_mixture(train, meta, prediction, origin, metric, aggs, calendar, prices, metric_parts=None):
    if metric_parts is None:
        scales, weights, _ = metric.scales_and_weights(train, meta, aggs, calendar, prices, origin)
    else:
        scales, weights = metric_parts
    actual = np.asarray(train[:, origin:origin + H], dtype=float)
    total = 0.0
    for agg, scale, weight in zip(aggs, scales, weights):
        py = np.asarray(agg @ prediction); yy = np.asarray(agg @ actual)
        mse = np.mean((py - yy) ** 2, axis=1)
        valid = scale > 0
        rms = np.zeros_like(scale)
        rms[valid] = np.sqrt(mse[valid] / scale[valid])
        total += float(weight @ rms) / len(aggs)
    return total


def gram_record(b, l, x, bins, train, origin, aggs, metric_parts):
    """Compress one origin to per-node quadratic WRMSSE coefficients."""
    scales, weights = metric_parts
    actual = np.asarray(train[:, origin:origin + H], dtype=float)
    components = [b - actual]
    for bi in range(3):
        mask = bins == bi
        components.append(np.where(mask, l - b, 0.0))
        components.append(np.where(mask, x - b, 0.0))
    pairs = [(i, j) for i in range(7) for j in range(i, 7)]
    packed = []
    for agg, scale, weight in zip(aggs, scales, weights):
        agg_parts = [np.asarray(agg @ part, dtype=np.float64) for part in components]
        coeff = np.zeros((agg.shape[0], len(pairs)), dtype=np.float32)
        valid = scale > 0
        for p, (i, j) in enumerate(pairs):
            value = np.mean(agg_parts[i] * agg_parts[j], axis=1)
            value = np.divide(value, scale, out=np.zeros_like(value), where=valid)
            if i != j:
                value *= 2.0
            coeff[:, p] = value.astype(np.float32)
        packed.append((coeff, weight.astype(np.float32)))
    return {"pairs": pairs, "levels": packed}


def score_gram(gram, weights):
    """Evaluate exact level-wise WRMSSE from cached node Gram terms."""
    a = np.array([1.0, weights[0,1], weights[0,2], weights[1,1], weights[1,2],
                  weights[2,1], weights[2,2]], dtype=float)
    terms = np.array([a[i] * a[j] for i, j in gram["pairs"]], dtype=float)
    total = 0.0
    for coeff, weight in gram["levels"]:
        mse_scaled = np.asarray(coeff, dtype=np.float64) @ terms
        total += float(np.dot(weight, np.sqrt(np.maximum(mse_scaled, 0.0)))) / len(gram["levels"])
    return total


def optimize_weights(records, kinds):
    """Fit simplex mixture against cached hierarchy Gram coefficients."""
    def unpack(theta):
        w = np.zeros((3, 3)); cursor = 0
        for b in range(3):
            if kinds == "B": w[b, 0] = 1
            elif kinds == "BL": w[b, :2] = [theta[cursor], 1 - theta[cursor]]; cursor += 1
            elif kinds == "BX": w[b, [0, 2]] = [theta[cursor], 1-theta[cursor]]; cursor += 1
            else:
                a, z = theta[cursor:cursor+2]; cursor += 2
                w[b] = [a, z, 1-a-z]
        return w
    n = {"B": 0, "BL": 3, "BX": 3, "BLX": 6}[kinds]
    def objective(theta):
        w = unpack(theta)
        return float(np.mean([score_gram(rec["gram"], w) for rec in records]))
    if n == 0:
        return unpack(np.zeros(0)), objective(np.zeros(0))
    x0 = np.tile([1/3, 1/3], 3) if kinds == "BLX" else np.full(n, .5)
    cons = []
    if kinds == "BLX":
        cons = [{"type": "ineq", "fun": lambda t, i=i: 1-t[2*i]-t[2*i+1]} for i in range(3)]
    res = minimize(objective, x0, method="SLSQP", bounds=[(0, 1)]*n,
                   constraints=cons, options={"maxiter":100, "ftol":1e-7})
    if not res.success and not np.isfinite(res.fun):
        raise RuntimeError(f"mixture optimization failed: {res.message}")
    return unpack(res.x), float(res.fun)


def plot_results(means: pd.DataFrame, scores: pd.DataFrame,
                 final_weight_rows: list[dict]):
    """Show both the broad gain and the tiny incremental cross-store effect."""
    best_by = (means.sort_values("wrmsse")
               .drop_duplicates("candidate")
               .set_index("candidate")
               .loc[["B", "BX", "BL", "BLX"]])
    labels = {"B": "Calendar baseline", "BX": "+ other stores",
              "BL": "+ other weekdays", "BLX": "+ both sources"}
    colors = {"B": "#778594", "BX": "#6F9CB4",
              "BL": "#2B8C8C", "BLX": "#193F5E"}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.18]})
    ax = axes[0]
    for y, (name, row) in enumerate(best_by.iterrows()):
        value = float(row.wrmsse)
        ax.hlines(y, .695, value, color=colors[name], linewidth=2.5)
        ax.plot(value, y, "o", color=colors[name], markersize=9)
        ax.text(value + .0007, y, f"{value:.6f}", va="center", fontsize=10)
    ax.set_yticks(range(4), [labels[n] for n in best_by.index])
    ax.invert_yaxis(); ax.set_xlim(.695, .742)
    ax.set_xlabel("Mean validation WRMSSE  ·  lower is better")
    ax.set_title("Overall comparison", loc="left", weight="bold")
    ax.grid(axis="x", alpha=.17)
    pivot = scores.pivot_table(index=["fold", "origin_d"],
                               columns=["candidate", "kappa"], values="wrmsse")
    delta = (pivot[("BL", 5.0)] - pivot[("BLX", 80.0)]).reset_index(name="gain")
    ax = axes[1]
    ax.axhline(0, color="#697887", linewidth=1)
    for fold, part in delta.groupby("fold"):
        ax.scatter(part.origin_d, part.gain, s=35, color=colors["BLX"],
                   alpha=.8, label="Origin" if fold == 1 else None)
        if fold < 4:
            ax.axvline(part.origin_d.max() + 14, color="#D7DEE3", linewidth=1)
    ax.axhline(delta.gain.mean(), color="#CB703D", linewidth=1.8,
               linestyle="--", label=f"Mean gain {delta.gain.mean():+.6f}")
    ax.set_xlabel("Validation origin (d number)")
    ax.set_ylabel("BL WRMSSE − BLX WRMSSE")
    ax.set_title("Does the cross-store expert add value?", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.17)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIG / "01_validation_candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    wt = (pd.DataFrame(final_weight_rows)
          .pivot(index="reliability_bin", columns="expert", values="weight")
          .loc[["0", "1-2", ">=3"], ["B", "L_perp", "X"]]
          .clip(lower=0))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    color = {"B": "#778594", "L_perp": "#2B8C8C", "X": "#CB703D"}
    names = {"B": "Baseline", "L_perp": "Other weekdays", "X": "Other stores"}
    left = np.zeros(3)
    for col in wt.columns:
        values = wt[col].to_numpy()
        ax.barh(np.arange(3), values, left=left, height=.56,
                color=color[col], label=names[col])
        for i, value in enumerate(values):
            if value > .08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}",
                        ha="center", va="center", color="white", weight="bold")
        left += values
    ax.set_yticks(range(3), ["0 positive days", "1–2 positive days", "3–8 positive days"])
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Final mixture weight")
    ax.set_title("How much does each source contribute?", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.16))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "02_final_weights.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    # This script intentionally has no smoke execution: a full run is supervised
    # because each origin requires constructing three 30,490 × 28 forecasts.
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    metric = load_module(Path(__file__).with_name("02_select_x_wrmsse.py"), "borrow_metric")
    base_metric = load_module(Path(__file__).with_name("01_snaive_wrmsse.py"), "borrow_holdout_metric")
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (30490, 1913):
        raise ValueError(f"Unexpected M5 sales matrix {train.shape}")
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["d_num"] = cal.d.str.removeprefix("d_").astype(int)
    cal["holiday"] = cal.event_name_1.notna() & cal.event_name_1.ne("") | cal.event_name_2.notna() & cal.event_name_2.ne("")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    release = metric.first_price_days(meta, prices, cal)
    codes, labels, rows, _, item_codes, _, store_codes, stores, cat_codes, cats = setup(meta)
    aggs = metric.hierarchy(meta)
    rho_fold = pd.read_csv(RHO_DIR / "learned_rho_by_fold.csv")
    rho_final_df = pd.read_csv(RHO_DIR / "learned_rho_final.csv")
    coef_fold = pd.read_csv(CAL_DIR / "calendar_coefficients_by_fold.csv")
    coef_final = pd.read_csv(CAL_DIR / "final_calendar_coefficients.csv")
    # Calendar experiment's predeclared winner is both; the prior run records
    # this choice in its summary. The guard prevents silently using another model.
    summary06 = json.loads((CAL_DIR / "experiment_summary.json").read_text(encoding="utf-8"))
    if summary06["selected_model_by_validation"] != "both":
        raise ValueError("Experiment 08 is specified against calendar model 'both'; 06 selected another model")

    # Process one fold/kappa at a time. Forecast arrays are large, so discard
    # each temporary cache after scoring instead of retaining them all in RAM.
    # The same origin-specific WRMSSE scales/weights recur in multiple folds
    # and for every kappa; compute them once and reuse.
    metric_parts_by_origin={}
    for count, origin in enumerate(ORIGINS, 1):
        metric_parts_by_origin[origin] = metric.scales_and_weights(train,meta,aggs,cal,prices,origin)[:2]
        if count % 8 == 0:
            print(f"WRMSSE scales: {count}/{len(ORIGINS)} origins", flush=True)
    b_score_rows=[]; result_rows=[]; weight_rows=[]; choice_rows=[]
    candidate_names=("B","BL","BX","BLX")
    for fold,(ntrain,nvalid) in enumerate(FOLDS,1):
        print(f"Fold {fold}/{len(FOLDS)}: constructing experts and Gram records", flush=True)
        train_orig=ORIGINS[:ntrain]; valid_orig=ORIGINS[ntrain:ntrain+nvalid]
        rrho=rho_fold[(rho_fold.fold==fold)&(rho_fold.partition=="state_category")]
        rho=np.array([float(rrho.loc[rrho.group==lab,"rho"].iloc[0]) for lab in labels])
        beta,gamma=params_for(coef_fold,fold,labels)
        candidate_summary=[]
        gram_caches={k:{ } for k in KAPPAS}
        for origin in (*train_orig,*valid_orig):
            q,prior,_,_,_,key_code=q_and_prior(train,meta,cal,release,origin,beta,codes)
            b,l,x_by_kappa,bins=local_experts(train,meta,cal,release,origin,rho,beta,gamma,
                rows,codes,q,prior,store_codes,len(stores),len(cats),key_code,item_codes,KAPPAS)
            metric_parts=metric_parts_by_origin[origin]
            if origin in valid_orig:
                b_score_rows.append({"fold":fold,"origin_d":origin,
                    "wrmsse":score_mixture(train,meta,b,origin,metric,aggs,cal,prices,metric_parts)})
            for kappa in KAPPAS:
                gram=gram_record(b.astype(np.float32),l.astype(np.float32),
                    x_by_kappa[float(kappa)].astype(np.float32),bins,train,origin,aggs,metric_parts)
                gram_caches[kappa][origin]={"gram":gram,"origin":origin}
            del b,l,x_by_kappa
        for kappa in KAPPAS:
            recs=[gram_caches[kappa][o] for o in train_orig]
            for model in candidate_names:
                weights, train_loss=optimize_weights(recs,model)
                vrecs=[gram_caches[kappa][o] for o in valid_orig]
                valid_losses=[]
                for rec in vrecs:
                    loss=score_gram(rec["gram"],weights)
                    valid_losses.append(loss)
                    result_rows.append({"fold":fold,"origin_d":rec["origin"],"kappa":kappa,"candidate":model,"wrmsse":loss})
                candidate_summary.append({"fold":fold,"kappa":kappa,"candidate":model,"mean_validation_wrmsse":float(np.mean(valid_losses)),"train_wrmsse":train_loss,"weights":weights})
                for bi in range(3):
                    for expert,weight in zip(("B","L_perp","X"),weights[bi]):
                        weight_rows.append({"fold":fold,"kappa":kappa,"candidate":model,"reliability_bin":("0","1-2",">=3")[bi],"expert":expert,"weight":float(weight)})
            del recs
        print(f"Fold {fold}/{len(FOLDS)}: weight search complete", flush=True)
        del gram_caches
        candidate_summary.sort(key=lambda r:(r["mean_validation_wrmsse"],r["kappa"],candidate_names.index(r["candidate"])))
        best=candidate_summary[0]
        choice_rows.append({"fold":fold,"selected_kappa":best["kappa"],"selected_candidate":best["candidate"],"mean_validation_wrmsse":best["mean_validation_wrmsse"]})
    # Mandatory baseline parity with the saved 06 validation scores.
    parity=pd.DataFrame(b_score_rows)
    reference=pd.read_csv(CAL_DIR/"rolling_validation_scores.csv")
    reference=reference[reference.model=="both"]
    merged=parity.merge(reference,on=["fold","origin_d"],suffixes=("_08","_06"))
    if len(merged)!=32:
        raise AssertionError(f"Expected 32 matched B validation origins, got {len(merged)}")
    max_parity=float(np.max(np.abs(merged.wrmsse_08-merged.wrmsse_06)))
    if max_parity>2e-5:
        raise AssertionError(f"B parity with 06 failed: max score delta {max_parity:g}")
    pd.DataFrame(result_rows).to_csv(OUT/"rolling_candidate_scores.csv",index=False)
    pd.DataFrame(weight_rows).to_csv(OUT/"fold_mixture_weights.csv",index=False)
    pd.DataFrame(choice_rows).to_csv(OUT/"fold_selections.csv",index=False)
    pd.DataFrame(b_score_rows).to_csv(OUT/"baseline_parity_scores.csv",index=False)

    # Compare candidates over the four validation blocks; ties select simpler model.
    scores=pd.DataFrame(result_rows)
    rank={name:i for i,name in enumerate(candidate_names)}
    means=scores.groupby(["candidate","kappa"],as_index=False).wrmsse.mean()
    means["tie_rank"]=means.candidate.map(rank)
    means=means.sort_values(["wrmsse","kappa","tie_rank"],kind="stable")
    best=means.iloc[0]
    final_kappa=float(best.kappa); final_candidate=str(best.candidate)
    # Final coefficients fit on all 64 rolling origins, frozen before holdout load.
    final_rho=np.array([float(rho_final_df.loc[rho_final_df.group == lab,"rho"].iloc[0]) for lab in labels])
    beta,gamma=params_for(coef_final.assign(fold=1),1,labels)
    final_records=[]
    print("Refitting selected mixture on all historical origins", flush=True)
    for origin in ORIGINS:
        q,prior,_,_,_,key_code=q_and_prior(train,meta,cal,release,origin,beta,codes)
        b,l,x_by_kappa,bins=local_experts(train,meta,cal,release,origin,final_rho,beta,gamma,
            rows,codes,q,prior,store_codes,len(stores),len(cats),key_code,item_codes,(final_kappa,))
        gram=gram_record(b.astype(np.float32),l.astype(np.float32),x_by_kappa[final_kappa].astype(np.float32),
                         bins,train,origin,aggs,metric_parts_by_origin[origin])
        final_records.append({"gram":gram,"origin":origin})
    final_weights,_=optimize_weights(final_records,final_candidate)
    print("Final mixture fit complete; checking baseline parity and writing forecast", flush=True)
    target=np.arange(1914,1942)
    q,prior,_,_,_,key_code=q_and_prior(train,meta,cal,release,1913,beta,codes)
    b,l,x_by_kappa,bins=local_experts(train,meta,cal,release,1913,final_rho,beta,gamma,
        rows,codes,q,prior,store_codes,len(stores),len(cats),key_code,item_codes,(final_kappa,))
    x=x_by_kappa[final_kappa]
    reference_b=pd.read_csv(CAL_DIR/"forecast_both.csv")
    ref_ids=meta.item_id.astype(str)+"_"+meta.store_id.astype(str)+"_validation"
    if not np.array_equal(reference_b.id.astype(str).to_numpy(),ref_ids.to_numpy()):
        raise AssertionError("06 forecast_both.csv IDs/order differ from metadata")
    max_final_b=float(np.max(np.abs(b-reference_b[[f"d_{d}" for d in range(1914,1942)]].to_numpy(float))))
    # The earlier script uses float32 rho powers; recomputing them in float64
    # changes a few leaf forecasts by under 1e-5 units without changing scores.
    if max_final_b>2e-5:
        raise AssertionError(f"Final B forecast differs from 06 forecast_both.csv by {max_final_b:g}")
    forecast=b.copy()
    for bi in range(3):
        mask=bins==bi
        forecast[mask]+=final_weights[bi,1]*(l[mask]-b[mask])+final_weights[bi,2]*(x[mask]-b[mask])
    if not np.isfinite(forecast).all() or (forecast < -1e-10).any():
        raise AssertionError("Forecast must be finite and nonnegative")
    forecast=np.maximum(forecast,0)
    # Coherence is guaranteed by aggregating a single leaf table through M5 matrices.
    table=pd.DataFrame(forecast,columns=[f"d_{d}" for d in target])
    table.insert(0,"id",meta.item_id.astype(str)+"_"+meta.store_id.astype(str)+"_validation")
    table.to_csv(OUT/"forecast_borrow_information.csv",index=False)
    final_weight_rows=[]
    for bi in range(3):
        for expert,w in zip(("B","L_perp","X"),final_weights[bi]):
            final_weight_rows.append({"reliability_bin":("0","1-2",">=3")[bi],"expert":expert,"weight":float(w)})
    pd.DataFrame(final_weight_rows).to_csv(OUT/"final_mixture_weights.csv",index=False)
    # Holdout labels are read only after the forecast and final weights are saved.
    actual=base_metric.load_test_actuals(meta)
    cal_train=cal.loc[cal.d_num.between(1,1913)].copy(); cal_test=cal.loc[cal.d_num.between(1914,1941)].copy()
    weekset=set(cal_train.loc[cal_train.d_num.between(1886,1913),"wm_yr_wk"].astype(int))
    official_prices=base_metric.load_weight_window_prices(weekset)
    holdout,levels=base_metric.score_wrmsse(meta,train,actual,forecast,cal_train,cal_test,official_prices)
    levels.to_csv(OUT/"final_holdout_wrmsse_by_level_descriptive.csv",index=False)
    (OUT/"experiment_summary.json").write_text(json.dumps({
        "folds":[{"train_origins":a,"validation_origins":b} for a,b in FOLDS],
        "selected_kappa":final_kappa,"selected_candidate":final_candidate,
        "validation_mean_wrmsse":float(best.wrmsse),"baseline_max_abs_parity_delta_vs_06":max_parity,
        "baseline_final_forecast_max_abs_delta_vs_06":max_final_b,
        "final_weights":final_weight_rows,"holdout_wrmsse_descriptive":holdout,
        "forecast_days":[1914,1941],"holdout_read_after_forecast_frozen":True,
        "coefficient_source":"06 both and 04 state × category rho; fold-specific values during validation, final values for d_1914–d_1941",
        "q_smoothing_pseudo_days":Q_SMOOTH_DAYS,"kappa_grid":list(KAPPAS),
        "x_share_odds_cap":MAX_SHARE_ODDS,"nonnegative":True,"coherent_leaf_aggregation":True
    },indent=2,ensure_ascii=False),encoding="utf-8")
    means.to_csv(OUT/"validation_candidate_summary.csv",index=False)
    plot_results(means, scores, final_weight_rows)


if __name__ == "__main__":
    main()
