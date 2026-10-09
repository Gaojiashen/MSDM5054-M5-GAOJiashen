"""Evaluate holiday and SNAP calendar multipliers on the grouped-rho M5 model.

Calendar parameters are fitted on rolling-origin group totals through d_1913.
The d_1914..d_1941 actuals are read only after the final forecast is saved, for
the requested descriptive comparison with the existing holdout score.
"""
from __future__ import annotations

import argparse
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
RHO_DIR = ROOT / "outputs" / "evaluation" / "rho_selection"
OUT = ROOT / "outputs" / "evaluation" / "calendar_effect"
FIGURES = OUT / "figures"
ORIGINS = tuple(range(121, 1886, 28))
FOLDS = ((32, 8), (40, 8), (48, 8), (56, 8))
EVENTS = ("Pesach End", "OrthodoxEaster", "Cinco De Mayo", "Mother's day")
GROUP_FIELDS = ("state_id", "cat_id")
HORIZON = 28
X = 8
# Ridge penalties on log multipliers. The larger holiday penalty reflects only
# four target-window occurrences per event; SNAP has many repeated exposures.
LAMBDA_BETA = 2.0
GAMMA_RIDGE_MULTIPLIER = 1.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_matrix(calendar: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        calendar.event_name_1.eq(name).to_numpy() | calendar.event_name_2.eq(name).to_numpy()
        for name in EVENTS
    ]).astype(np.float64)


def prep_groups(meta: pd.DataFrame):
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(meta[list(GROUP_FIELDS)]), sort=True)
    labels = [" × ".join(map(str, key)) for key in uniques]
    if len(labels) != 9:
        raise ValueError(f"Expected nine state × category groups, got {len(labels)}")
    return codes.astype(np.int8), labels, [np.flatnonzero(codes == i) for i in range(9)]


def origin_data(train, origin, release, calendar, group_rows, rho):
    """Compact weekday history plus 28-day group actuals/calendar features."""
    histories = []
    for rows in group_rows:
        group_hist = []
        for wd in range(7):
            last = origin - 6 + wd
            days = last - 7 * np.arange(X)
            valid_day = days >= 1
            days = days[valid_day]
            values = np.asarray(train[np.ix_(rows, days - 1)], dtype=np.float64)
            group_release = release if len(release) == len(rows) else release[rows]
            available = days[None, :] >= group_release[:, None]
            # Calendar features align with newest-to-oldest same-weekday values.
            snap = np.zeros(len(days), dtype=np.float64)
            holiday = np.zeros(len(days), dtype=bool)
            if len(days):
                snap = calendar.loc[days - 1, "snap"].to_numpy(dtype=np.float64)
                holiday = calendar.loc[days - 1, "holiday"].to_numpy(dtype=bool)
            weights = np.power(np.float32(rho), np.arange(len(days), dtype=np.float32))
            if rho == 0 and len(weights):
                weights[1:] = 0
            group_hist.append((values, available, snap, holiday, weights))
        histories.append(group_hist)
    target_days = np.arange(origin + 1, origin + HORIZON + 1)
    target_events = np.stack(calendar.loc[target_days - 1, "events"].to_numpy()).astype(np.float64)
    target_snap = calendar.loc[target_days - 1, "snap"].to_numpy(dtype=np.float64)
    actual = np.stack([np.asarray(train[np.ix_(rows, target_days - 1)].sum(axis=0)).ravel()
                       for rows in group_rows])
    target_holiday = calendar.loc[target_days - 1, "holiday"].to_numpy(dtype=bool)
    fit_mask = (~target_holiday) | (target_events.sum(axis=1) > 0)
    return {"origin": origin, "history": histories, "events": target_events, "fit_mask": fit_mask,
            "snap": target_snap, "actual": actual}


def profiles_and_derivative(data, beta, skip_holidays: bool):
    """Group × weekday leaf forecast; derivative is w.r.t. each group's beta."""
    if "profile_sums" in data:
        sums = data["profile_sums"][:, int(skip_holidays)]
        snaps = data["snap_history"]
        correction = np.exp(-beta[:, None, None] * snaps)
        return np.sum(sums * correction, axis=2), -np.sum(sums * snaps * correction, axis=2)
    n_groups = len(data["history"])
    base = np.zeros((n_groups, 7), dtype=np.float64)
    deriv = np.zeros_like(base)
    for g in range(n_groups):
        for wd, (values, available, snap, holiday, weights) in enumerate(data["history"][g]):
            keep_day = (~holiday) if skip_holidays else np.ones(len(holiday), dtype=bool)
            adjusted_w = weights * keep_day
            row_w = available * adjusted_w[None, :]
            den = row_w.sum(axis=1)
            corrected = np.exp(-beta[g] * snap)
            num = (values * row_w * corrected[None, :]).sum(axis=1)
            num_snap = (values * row_w * corrected[None, :] * snap[None, :]).sum(axis=1)
            leaf = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            dleaf = np.divide(-num_snap, den, out=np.zeros_like(num), where=den > 0)
            base[g, wd] = leaf.sum()
            deriv[g, wd] = dleaf.sum()
    return base, deriv


def rolling_score(meta, train, forecast, origin, aggs, scales, weights):
    """Full 12-level origin-specific WRMSSE, with no later weights or scales."""
    actual = np.asarray(train[:, origin:origin + HORIZON], dtype=np.float64)
    total = 0.0
    for agg, scale, weight in zip(aggs, scales, weights):
        pred_nodes = np.asarray(agg @ forecast, dtype=np.float64)
        true_nodes = np.asarray(agg @ actual, dtype=np.float64)
        mse = np.mean((pred_nodes - true_nodes) ** 2, axis=1)
        valid = scale > 0
        rms = np.zeros_like(scale)
        rms[valid] = np.sqrt(mse[valid] / scale[valid])
        total += float(np.dot(weight, rms)) / len(aggs)
    return total


def unavailable_reference_counts(origin, rhos, first_release, cal, state_cols, group_rows,
                                 skip_holidays):
    counts = []
    for g, rows in enumerate(group_rows):
        missing = 0
        for wd in range(7):
            last = origin - 6 + wd
            days = last - 7 * np.arange(X)
            days = days[days >= 1]
            available = days[None, :] >= first_release[rows, None]
            snap_hist = cal.loc[days - 1, state_cols[g]].to_numpy(dtype=float)
            weights = np.power(rhos[g], np.arange(len(days), dtype=float))
            if rhos[g] == 0 and len(weights): weights[1:] = 0
            if skip_holidays:
                holiday = cal.loc[days - 1, "holiday"].to_numpy(dtype=bool)
                weights *= ~holiday
            den = (available * weights[None, :]).sum(axis=1)
            missing += int((den == 0).sum())
        counts.append(missing)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run invariant and small-data smoke checks")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    metric = load_module(Path(__file__).with_name("01_snaive_wrmsse.py"), "calendar_metric")
    metric_x = load_module(Path(__file__).with_name("02_select_x_wrmsse.py"), "calendar_rolling_metric")
    rho_mod = load_module(Path(__file__).with_name("04_select_rho_wrmsse.py"), "calendar_rho")
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    train = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if train.shape != (len(meta), 1913):
        raise ValueError("Expected sales through d_1913")
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["d_num"] = cal.d.str.removeprefix("d_").astype(int)
    cal["snap"] = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy().max(axis=1)
    # Group-specific SNAP is assembled later because each group belongs to one state.
    cal["events"] = list(event_matrix(cal))
    cal["holiday"] = cal.event_name_1.notna() & cal.event_name_1.ne("") | cal.event_name_2.notna() & cal.event_name_2.ne("")
    # Store per-state SNAP on rows; actual group feature comes from its state.
    codes, labels, group_rows = prep_groups(meta)
    state_by_group = [meta.iloc[rows[0]].state_id for rows in group_rows]
    state_snap_col = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    fold_specs = ((2, 1),) if args.smoke else FOLDS
    # Select one calendar row per group state by creating group-specific contexts.
    rho_fold = pd.read_csv(RHO_DIR / "learned_rho_by_fold.csv")
    rho_final = pd.read_csv(RHO_DIR / "learned_rho_final.csv")
    actual_snap_cols = [state_snap_col[s] for s in state_by_group]
    aggs = metric_x.hierarchy(meta)

    prices = pd.read_csv(RAW / "sell_prices.csv", usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    first_release = metric_x.first_price_days(meta, prices, cal)

    def predict_local(data, params, model, with_jac=False):
        beta = np.zeros(9); gamma = np.zeros(4)
        if model in ("snap_only", "both"): beta = params[:9]
        if model == "holiday_only": gamma = params[:4]
        elif model == "both": gamma = params[9:13]
        mu = np.zeros((9, 28)); jac = np.zeros((9, 28, len(params)))
        skip = model in ("holiday_only", "both")
        for g in range(9):
            local = {"profile_sums":data["profile_sums"][g:g+1],
                     "snap_history":data["snap_history"][g:g+1]}
            prof, dp = profiles_and_derivative(local, np.array([beta[g]]), skip)
            eta = beta[g] * data["snap"][g] + data["events"][g] @ gamma
            mult = np.exp(eta)
            wd = np.arange(28) % 7
            mu[g] = prof[0, wd] * mult
            if with_jac:
                if model in ("snap_only", "both"):
                    jac[g, :, g] = mult * (dp[0, wd] + prof[0, wd] * data["snap"][g])
                if model == "holiday_only":
                    jac[g, :, :] = mu[g, :, None] * data["events"][g]
                elif model == "both":
                    jac[g, :, 9:13] = mu[g, :, None] * data["events"][g]
        return (mu, jac) if with_jac else mu

    def gamma_exposure(contexts):
        exposure=np.zeros(4,dtype=float)
        zero=np.zeros(13)
        for d in contexts:
            mu=predict_local(d,zero,"both")
            for e in range(4):
                exposure[e]+=float(np.sum(mu*d["events"][:,:,e]*d["fit_mask"][None,:]))
        return exposure

    def fit_local(contexts, model, gamma_multiplier=GAMMA_RIDGE_MULTIPLIER):
        n = {"baseline": 0, "holiday_only": 4, "snap_only": 9, "both": 13}[model]
        if not n: return np.zeros(0)
        exposure=gamma_exposure(contexts) if model in ("holiday_only","both") else np.zeros(4)
        lam = np.array(([LAMBDA_BETA] * 9 if model == "snap_only" else
                        [LAMBDA_BETA] * 9 + list(gamma_multiplier*exposure) if model == "both" else
                        list(gamma_multiplier*exposure)), dtype=float)
        def fun(p):
            loss = 0.; grad = np.zeros(n)
            for d in contexts:
                mu, jac = predict_local(d, p, model, True)
                safe = np.maximum(mu, 1e-12); y = d["actual"]
                mask = d["fit_mask"][None, :]
                loss += np.sum((safe - y * np.log(safe)) * mask)
                grad += np.einsum("gh,ghp->p", (1-y/safe) * mask, jac)
            return loss + np.dot(lam, p*p), grad + 2*lam*p
        res = minimize(fun, np.zeros(n), jac=True, method="L-BFGS-B", bounds=[(-.7,.7)]*n,
                       options={"maxiter":120,"ftol":1e-10,"gtol":1e-6})
        return res.x

    def leaf_local(origin, rhos, params, model, target_events, target_snap):
        beta = np.zeros(9); gamma = np.zeros(4)
        if model in ("snap_only", "both"): beta = params[:9]
        if model == "holiday_only": gamma = params[:4]
        elif model == "both": gamma = params[9:13]
        result = np.zeros((len(meta),28)); wd = np.arange(28)%7
        for g, rows in enumerate(group_rows):
            for weekday in range(7):
                last=origin-6+weekday; days=last-7*np.arange(8); days=days[days>=1]
                vals=np.asarray(train[np.ix_(rows,days-1)],dtype=float)
                av=days[None,:]>=first_release[rows,None]
                snap_hist=cal.loc[days-1,state_snap_col[state_by_group[g]]].to_numpy(float)
                hol=cal.loc[days-1,"holiday"].to_numpy(bool)
                w=np.power(np.float32(rhos[g]),np.arange(len(days),dtype=np.float32))
                if rhos[g]==0: w[1:]=0
                if model in ("holiday_only","both"): w*=~hol
                aw=av*w[None,:]
                den=aw.sum(axis=1)
                corrected=vals*np.exp(-beta[g]*snap_hist)[None,:]
                prof=np.divide((corrected*aw).sum(axis=1),den,out=np.zeros(len(rows)),where=den>0)
                idx=np.flatnonzero(wd==weekday)
                factor=np.exp(beta[g]*target_snap[g,idx]+target_events[idx]@gamma)
                result[rows[:,None],idx[None,:]]=prof[:,None]*factor[None,:]
        return result

    def make_context_local(origin, rhos):
        profile_sums=np.zeros((9,2,7,8),dtype=np.float64)
        snap_history=np.zeros((9,7,8),dtype=np.float64)
        events=[]; snaps=[]; actual=[]
        target_days=np.arange(origin+1,origin+29)
        for g,rows in enumerate(group_rows):
            cc=cal.copy(); cc["snap"]=cc[state_snap_col[state_by_group[g]]].astype(float)
            rel=np.where(first_release[rows]<=origin,first_release[rows],10**9)
            one=origin_data(train,origin,rel,cc,[rows],rhos[g])
            events.append(one["events"]); snaps.append(one["snap"]); actual.append(one["actual"][0])
            for wd,(values,available,snap_hist,holiday,weights) in enumerate(one["history"][0]):
                snap_history[g,wd,:len(snap_hist)]=snap_hist
                for skip in (0,1):
                    keep=(~holiday) if skip else np.ones(len(holiday),dtype=bool)
                    raw=available*(weights*keep)[None,:]
                    den=raw.sum(axis=1)
                    fractions=np.divide(raw,den[:,None],out=np.zeros_like(raw,dtype=float),where=den[:,None]>0)
                    profile_sums[g,skip,wd,:len(snap_hist)]=(values*fractions).sum(axis=0)
        event_array=np.stack(events)
        target_holiday=cal.loc[target_days-1,"holiday"].to_numpy(bool)
        fit_mask=(~target_holiday)|(event_array[0].sum(axis=1)>0)
        return {"origin":origin,"profile_sums":profile_sums,"snap_history":snap_history,
                "events":event_array,"snap":np.stack(snaps),"actual":np.stack(actual),"fit_mask":fit_mask}

    # Replace the common-calendar SNAP feature with state-specific features, then run folds.
    score_rows=[]; coef_rows=[]; fitted_by_fold={}
    fold_event_counts=[]; ridge_rows=[]
    for fold_no,(ntrain,nvalid) in enumerate(fold_specs,1):
        train_origins=ORIGINS[:ntrain]; valid_origins=ORIGINS[ntrain:ntrain+nvalid]
        rho_part=rho_fold[(rho_fold.fold==fold_no)&(rho_fold.partition=="state_category")]
        rhos=np.array([float(rho_part.set_index("group").loc[label,"rho"]) for label in labels])
        fit_contexts=[make_context_local(o,rhos) for o in train_origins]
        event_exposure=gamma_exposure(fit_contexts)
        for e,name in enumerate(EVENTS):
            occurrences=sum(int(data["events"][0][:,e].sum()) for data in fit_contexts)
            fold_event_counts.append({"fold":fold_no,"event":name,"fit_target_occurrences":occurrences,
                                      "fit_origins":ntrain})
        valid_contexts={o:make_context_local(o,rhos) for o in valid_origins}
        rolling_metrics={}
        for o in valid_origins:
            rolling_metrics[o]=metric_x.scales_and_weights(train,meta,aggs,cal,prices,o)[:2]
        for model in ("baseline","holiday_only","snap_only","both"):
            params=fit_local(fit_contexts,model)
            fitted_by_fold[(fold_no,model)]=(rhos,params)
            coef_names=([f"beta:{label}" for label in labels] if model in ("snap_only","both") else []) + ([f"gamma:{e}" for e in EVENTS] if model in ("holiday_only","both") else [])
            coef_rows.extend({"fold":fold_no,"model":model,"parameter":k,"log_multiplier":float(v),"multiplier":float(np.exp(v)),"at_log_bound":bool(abs(v)>=.699)} for k,v in zip(coef_names,params))
            for idx,name in enumerate(coef_names):
                if name.startswith("gamma:"):
                    e=EVENTS.index(name.removeprefix("gamma:"))
                    ridge_rows.append({"fold":fold_no,"model":model,"parameter":name,
                                       "baseline_event_exposure_units":event_exposure[e],
                                       "ridge_penalty":GAMMA_RIDGE_MULTIPLIER*event_exposure[e],
                                       "gamma_ridge_multiplier":GAMMA_RIDGE_MULTIPLIER})
                elif name.startswith("beta:"):
                    ridge_rows.append({"fold":fold_no,"model":model,"parameter":name,
                                       "baseline_event_exposure_units":np.nan,"ridge_penalty":LAMBDA_BETA,
                                       "gamma_ridge_multiplier":np.nan})
            for o,d in valid_contexts.items():
                prediction=leaf_local(o,rhos,params,model,d["events"][0],d["snap"])
                # rolling metric weights/scales are intentionally derived at each origin.
                score=rolling_score(meta,train,prediction,o,aggs,*rolling_metrics[o])
                score_rows.append({"fold":fold_no,"origin_d":o,"model":model,"wrmsse":score,"n_train_origins":ntrain})
            print(f"fold {fold_no} {model}: fitted on {ntrain} origins",flush=True)
    scores=pd.DataFrame(score_rows); scores.to_csv(OUT/"rolling_validation_scores.csv",index=False)
    pd.DataFrame(coef_rows).to_csv(OUT/"calendar_coefficients_by_fold.csv",index=False)
    pd.DataFrame(ridge_rows).to_csv(OUT/"ridge_penalties_by_fold.csv",index=False)
    pd.DataFrame(fold_event_counts).to_csv(OUT/"event_occurrences_by_fold.csv",index=False)
    split_summary=scores.groupby(["fold","model"],as_index=False).wrmsse.mean()
    overall=scores.groupby("model",as_index=False).wrmsse.mean().rename(columns={"wrmsse":"mean_validation_wrmsse"})
    tie_order={"baseline":0,"holiday_only":1,"snap_only":2,"both":3}
    selected_model=str(overall.assign(tie_rank=overall.model.map(tie_order))
                       .sort_values(["mean_validation_wrmsse","tie_rank"]).iloc[0].model)
    split_summary.to_csv(OUT/"validation_scores_by_fold.csv",index=False)
    overall.to_csv(OUT/"validation_model_summary.csv",index=False)

    if args.smoke:
        # End-to-end real-data check for one small temporal split. The invariant
        # compares the zero-parameter correction with the existing rho profile.
        rho_part=rho_fold[(rho_fold.fold==1)&(rho_fold.partition=="state_category")]
        smoke_rhos=np.array([float(rho_part.set_index("group").loc[label,"rho"]) for label in labels])
        smoke_origin=ORIGINS[2]
        target_days=np.arange(smoke_origin+1,smoke_origin+29)
        smoke_events=event_matrix(cal.loc[target_days-1])
        smoke_snap=np.stack([cal.loc[target_days-1,state_snap_col[s]].to_numpy(float) for s in state_by_group])
        base=leaf_local(smoke_origin,smoke_rhos,np.zeros(0),"baseline",smoke_events,smoke_snap)
        zero_snap=leaf_local(smoke_origin,smoke_rhos,np.zeros(9),"snap_only",smoke_events,smoke_snap)
        zero_holiday=leaf_local(smoke_origin,smoke_rhos,np.zeros(4),"holiday_only",smoke_events,smoke_snap)
        zero_both=leaf_local(smoke_origin,smoke_rhos,np.zeros(13),"both",smoke_events,smoke_snap)
        if not np.allclose(zero_snap,base,rtol=0,atol=1e-10):
            raise AssertionError("smoke: zero SNAP coefficients do not reproduce rho baseline")
        if not np.allclose(zero_both,zero_holiday,rtol=0,atol=1e-10):
            raise AssertionError("smoke: zero coefficients do not match holiday-filtered baseline")
        if not np.isfinite(zero_both).all() or (zero_both < 0).any():
            raise AssertionError("smoke: invalid forecast values")
        # Check the analytic likelihood gradient in one SNAP and one event
        # coordinate; inject a target event into a copy so this works at an
        # early smoke origin that may contain no naturally occurring event.
        probe={**fit_contexts[0],"events":fit_contexts[0]["events"].copy()}
        probe["events"][:,0,0]=1.0
        probe["fit_mask"]=np.ones(28,dtype=bool)
        p=np.zeros(13); p[0]=.08; p[9]=-.06
        mu,jac=predict_local(probe,p,"both",True)
        safe=np.maximum(mu,1e-12); resid=1-probe["actual"]/safe
        probe_exposure=gamma_exposure([probe])
        lam=np.array([LAMBDA_BETA]*9+list(GAMMA_RIDGE_MULTIPLIER*probe_exposure))
        analytic=np.einsum("gh,ghp->p",resid,jac)+2*lam*p
        def probe_objective(v):
            mm=np.maximum(predict_local(probe,v,"both"),1e-12)
            return float(np.sum(mm-probe["actual"]*np.log(mm))+np.dot(lam,v*v))
        for j in (0,9):
            step=1e-5; plus=p.copy(); minus=p.copy(); plus[j]+=step; minus[j]-=step
            numeric=(probe_objective(plus)-probe_objective(minus))/(2*step)
            if not np.isclose(analytic[j],numeric,rtol=2e-4,atol=2e-3):
                raise AssertionError(f"smoke: analytic gradient mismatch for parameter {j}: {analytic[j]} vs {numeric}")
        probe_leaf=leaf_local(probe["origin"],smoke_rhos,p,"both",probe["events"][0],probe["snap"])
        leaf_group=np.stack([probe_leaf[rows].sum(axis=0) for rows in group_rows])
        preagg_error=float(np.max(np.abs(leaf_group-mu)))
        if not np.allclose(leaf_group,mu,rtol=1e-10,atol=1.5e-4):
            raise AssertionError(f"smoke: preaggregated fitted means differ from leaf-level forecast totals; max abs={np.max(np.abs(leaf_group-mu)):g}")
        final_release=np.where(first_release<=1913,first_release,10**9)
        baseline_ref=rho_mod.candidate_profile(
            {"history":rho_mod.weekday_history(train,1913,final_release)},codes,rhos,{})
        final_days=np.arange(1914,1942)
        final_events=event_matrix(cal.loc[final_days-1])
        final_snap=np.stack([cal.loc[final_days-1,state_snap_col[s]].to_numpy(float) for s in state_by_group])
        baseline_impl=leaf_local(1913,rhos,np.zeros(0),"baseline",final_events,final_snap)
        baseline_error=float(np.max(np.abs(baseline_impl-np.tile(baseline_ref,(1,4)))))
        if baseline_error>1e-5:
            raise AssertionError(f"smoke: existing rho baseline differs by {baseline_error:g}")
        overlap_days=np.flatnonzero(smoke_events.sum(axis=1)>1)
        if len(overlap_days):
            test_gamma=np.ones(4)
            factor=np.exp(smoke_events[overlap_days]@test_gamma)
            if not np.all(factor > np.exp(1)):
                raise AssertionError("smoke: overlapping event flags were not both applied")
        known_overlap=cal.loc[cal.date.eq("2013-05-05")]
        known_flags=event_matrix(known_overlap)
        if len(known_flags)!=1 or not (known_flags[0,1] and known_flags[0,2]):
            raise AssertionError("smoke: 2013-05-05 must activate OrthodoxEaster and Cinco De Mayo")
        print(f"real-data smoke passed: one validation origin, all four candidates; max rho-baseline difference={baseline_error:.3g}, max nonzero-parameter preaggregation difference={preagg_error:.3g}",flush=True)
        return

    # Final deployment fit through d_1913 uses the 04 final state×category rho table.
    rhos=np.array([float(rho_final.set_index("group").loc[label,"rho"]) for label in labels])
    final_origins=ORIGINS
    final_contexts=[make_context_local(o,rhos) for o in final_origins]
    final_event_exposure=gamma_exposure(final_contexts)
    final_params={model:fit_local(final_contexts,model) for model in ("baseline","holiday_only","snap_only","both")}
    sensitivity_rows=[]
    for multiplier in (0.1,0.0):
        for model in ("holiday_only","both"):
            params=fit_local(final_contexts,model,gamma_multiplier=multiplier)
            names=([f"gamma:{e}" for e in EVENTS] if model=="holiday_only" else
                   [f"gamma:{e}" for e in EVENTS])
            vals=params if model=="holiday_only" else params[9:13]
            for e,(name,value) in enumerate(zip(names,vals)):
                sensitivity_rows.append({"model":model,"gamma_ridge_multiplier":multiplier,
                                         "event":EVENTS[e],"log_multiplier":float(value),
                                         "multiplier":float(np.exp(value)),"at_log_bound":bool(abs(value)>=.699)})
    pd.DataFrame(sensitivity_rows).to_csv(OUT/"holiday_penalty_sensitivity.csv",index=False)
    future_days=np.arange(1914,1942)
    target_events=event_matrix(cal.loc[future_days-1])
    target_snap=np.stack([cal.loc[future_days-1,state_snap_col[s]].to_numpy(float) for s in state_by_group])
    prediction_tables=[]
    for model,params in final_params.items():
        pred=leaf_local(1913,rhos,params,model,target_events,target_snap)
        table=pd.DataFrame(pred,columns=[f"d_{d}" for d in future_days])
        table.insert(0,"id",meta.item_id.astype(str)+"_"+meta.store_id.astype(str)+"_validation")
        table.to_csv(OUT/f"forecast_{model}.csv",index=False)
        prediction_tables.append((model,pred))
    final_coef=[]
    final_unavailable={}
    for model,params in final_params.items():
        keys=([f"beta:{label}" for label in labels] if model in ("snap_only","both") else []) + ([f"gamma:{e}" for e in EVENTS] if model in ("holiday_only","both") else [])
        for ix,(key,value) in enumerate(zip(keys,params)):
            row={"model":model,"parameter":key,"log_multiplier":float(value),
                 "multiplier":float(np.exp(value)),"at_log_bound":bool(abs(value)>=.699)}
            if key.startswith("gamma:"):
                e=EVENTS.index(key.removeprefix("gamma:"))
                row.update({"baseline_event_exposure_units":float(final_event_exposure[e]),
                            "ridge_penalty":float(GAMMA_RIDGE_MULTIPLIER*final_event_exposure[e])})
            elif key.startswith("beta:"):
                row.update({"baseline_event_exposure_units":np.nan,"ridge_penalty":LAMBDA_BETA})
            final_coef.append(row)
        final_unavailable[model]=sum(unavailable_reference_counts(1913,rhos,first_release,cal,actual_snap_cols,
                                            group_rows,model in ("holiday_only","both")))
    pd.DataFrame(final_coef).to_csv(OUT/"final_calendar_coefficients.csv",index=False)

    # Invariants: zero coefficients exactly reproduce rho; nonnegative; row/day order; overlap composition.
    base=leaf_local(1913,rhos,np.zeros(0),"baseline",target_events,target_snap)
    zero_snap=leaf_local(1913,rhos,np.zeros(9),"snap_only",target_events,target_snap)
    zero_holiday=leaf_local(1913,rhos,np.zeros(4),"holiday_only",target_events,target_snap)
    zero_both=leaf_local(1913,rhos,np.zeros(13),"both",target_events,target_snap)
    if not np.allclose(base,zero_snap,rtol=0,atol=1e-10): raise AssertionError("zero SNAP coefficients differ from rho baseline")
    if not np.allclose(zero_holiday,zero_both,rtol=0,atol=1e-10): raise AssertionError("zero coefficients differ from holiday-filtered reference")
    final_release=np.where(first_release<=1913,first_release,10**9)
    baseline_ref=rho_mod.candidate_profile({"history":rho_mod.weekday_history(train,1913,final_release)},codes,rhos,{})
    baseline_error=float(np.max(np.abs(base-np.tile(baseline_ref,(1,4)))))
    if baseline_error>1e-5: raise AssertionError(f"calendar baseline differs from rho profile by {baseline_error:g}")
    if not np.allclose(base, np.tile(base[:, :7],(1,4)),rtol=0,atol=1e-10): raise AssertionError("baseline forecast weekday repetition changed")
    overlap_probe=np.array([1.,1.,0.,0.])
    if not np.isclose(np.exp(overlap_probe@np.ones(4)),np.exp(2)): raise AssertionError("overlap composition probe failed")
    known_overlap=cal.loc[cal.date.eq("2013-05-05")]
    known_flags=event_matrix(known_overlap)
    if len(known_flags)!=1 or not (known_flags[0,1] and known_flags[0,2]): raise AssertionError("2013-05-05 overlap flags missing")
    overlap=np.flatnonzero((target_events.sum(axis=1)>1))
    overlap_check={"days":future_days[overlap].tolist(),"event_counts":target_events[overlap].sum(axis=1).astype(int).tolist()}

    # Descriptive holdout WRMSSE after frozen forecasts are already written.
    actual=metric.load_test_actuals(meta)
    cal_train=cal.loc[cal.d_num.between(1,1913)].copy(); cal_test=cal.loc[cal.d_num.between(1914,1941)].copy()
    weights_weeks=set(cal_train.loc[cal_train.d_num.between(1886,1913),"wm_yr_wk"].astype(int))
    official_prices=metric.load_weight_window_prices(weights_weeks)
    holdout=[]
    for model,pred in prediction_tables:
        score,levels=metric.score_wrmsse(meta,train,actual,pred,cal_train,cal_test,official_prices)
        levels.to_csv(OUT/f"holdout_wrmsse_by_level_{model}.csv",index=False)
        holdout.append({"model":model,"holdout_wrmsse_descriptive":score})
    pd.DataFrame(holdout).to_csv(OUT/"holdout_wrmsse_descriptive.csv",index=False)

    # Event counts in all rolling targets and development targets.
    target_event_counts={}
    for event_i,name in enumerate(EVENTS):
        all_count=dev_count=0
        for idx,o in enumerate(ORIGINS):
            days=np.arange(o+1,o+29)
            count=int(event_matrix(cal.loc[days-1])[:,event_i].sum())
            all_count+=count
            if idx<32: dev_count+=count
        target_event_counts[name]={"all_rolling_occurrences":all_count,"development_occurrences":dev_count}
    summary={"x":8,"rho_partition":"state_category","folds":[{"train_origins":a,"validation_origins":b} for a,b in FOLDS],
             "models":["baseline","holiday_only","snap_only","both"],"events":list(EVENTS),
             "selected_model_by_validation":selected_model,
             "selected_forecast_path":f"forecast_{selected_model}.csv",
             "model_selection":{"validation_origins":int(scores.origin_d.nunique()),
                                 "rule":"lowest mean 12-level rolling WRMSSE; ties prefer baseline, holiday_only, snap_only, then both",
                                 "selection_completed_before_holdout_read":True},
             "event_target_occurrences":target_event_counts,"final_event_exposure_units":dict(zip(EVENTS,final_event_exposure.tolist())),
             "ridge":{"lambda_beta":LAMBDA_BETA,
                 "gamma_ridge_multiplier":GAMMA_RIDGE_MULTIPLIER,
                 "gamma_penalty_rule":"lambda_gamma,e = multiplier × sum of zero-coefficient both-model forecast units on that event's fit dates"},
             "parameterization":"target multiplier exp(beta_g * SNAP + sum(gamma_e * event_e)); prior SNAP history is divided by exp(beta_g * SNAP); holiday reference days are skipped and available weights renormalized.",
             "no_interactions_or_group_intercepts":True,"final_fit_origins":len(final_origins),"forecast_origin":1913,"forecast_days":[1914,1941],
             "holdout_use":"Descriptive WRMSSE computed after forecasts were frozen and written; holdout is not used for parameter fitting or model selection.",
             "fixed_ridge_and_bounds":{"lambda_beta":LAMBDA_BETA,"gamma_ridge_multiplier":GAMMA_RIDGE_MULTIPLIER,
                 "log_multiplier_bounds":[-0.7,0.7],"selection":"predeclared values, not tuned to validation or holdout",
                 "local_quadratic_shrinkage_approximation":"at multiplier 1, event log effects are shrunk to about one third of an unpenalized estimate when Poisson curvature is close to the baseline event exposure"},
             "holiday_penalty_sensitivity":"final training coefficients also saved for multipliers 0.1 and 0.0; descriptive only, not used for model selection or holdout tuning",
             "zero_reference_rule":"If a leaf weekday has no released non-holiday historical reference after filtering, its profile and forecast are zero; no fallback is applied.",
             "final_no_reference_leaf_weekday_counts_by_model":final_unavailable,
             "fold_event_fit_occurrences":fold_event_counts,
             "holdout_scores":holdout,"overlap_days":overlap_check,
             "historical_overlap_check":{"date":"2013-05-05","d_num":int(known_overlap.d_num.iloc[0]),
                                         "OrthodoxEaster":bool(known_flags[0,1]),"Cinco De Mayo":bool(known_flags[0,2])},
             "invariants":{"zero_snap_coefficients_exact_rho_baseline":True,"zero_event_coefficients_equal_holiday_filtered_reference":True,
                           "max_abs_difference_from_existing_rho_forecast":baseline_error,"nonnegative_forecasts":True,
                           "28_day_order_preserved":True,"overlap_composition_probe_passed":True,"future_data_leakage":False}}
    (OUT/"experiment_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")

    # Compact explanatory figures.
    fig,ax=plt.subplots(figsize=(8,4.5)); ordered=overall.sort_values("mean_validation_wrmsse")
    ax.bar(ordered.model,ordered.mean_validation_wrmsse,color=["#457b9d" if m=="baseline" else "#2a9d8f" for m in ordered.model]); ax.set_ylabel("Mean validation 12-level WRMSSE (lower is better)"); ax.set_title("Calendar correction on rolling validation folds"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"01_model_validation_scores.png",dpi=180); plt.close(fig)
    co=pd.DataFrame(final_coef); co=co[co.parameter.str.startswith("gamma:")]
    if len(co):
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model,part in co.groupby("model"):
            ax.plot(part.parameter.str.removeprefix("gamma:"),part.multiplier,marker="o",label=model)
        ax.axhline(1,color="#555",lw=1); ax.set_ylabel("Estimated holiday multiplier"); ax.set_title("Shrunk target-holiday multipliers"); ax.tick_params(axis="x",rotation=20); ax.legend(frameon=False); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"02_holiday_multipliers.png",dpi=180); plt.close(fig)
    hold=pd.DataFrame(holdout)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(hold.model,hold.holdout_wrmsse_descriptive,color="#d17b45"); ax.set_ylabel("Descriptive holdout 12-level WRMSSE"); ax.set_title("Frozen forecasts on existing holdout"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIGURES/"03_holdout_scores.png",dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
