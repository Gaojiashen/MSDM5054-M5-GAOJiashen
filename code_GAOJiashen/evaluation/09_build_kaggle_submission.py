"""Build the M5 Accuracy late-submission CSV from the selected BLX model.

The validation rows come from the frozen d_1914..d_1941 experiment. For the
evaluation rows, we keep its selected rho, calendar coefficients, kappa and
mixture weights, but advance the forecast origin to d_1941. Thus the new
28-day history in sales_train_evaluation.csv informs the forecast without
looking at any d_1942..d_1969 labels.

Run with this project's Windows Python environment. Outputs are written to
outputs/submission/; source data and previous experiment outputs are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "m5-forecasting-accuracy"
PROCESSED = ROOT / "data" / "processed"
EXPERIMENT = ROOT / "outputs" / "evaluation"
OUT = ROOT / "outputs" / "submission"
ORIGIN = 1941
HORIZON = 28
LEAF_COUNT = 30490


def load_experiment_module():
    """Reuse the exact prediction functions from the selected experiment."""
    path = Path(__file__).with_name("08_borrow_information_wrmsse.py")
    spec = spec_from_file_location("m5_borrow_information", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_history(meta: pd.DataFrame) -> np.ndarray:
    """Append the newly public 28 days to our established training matrix.

    Read only d_1913 (an overlap check) and d_1914..d_1941 from the larger
    evaluation file. This avoids loading its earlier 1,912 columns twice.
    """
    old = np.load(PROCESSED / "sales_matrix.npy", mmap_mode="r")
    if old.shape != (LEAF_COUNT, 1913):
        raise ValueError(f"Unexpected processed training shape: {old.shape}")
    new_cols = [f"d_{d}" for d in range(1914, ORIGIN + 1)]
    public = pd.read_csv(
        RAW / "sales_train_evaluation.csv",
        usecols=["id", "d_1913", *new_cols],
    )
    expected_ids = meta["id"].str.replace("_validation$", "_evaluation", regex=True)
    if len(public) != LEAF_COUNT or not np.array_equal(public["id"], expected_ids):
        raise ValueError("Evaluation sales rows do not match processed metadata order")
    if not np.array_equal(public["d_1913"].to_numpy(), old[:, -1]):
        raise ValueError("Old and newly public sales disagree on d_1913")
    recent = public[new_cols].to_numpy()
    if not np.issubdtype(recent.dtype, np.integer) or (recent < 0).any():
        raise ValueError("Newly public sales must be nonnegative integers")
    if recent.max() > np.iinfo(np.uint16).max:
        raise ValueError("Newly public sales exceed the processed matrix dtype")
    history = np.empty((LEAF_COUNT, ORIGIN), dtype=np.uint16)
    history[:, :1913] = old
    history[:, 1913:] = recent
    return history


def evaluation_forecast(model, meta: pd.DataFrame, history: np.ndarray) -> np.ndarray:
    """Apply frozen model parameters at the new d_1941 forecast origin."""
    summary = json.loads(
        (EXPERIMENT / "borrow_information" / "experiment_summary.json").read_text(encoding="utf-8")
    )
    if summary["selected_candidate"] != "BLX":
        raise ValueError("Expected the selected BLX model")
    kappa = float(summary["selected_kappa"])
    if kappa <= 0:
        raise ValueError("Kappa must be positive")

    calendar = pd.read_csv(RAW / "calendar.csv")
    calendar["d_num"] = calendar["d"].str.removeprefix("d_").astype(int)
    if len(calendar) < ORIGIN + HORIZON or not np.array_equal(
        calendar["d_num"].to_numpy()[:ORIGIN + HORIZON],
        np.arange(1, ORIGIN + HORIZON + 1),
    ):
        raise ValueError("Calendar does not cover d_1..d_1969 in order")
    prices = pd.read_csv(RAW / "sell_prices.csv")
    metric = model.load_module(Path(__file__).with_name("02_select_x_wrmsse.py"), "submission_metric")
    release = metric.first_price_days(meta, prices, calendar)
    codes, labels, rows, _, item_codes, _, store_codes, stores, _, cats = model.setup(meta)

    rho_table = pd.read_csv(EXPERIMENT / "rho_selection" / "learned_rho_final.csv")
    rho_table = rho_table[rho_table["partition"] == "state_category"].set_index("group")
    rho = rho_table.loc[labels, "rho"].to_numpy(float)
    coef_table = pd.read_csv(EXPERIMENT / "calendar_effect" / "final_calendar_coefficients.csv")
    beta, gamma = model.params_for(coef_table.assign(fold=1), 1, labels)

    weight_table = pd.read_csv(EXPERIMENT / "borrow_information" / "final_mixture_weights.csv")
    bins = ("0", "1-2", ">=3")
    experts = ("B", "L_perp", "X")
    weights = (weight_table.pivot(index="reliability_bin", columns="expert", values="weight")
               .loc[list(bins), list(experts)].to_numpy(float))
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Saved mixture weights are invalid")
    if (weights < -1e-9).any():
        raise ValueError("Saved mixture weights are negative")

    q, prior, _, _, _, key_code = model.q_and_prior(
        history, meta, calendar, release, ORIGIN, beta, codes
    )
    baseline, local, cross_by_kappa, reliability = model.local_experts(
        history, meta, calendar, release, ORIGIN, rho, beta, gamma,
        rows, codes, q, prior, store_codes, len(stores), len(cats),
        key_code, item_codes, (kappa,),
    )
    cross = cross_by_kappa[kappa]
    result = baseline.copy()
    for bin_index in range(3):
        mask = reliability == bin_index
        result[mask] += (weights[bin_index, 1] * (local[mask] - baseline[mask])
                         + weights[bin_index, 2] * (cross[mask] - baseline[mask]))
    if result.shape != (LEAF_COUNT, HORIZON) or not np.isfinite(result).all():
        raise ValueError("Evaluation forecast has missing, infinite, or mis-sized values")
    if (result < -1e-9).any():
        raise ValueError("Evaluation forecast contains negative values")
    return np.maximum(result, 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_experiment_module()
    meta = pd.read_csv(PROCESSED / "sales_metadata.csv")
    if len(meta) != LEAF_COUNT or meta["id"].duplicated().any():
        raise ValueError("Invalid processed sales metadata")
    print("Reading public sales through d_1941...", flush=True)
    history = load_history(meta)
    print("Forecasting d_1942..d_1969 with the selected model...", flush=True)
    evaluation = evaluation_forecast(model, meta, history)

    # The template is the definitive order and layout for Kaggle. Validation
    # forecasts are the untouched frozen experiment, matched by exact ID.
    template = pd.read_csv(RAW / "sample_submission.csv", usecols=["id"])
    expected_columns = ["id", *[f"F{i}" for i in range(1, HORIZON + 1)]]
    template_header = pd.read_csv(RAW / "sample_submission.csv", nrows=0).columns.tolist()
    if template_header != expected_columns or len(template) != 2 * LEAF_COUNT:
        raise ValueError("Unexpected Kaggle sample_submission.csv layout")
    validation = pd.read_csv(EXPERIMENT / "borrow_information" / "forecast_borrow_information.csv")
    validation_cols = [f"d_{d}" for d in range(1914, 1942)]
    if validation.columns.tolist() != ["id", *validation_cols]:
        raise ValueError("Frozen validation forecast has unexpected columns")
    if validation["id"].duplicated().any() or len(validation) != LEAF_COUNT:
        raise ValueError("Frozen validation forecast has duplicate or missing IDs")
    validation = validation.set_index("id").reindex(template["id"].iloc[:LEAF_COUNT])
    validation_values = validation[validation_cols].to_numpy(float)
    if not np.isfinite(validation_values).all() or (validation_values < 0).any():
        raise ValueError("Frozen validation forecast has invalid values or unmatched IDs")

    evaluation_ids = meta["id"].str.replace("_validation$", "_evaluation", regex=True)
    evaluation_table = pd.DataFrame(evaluation, index=evaluation_ids.to_numpy())
    if evaluation_table.index.has_duplicates:
        raise ValueError("Evaluation forecast IDs are duplicated")
    evaluation_table = evaluation_table.reindex(template["id"].iloc[LEAF_COUNT:])
    evaluation_values = evaluation_table.to_numpy(float)
    if not np.isfinite(evaluation_values).all():
        raise ValueError("Evaluation forecast has IDs missing from Kaggle template")

    result = template.copy()
    result[[f"F{i}" for i in range(1, HORIZON + 1)]] = np.vstack(
        (validation_values, evaluation_values)
    )
    if result.columns.tolist() != expected_columns or result["id"].duplicated().any():
        raise ValueError("Final submission layout is invalid")
    destination = OUT / "m5_accuracy_submission.csv"
    result.to_csv(destination, index=False, float_format="%.8f")
    # Re-read the actual deliverable: this catches CSV serialization mistakes.
    saved = pd.read_csv(destination)
    numbers = saved.iloc[:, 1:].to_numpy(float)
    if (saved.columns.tolist() != expected_columns or len(saved) != 2 * LEAF_COUNT
            or not saved["id"].equals(template["id"])
            or not np.isfinite(numbers).all() or (numbers < 0).any()):
        raise AssertionError("Written submission failed the final checks")
    print(f"Ready: {destination} ({len(saved):,} rows; {HORIZON} forecast days)", flush=True)


if __name__ == "__main__":
    main()
