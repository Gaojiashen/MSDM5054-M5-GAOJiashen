# M5 evaluation and averaging-window selection

## Selecting the averaging window `x`

Run `02_select_x_wrmsse.py` to compare same-weekday means over candidate
windows `x = {1, 2, 3, 4, 5, 6, 8, 12, 16}`. It jointly searches the exact
729 `(FOODS, HOBBIES, HOUSEHOLD)` vectors using the full 12-level WRMSSE.
The shared-`x` diagonal is also summarized as a baseline. Forecasts use only
information through each rolling origin and predict the next 28 days as one
block. Origins are `d_121 + 28k` through `d_1885` (64 total); the last target
day is `d_1913`. The script does not read `d_1914..d_1941`.

Windows are split in time: first 32 origins for selection, next 16 for
confirmation, final 16 for audit. Development selects the best joint category
vector and the best shared-`x` diagonal. Confirmation compares those fixed
candidates with `x=1` seasonal naive and adopts the one with the lowest mean
WRMSSE. Ties prefer `x=1`, then shared `x`, then the joint vector. Audit
evaluates the adopted choice alongside the development-selected joint vector,
shared baseline, and `x=1` seasonal naive, without further tuning. The
confirmation and audit periods never search the 729-vector grid.
For each origin, the first priced week observed on or before the origin is
treated as the release point. Before release the forecast is zero. After
release, only same-weekday observations at or after release are averaged,
using up to `x` available weeks; observed zero sales remain in the average.

Run from the repository root with the Python environment described in the top-level README:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/02_select_x_wrmsse.py
```

Outputs are written to `outputs/evaluation/x_selection/`:

- `selection_summary.json`: development winners, confirmation gate scores and
  adopted choice, plus fixed comparison scores on each split.
- `development_vector_means.csv`: all 729 vectors ranked by mean development
  WRMSSE. This is the main file for reviewing the selection.
- `rolling_origin_scores.csv`: per-origin WRMSSE for every vector on
  development; confirmation contains only the fixed joint, shared, and `x=1`
  candidates, while audit contains the adopted choice, joint candidate, shared
  baseline, and `x=1` seasonal naive.
- `selected_vector_split_scores.csv`: split summaries for the candidates
  compared at each stage.
- `origin_metric_coverage.csv`: zero-scale node counts and the dollar-weight
  share excluded from each rolling origin and hierarchy level.

## Visualizing the complete `x` selection

After `02_select_x_wrmsse.py` has completed, run:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/03_visualize_x_selection.py
```

The script writes **six SVG/PNG figure pairs** under
`outputs/evaluation/x_selection/figures/`; it no longer generates an HTML
report. The figures present the nine shared-`x` candidates, all 729
three-category vectors as nine heatmaps, the development/confirmation/audit
decision, a window-by-window comparison of `x=5` with `x=8`, and one previously
selected HOBBIES product-store example. Exact values are retained in the
existing CSV files; paired values are saved to `paired_x5_x8_scores.csv`.

The original search evaluated `x=5` on development but not confirmation.
On the first visualization run, this script scores shared `x=5` on the same
16 confirmation origins and caches them in `shared_x5_confirmation_scores.csv`;
later visualization runs reuse that file. This is a **post-selection diagnostic**.
It does not change `selection_summary.json` or use the audit window to choose a
new x. The x=5 comparison explains the selection process; the final adopted
method remains shared x=8. No new dependency is required beyond
`requirements.txt`.

The full score is computed efficiently by separately scoring hierarchy levels
whose nodes each belong to one category, then jointly scoring the total, state,
and store levels where forecasts from multiple categories combine.

## Selecting grouped recency weights with `x=8` fixed

Run `04_select_rho_wrmsse.py` to tune `rho` directly. For each of the eight
past same-weekday observations, `w_k` is proportional to `rho^k`, with `k=0`
for the newest observation. Weights are normalized over values available after
that series’ first priced day; observed zero sales count. `rho=0` is sNaive and
`rho=1` is equal-weight x=8. The seven-day profile is repeated four times.

The script jointly scores each candidate against the full 12-level WRMSSE and
compares four increasingly flexible groupings: global, state, state × category,
and store × category. Coordinate descent uses candidates
`{0, .25, .5, .65, .75, .8, .85, .9, .95, 1}`. Four rolling folds train on the
first 32, 40, 48, then 56 origins and validate on the next eight origins.
Partition choice compares every candidate with the lowest-mean partition using
paired origin losses. Within each validation fold, a moving-block bootstrap
resamples contiguous four-origin blocks. The simplest partition is selected
when its mean loss from the best-mean model is no greater than its paired
bootstrap standard error. The block intervals describe uncertainty and
are not significance tests. Coordinate descent makes two grid passes and may
reach a local optimum rather than the global best rho vector. The selected
grouping is refit on all 64 origins; held-out actuals are read only after the
model and forecast are frozen.

Run the full selection from the repository root:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/04_select_rho_wrmsse.py
```

If the four rolling folds have already completed, resume from their saved
validation scores and refit only the final selected partition:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/04_select_rho_wrmsse.py --resume-selection
```

Outputs are written to `outputs/evaluation/rho_selection/`: the selection
summary, fold scores, learned fold and final rho tables, coverage, forecast,
final official WRMSSE score and level breakdown, and PNG/SVG figures for the
global rho grid, partition comparison, paired improvements, weight profiles,
final group rho values, and final held-out total forecast. The held-out score
is computed only after the selected model has been refit and its forecast saved.
The sNaive reference score is read from the existing sNaive summary. The
forecast CSV retains `_validation` IDs.

### Early rolling origins with an undefined RMSSE scale

At an early origin, a newly active series can have sales revenue in the last
28 days but no nonzero first difference after its first sale. Its RMSSE
denominator is zero, so the official expression is undefined for that node.
For the **rolling x-selection score only**, the script excludes every zero-scale
node and renormalizes the remaining dollar weights within its hierarchy level.
This is done before scoring candidates, so every x-vector uses exactly the
same nodes and weights at a given origin. `origin_metric_coverage.csv` records
the number of excluded nodes and their original revenue share for every origin
and level; `selection_summary.json` records the largest excluded share. When
that share is zero, the rolling score equals the exact WRMSSE. If it is large,
interpret the early-window comparison cautiously. The separate sNaive final
evaluation script retains its original strict official metric.

`01_snaive_wrmsse.py` evaluates the official M5 Seasonal Naive forecast over
the 28-day validation horizon.

## Data split

- Training and all forecast inputs: `sales_train_validation.csv`, `d_1` through
  `d_1913`.
- Held-out outcomes: `sales_train_evaluation.csv`, `d_1914` through `d_1941`.
- The forecast is generated first by repeating `d_1907` through `d_1913` four
  times. The held-out file is loaded afterward and aligned by `item_id` and
  `store_id`; the `_validation` and `_evaluation` ID suffixes are checked after
  normalization.
- WRMSSE dollar weights use only training sales on `d_1886` through `d_1913`
  and the corresponding weekly prices. A positive sale with no matching price
  raises an error.

## Metric

The script aggregates leaf series into the 12 M5 hierarchy levels (42,840
nodes total), computes weighted RMSSE separately within each level, and averages
the 12 level scores with equal weight. Each node's weight is its sales revenue
over the final 28 training days divided by total revenue for its level. The
RMSSE scale is the mean squared first difference of the aggregate training
series, starting with differences after its first nonzero observation. A
zero-scale node with zero forecast error contributes zero; a positive-weight
zero-scale node with nonzero error stops the calculation rather than emitting
an undefined score. Zero-scale counts are retained in the summary.

## Outputs

Running the script writes to `outputs/evaluation/snaive/`:

- `snaive_predictions.csv`: 30,490 leaf forecasts for `d_1914`–`d_1941`,
  retaining their validation IDs. The evaluation file only supplies test labels.
  This is a local prediction table, not the two-stage Kaggle submission template.
- `snaive_wrmsse_by_level.csv`: node counts, zero-scale counts, and the score
  for each of the 12 levels.
- `snaive_wrmsse_summary.json`: overall WRMSSE and split/metric metadata.
- `daily_total_comparison.csv`: each test day's total forecast and actual units.
- `example_series_comparison.csv`: each test day's forecast and actual units for three product-store examples.
- `01_total_forecast.png` and `.svg`: total daily forecast against actuals.
- `02_example_series_forecast.png` and `.svg`: one product-store series from
  each of the three categories, with forecast and actual values.
- `03_wrmsse_by_level.png` and `.svg`: score by hierarchy level.

From the repository root, run:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/01_snaive_wrmsse.py
```

Required packages are listed in the project-level `requirements.txt`.

## Comparing constant, linear, and damped grouped trends

Run `05_grouped_trend_wrmsse.py` to test whether the selected state × category
rho baseline benefits from a changing weekly level. It loads the nine frozen
rho values from `rho_selection/learned_rho_final.csv`. The constant candidate
is the same release-aware seven-day profile repeated four times. For trend
candidates, it sums daily sales within each of the nine groups, computes eight
weekly means, and estimates a rho-weighted WLS slope. The group level ratio
scales each leaf's existing seven-day profile, preserving its weekday shape.
Negative projected group levels are clipped to zero before ratios are applied;
the experiment summary records clipping counts and affected final groups.

The global `phi` grid is selected on the first 32 rolling origins; its fixed
choice is compared with constant and linear models on the next 16 confirmation
origins; the best confirmation model is then reported on the final 16 audit
origins. If damped wins confirmation, its `phi` is refit on the first 48 origins
to create the frozen d_1914–d_1941 forecast. Held-out labels are read only after
that forecast is saved, for a strict official 12-level WRMSSE report. Use
`--skip-final-holdout` to stop after rolling scores and forecast generation.
After a complete run, use `--reuse-rolling-scores` to rebuild final-origin
tables, figure, and forecast from the saved rolling candidate scores without
recomputing all rolling origins.
The earliest d_121 origin has 56 prior days. Outputs are under
`outputs/evaluation/grouped_trend/`, including per-origin candidate scores,
split summaries, two PNG figures, forecast, and experiment metadata.
The final-origin outputs also include `final_group_trend_levels.csv` (eight
weighted history weeks, WLS parameters, and four future levels for all three
models), `final_group_weekday_effects.csv` (rho-weighted within-week residuals),
and `final_group_trend_panels.png` (one panel per state × category group with
the confirmation-selected model marked). Weekday effects sum to zero within
each group by construction and are descriptive; the leaf weekday profile used
for forecasting remains the frozen rho baseline.

Run from the repository root:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/05_grouped_trend_wrmsse.py
```

## Testing calendar multipliers on the grouped-rho baseline

Run `06_calendar_effect_wrmsse.py` to compare four fixed candidates on top of the selected state × category `rho` profile: the original baseline, holiday only, SNAP only, and both. For holiday candidates, all event-marked dates in the preceding eight weeks are removed from each same-weekday reference average and the remaining weights are renormalized. The four forecast-horizon events (`Pesach End`, `OrthodoxEaster`, `Cinco De Mayo`, and `Mother's day`) receive shared log multipliers. For SNAP candidates, historical SNAP sales are divided by the state × category SNAP multiplier before averaging; the denominator remains the sum of available historical `rho` weights. The target-day calendar multiplier is `exp(beta × SNAP + sum(gamma × event))`, so both event flags apply on an overlapping date. Other historical holiday names are only filtered from reference windows; they do not receive fitted coefficients. Target-day observations with a different holiday are omitted from coefficient fitting, while the full 28 days remain in WRMSSE scoring.

The 9 SNAP and 4 event log multipliers are fitted with a group-total Poisson negative log likelihood and log multipliers bounded to `[-0.7, 0.7]`. SNAP uses the fixed ridge penalty `lambda_beta=2`. For each event, the ridge penalty is `kappa × baseline forecast units on that event's fitting dates`, with predeclared `kappa=1`. Under a local quadratic approximation where the Poisson curvature matches baseline event exposure, this shrinks an unpenalized event log multiplier to about one third. The script also fits final event coefficients with `kappa=0.1` and `kappa=0` as descriptive sensitivity checks; those values do not select a model or use holdout scores. These settings are fixed in advance, not selected as optimal. Holiday effects have sparse history: each of the four events appears four times in the 64 existing rolling target windows and twice in the first 32 development origins. No event × SNAP interactions or group calibration intercepts are fitted.

The four temporal folds match the rho experiment (32/40/48/56 training origins, then eight validation origins). Each fold uses its own rho values from `learned_rho_by_fold.csv`; origin-specific WRMSSE scales and weights use only data available at that origin. The final candidate fits use the final rho table and rolling outcomes through `d_1913` to generate `d_1914`–`d_1941` forecasts. The existing holdout is then scored descriptively after forecasts are frozen; it does not choose parameters or a model.

Outputs are written under `outputs/evaluation/calendar_effect/`: rolling scores by origin, fold summaries, fitted coefficients and ridge penalties, final coefficients, holiday penalty sensitivity coefficients, four forecast CSVs, descriptive holdout scores by hierarchy level, a JSON summary, and three PNG figures. Run `06_calendar_effect_wrmsse.py --smoke` for a small real-data run over one validation origin and the model invariants. This writes smoke outputs in the same directory and can be overwritten by the full run.

The selected candidate is the one with the lowest mean WRMSSE across the 32 rolling validation origins; ties prefer the baseline, then holiday only, SNAP only, and both. The chosen name and forecast path are recorded in `experiment_summary.json` before holdout scoring. All four holdout scores remain descriptive comparisons.

### Diagnosing zero-sales errors in the selected calendar model

Run `07_sparse_zero_diagnostic.py` to inspect the selected `both` calendar+SNAP
candidate on the same 32 saved rolling-validation origins. It reads each fold's
saved state × category rho values and calendar coefficients, then reconstructs
leaf forecasts with the same `leaf_local` calculation used by script 06. The
comparison is against the matching rho-only forecast at each origin. It does
not fit parameters or read the d_1914–d_1941 holdout.

Series cohorts use only information available through each origin: no-history
series, released series with fewer than 28 available days, and mature series
grouped by zero-sales share in the preceding 56 calendar days. Days before the
first priced release are excluded from that share. The outputs report the
fraction of actual-zero days forecast above 0, 0.5, and 1 unit, plus zero-day
squared-loss share and nonzero-day loss. Nonzero-day summaries also report the
fraction forecast exactly zero, the underprediction fraction, and mean actual
and forecast units, both per origin and pooled. The figure's right panel uses
each cohort's zero-day squared loss divided by all eligible leaf/day squared
loss for that model; the CSV retains the within-cohort zero-day loss share. The
CSVs also include `always_zero_squared_loss` (the sum of actual sales squared),
which is the loss of forecasting zero for every eligible cell. Compare it with
the calendar model's total squared loss for the high-sparsity cohort.
The zero threshold is a descriptive positive-forecast rate; thresholds and
squared loss are more useful for judging
the size and practical effect of zero-day overforecasting.

For mature series with at least 80% zero sales in the release-aware past 56 days,
the script also checks unconditional calibration across all 28 target days.
It saves fixed forecast-value bands, actual and predicted mean sales, and the
frequency of sales at or above three units. Three units is an illustrative
high-sales threshold, not a fitted parameter. The top predicted tenth is
defined from predictions within each validation fold; its observed high-sales
rate is compared with the cohort's overall rate.

From the repository root, run:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/07_sparse_zero_diagnostic.py
```

Outputs are written to `outputs/evaluation/sparse_zero_diagnostic/`:
`pooled_overall.csv`, `pooled_cohort.csv`, per-origin overall and cohort CSVs,
`sparse_unconditional_per_origin.csv`, `sparse_unconditional_pooled.csv`,
the fixed-band CSVs, `zero_day_diagnostic.png`, and
`sparse_forecast_calibration.png`. The per-origin tables show whether the
observed errors persist across folds and forecast origins.

## Borrowing local weekday and cross-store information

`08_borrow_information_wrmsse.py` adds two experts to the selected calendar
plus SNAP model from 06. `L_perp` estimates a de-weekdayed level from released
sales in the past 56 days, excluding the target weekday and every historically
event-marked day. It removes the fitted SNAP and store × category weekday
effects, applies the inherited group recency decay, then restores the target
weekday multiplier and target-day SNAP/event factors. If no usable other-weekday
observations remain, `L_perp` falls back to B. The store × category weekday
multiplier is estimated from the prior 364 days after removing SNAP and
excluding event-marked dates; it is normalized to mean one and shrunk with 28
pseudo-days.

`X` uses the same item's other stores. It forms a recency-weighted daily rate
from released sales in the preceding 56 days after removing fitted SNAP,
event, and weekday factors. Its conversion uses
`(H_own + kappa * p0) / (H_all_stores + kappa)`, where `H` is past-only
de-calendarized item-store volume and `p0` is the past-only store share for its
category. The share is shrunk with the fixed grid `kappa={5,20,80}`. The
share odds are capped at 20 to keep near-exclusive store histories from
creating extreme forecasts. If an expert lacks its required observations, it
falls back to B for that item; this keeps the mixture affine and preserves the
baseline forecast in unsupported cases.

The mixture is shared across all goods and distinguishes three reliability
bins for each target weekday: zero, one or two, and at least three positive
observations among the preceding eight matching weekdays. Each bin reports
three weights for B, `L_perp`, and X, summing to one. There are nine reported
weights and six free values. The script compares B, B+L, B+X, and B+L+X. It
fits the simplex weights on the training origins of each temporal fold by
minimizing the true 12-level rolling WRMSSE, then selects the expert set and
`kappa` by mean score on the next eight origins. Ties prefer smaller `kappa`
and then the simpler expert set. The final selected mixture is refit on all 64
historical origins using frozen final rho and calendar coefficients before
the d_1914–d_1941 forecast is written. Holdout actuals are read only afterward
for a descriptive score.

Target weekday is taken from each actual forecast date. Item-level history
sharing uses item factor codes and grouped sums. Origin-specific scale/weight
arrays and per-node quadratic Gram terms are cached so each optimizer step
evaluates WRMSSE without repeating sparse hierarchy aggregation. The script
checks all 32 B validation scores against 06 within `2e-5` and checks the final
B forecast against `forecast_both.csv`. Its outputs are under
`outputs/evaluation/borrow_information/`: rolling candidate scores, fold
weights and selections, baseline parity scores, nine final weights, the
forecast, a descriptive holdout score, a JSON summary, and two PNG figures.
Run it from the project environment only when ready for the full 30,490-series
experiment; it processes one fold at a time and retains compact float32 node
Gram coefficients rather than complete leaf forecasts to bound memory use.

From the repository root:

```bash
cd MSDM5054-M5-GAOJiashen
python code_GAOJiashen/evaluation/08_borrow_information_wrmsse.py
```
