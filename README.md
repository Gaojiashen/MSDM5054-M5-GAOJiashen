# MSDM5054 M5 forecasting: GAO Jiashen's code

This repository contains the data preparation, rolling evaluation, and forecasting code for GAO Jiashen's part of the MSDM5054 mini-project. The teammate's fixed-baseline and shared-tree code is maintained separately.

## Contents

- `code_GAOJiashen/data_prep/`: raw-data audit, cleaning, summaries, sales plots, and calendar plots.
- `code_GAOJiashen/evaluation/`: seasonal naive scoring, eight-week selection, grouped decay, trend test, calendar adjustment, sparse-demand diagnostics, information borrowing, and Kaggle submission generation.
- `code_GAOJiashen/M5_baseline_walkthrough.py`: an introductory baseline example.
- `requirements.txt`: Python dependencies.

The scripts write generated data and figures to `data/processed/` and `outputs/`. The original M5 data, generated results, and Python environment are intentionally excluded from the repository.

## Set up

Use Python 3.10 or a compatible version. From the repository root, install the dependencies into your own environment:

```bash
python -m pip install -r requirements.txt
```

Download the [M5 Forecasting - Accuracy data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place the original CSV files here:

```text
data/m5-forecasting-accuracy/
  calendar.csv
  sales_train_validation.csv
  sales_train_evaluation.csv
  sell_prices.csv
  sample_submission.csv
```

The introductory walkthrough also reads `data/m5-forecasting-accuracy.zip` if you want to run it. The later evaluation scripts use the extracted CSV files.

## Run order

First follow [data preparation](code_GAOJiashen/data_prep/README.md):

```bash
python code_GAOJiashen/data_prep/01_audit_raw.py
python code_GAOJiashen/data_prep/02_prepare_clean.py
python code_GAOJiashen/data_prep/03_summarize_eda.py
python code_GAOJiashen/data_prep/04_plot_eda.py
python code_GAOJiashen/data_prep/05_plot_calendar.py
```

Then follow the detailed [evaluation instructions](code_GAOJiashen/evaluation/README.md). The main numbered scripts are in `code_GAOJiashen/evaluation/`: `01` scores seasonal naive, `02` selects the averaging window, `03` plots that selection, `04` selects grouped decay, `05` tests trends, `06` tests calendar effects, `07` diagnoses sparse demand, `08` tests information borrowing, and `09` builds the Kaggle CSV. Run each script from the repository root, for example:

```bash
python code_GAOJiashen/evaluation/01_snaive_wrmsse.py
```

The complete rolling experiments are computationally intensive; the code is supplied for reproducibility and was not rerun as part of publishing this repository.
