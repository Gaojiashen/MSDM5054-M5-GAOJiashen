# Runnable companion for M5逐步实操手册.html.

from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "m5-forecasting-accuracy.zip"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
assert ZIP.exists(), f"找不到 {ZIP}"

with ZipFile(ZIP) as z:
    names = z.namelist()
    print(names)
    assert z.testzip() is None, "ZIP 内存在损坏文件"
    required = {"calendar.csv", "sell_prices.csv",
                "sales_train_validation.csv", "sample_submission.csv"}
    assert required.issubset(names)
    header = pd.read_csv(z.open("sales_train_validation.csv"), nrows=0)
    print("列数:", len(header.columns))
    print("前 8 列:", header.columns[:8].tolist())
    print("最后 3 列:", header.columns[-3:].tolist())

meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [f"d_{d}" for d in range(1830, 1914)]  # 84 天
types = {name: "float32" for name in day_cols}
with ZipFile(ZIP) as z:
    sales = pd.read_csv(z.open("sales_train_validation.csv"),
                        usecols=meta_cols + day_cols, dtype=types)
    calendar = pd.read_csv(z.open("calendar.csv"))

assert sales.shape == (30490, 90)
assert sales["id"].is_unique
assert (sales[day_cols].to_numpy() >= 0).all()
date_map = calendar.set_index("d")["date"]
print("sales shape:", sales.shape)
print("验证 B 日期:", date_map.loc["d_1886"], "→", date_map.loc["d_1913"])
print("门店 / 商品 / 品类:", sales.store_id.nunique(),
      sales.item_id.nunique(), sales.cat_id.nunique())

daily_total = sales[day_cols].sum(axis=0)
dates = pd.to_datetime(date_map.loc[day_cols].to_numpy())
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, daily_total.to_numpy(), color="#3158d6", lw=1.6)
ax.axvline(pd.Timestamp(date_map.loc["d_1858"]), color="#d68a2f",
           ls="--", label="验证 A 起点")
ax.axvline(pd.Timestamp(date_map.loc["d_1886"]), color="#c25462",
           ls="--", label="验证 B 起点")
ax.set(title="M5 总销量：最近 84 天", ylabel="售出件数")
ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(OUT / "01_daily_total.png", dpi=170); plt.close(fig)

def values(first, last):
    cols = [f"d_{d}" for d in range(first, last + 1)]
    return sales[cols].to_numpy(dtype=np.float32)

validation_b = values(1886, 1913)
print("验证 B 零销量比例:", round(float((validation_b == 0).mean()), 3))
print("验证 B 平均每日单品销量:", round(float(validation_b.mean()), 3))
print("大品类:", sales.cat_id.unique().tolist())

hist_a = values(1830, 1857)   # 在 d_1857 晚上能看到的最近 28 天
truth_a = values(1858, 1885)  # 未来 28 天，A 用于选择方案
hist_b = values(1858, 1885)   # 在 d_1885 晚上能看到的最近 28 天
truth_b = values(1886, 1913)  # 未来 28 天，B 用于最终检查

for arr in (hist_a, truth_a, hist_b, truth_b):
    assert arr.shape == (30490, 28)
print("切分完成：A/B 各有 30,490 × 28 个待预测值")

def make_baselines(history):
    assert history.shape[1] == 28
    last_week = history[:, -7:]
    b0 = np.tile(last_week, (1, 4))
    four_week_pattern = history.reshape(len(history), 4, 7).mean(axis=1)
    b1 = np.tile(four_week_pattern, (1, 4))
    b2 = np.repeat(history.mean(axis=1, keepdims=True), 28, axis=1)
    return {"B0_上周复制": b0, "B1_四周同星期": b1,
            "B2_四周总体均值": b2}

def rmse(y, pred):
    assert y.shape == pred.shape
    return float(np.sqrt(np.mean((y - pred) ** 2)))

models_a = make_baselines(hist_a)
models_b = make_baselines(hist_b)
for name in models_a:
    print(name, "A=", round(rmse(truth_a, models_a[name]), 4),
          "B=", round(rmse(truth_b, models_b[name]), 4))

candidate_weights = [0.0, 0.25, 0.50, 0.75, 1.0]
a_rows = []
for alpha in candidate_weights:
    pred = alpha * models_a["B0_上周复制"] + (1-alpha) * models_a["B1_四周同星期"]
    a_rows.append((alpha, rmse(truth_a, pred)))
print("A 调参结果:", a_rows)

chosen_alpha = min(a_rows, key=lambda row: row[1])[0]
chosen_b = (chosen_alpha * models_b["B0_上周复制"]
            + (1-chosen_alpha) * models_b["B1_四周同星期"])
print("A 选出的 alpha:", chosen_alpha)
print("B 上的混合 RMSE:", round(rmse(truth_b, chosen_b), 4))
print("B1 在 B 上的 RMSE:", round(rmse(truth_b, models_b["B1_四周同星期"]), 4))

pred0 = models_b["B0_上周复制"]
pred1 = models_b["B1_四周同星期"]
pred2 = models_b["B2_四周总体均值"]
e0 = (truth_b - pred0) ** 2
e1 = (truth_b - pred1) ** 2
e2 = (truth_b - pred2) ** 2

# 1. 第 1–28 天的误差是否越往后越大？
horizon = pd.DataFrame({
    "h": np.arange(1, 29),
    "B0_RMSE": np.sqrt(e0.mean(axis=0)),
    "B1_RMSE": np.sqrt(e1.mean(axis=0)),
    "B2_RMSE": np.sqrt(e2.mean(axis=0))})
horizon.to_csv(OUT / "02_horizon_rmse.csv", index=False)
ax = horizon.plot(x="h", y=["B0_RMSE", "B1_RMSE", "B2_RMSE"], figsize=(9,4))
ax.set(xlabel="未来第 h 天", ylabel="RMSE", title="预测步长误差")
ax.figure.tight_layout(); ax.figure.savefig(OUT / "02_horizon_rmse.png", dpi=170)
plt.close(ax.figure)

# 2. 门店与品类的 pooled RMSE：先按行算 MSE，再对组取均值开根号
for group in ["store_id", "cat_id"]:
    tmp = sales[[group]].copy()
    tmp["B0_MSE"] = e0.mean(axis=1)
    tmp["B1_MSE"] = e1.mean(axis=1)
    tmp["B2_MSE"] = e2.mean(axis=1)
    score = tmp.groupby(group)[["B0_MSE", "B1_MSE", "B2_MSE"]].mean().pow(0.5)
    score.to_csv(OUT / f"03_{group}_rmse.csv")
    print(group, "\n", score.round(3))

# 3. 寻找最难的底层序列，回看是否稀疏/异常
worst = sales[["id", "store_id", "cat_id"]].copy()
worst["B2_RMSE"] = np.sqrt(e2.mean(axis=1))
worst.nlargest(10, "B2_RMSE").to_csv(OUT / "04_worst_series.csv", index=False)

beta_rows = []
for beta in [0.0, 0.25, 0.50, 0.75, 1.0]:
    p = beta * models_a["B1_四周同星期"] + (1-beta) * models_a["B2_四周总体均值"]
    beta_rows.append((beta, rmse(truth_a, p)))
pd.DataFrame(beta_rows, columns=["beta", "A_RMSE"]).to_csv(
    OUT / "05_blend_search.csv", index=False)
chosen_beta = min(beta_rows, key=lambda row: row[1])[0]
pred_b3 = (chosen_beta * models_b["B1_四周同星期"]
           + (1-chosen_beta) * models_b["B2_四周总体均值"])
print("A 选出的 beta:", chosen_beta)
print("B3 在 B 上 RMSE:", round(rmse(truth_b, pred_b3), 4))
print("B2 在 B 上 RMSE:", round(rmse(truth_b, models_b["B2_四周总体均值"]), 4))
