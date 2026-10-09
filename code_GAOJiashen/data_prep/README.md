# M5 数据准备：运行说明

前四个脚本按编号运行；第 5 个脚本只画日历图，可在已有原始数据的情况下独立运行。它们读取 `data/m5-forecasting-accuracy/` 中的原始 CSV，**不会改写原始文件**。请在仓库根目录使用已安装依赖的 Python 环境运行，安装方法见根目录 README。

## 运行顺序

先进入项目目录：

```bash
cd MSDM5054-M5-GAOJiashen
```

1. 原始数据审计：检查销量、日期、价格的结构和异常值。

   ```bash
   python code_GAOJiashen/data_prep/01_audit_raw.py
   ```

   检查 `outputs/data_prep/reports/raw_audit.json` 的 `issues` 是否为空。若脚本报错，先检查报告，不要直接运行下一步。

2. 清洗与格式转换：保存矩阵、元数据、日历和价格；记录转换选择。

   ```bash
   python code_GAOJiashen/data_prep/02_prepare_clean.py
   ```

   检查 `data/processed/manifest.json`、`outputs/data_prep/reports/cleaning_summary.json`。矩阵为 `30490 × 1913`，行对应 `sales_metadata.csv`，列对应 `d_1` 至 `d_1913`。

3. 探索性汇总：只生成统计表，后续改图不用重新计算大矩阵。

   ```bash
   python code_GAOJiashen/data_prep/03_summarize_eda.py
   ```

4. 绘图：只读取第 3 步的表，生成七张统一样式的 PNG 和 SVG。以后如果只修改绘图代码，重跑这一步即可。

   ```bash
   python code_GAOJiashen/data_prep/04_plot_eda.py
   ```

   重点查看每日总销量、星期效应、品类/州的每序列走势、零销量比例、示例价格轨迹，以及第 7 张图中每个品类一条具体商品×门店的历史日销量。代表序列要求最近一年每个自然月均有正销量，并按最近一年日均销量和零销量比例接近品类中位数的程度确定；并列时按商品、门店和序列 ID 排序。图中每个面板使用独立纵轴，未来 28 天只标示预测区间，不填预测值。第 3 步另外生成 `example_series_selection.csv`（入选序列及统计量）和 `example_series_daily.csv`（三条序列最近 365 天的真实日销量）。原图中叠线难读的地方改用小多图和直接标注。设计依据：[GitHub data-visualization skill](https://github.com/anthropics/knowledge-work-plugins/blob/main/data/skills/data-visualization/SKILL.md)。图形仍只依赖现有 Matplotlib，无需再安装 Python 包。

5. 日历可视化：只读取原始 `calendar.csv`，不需要重跑前四步，也不读取任何未来销量。

   ```bash
   python code_GAOJiashen/data_prep/05_plot_calendar.py
   ```

   结果保存在 `outputs/data_prep/calendar/figures/`，每张图都有 PNG 和 SVG：

   - `01_training_calendar_overview`：训练期各月事件日比例、事件类型计数、三个州各月 SNAP 标记比例；这些是日期分布，不是销量影响。
   - `02_training_28_day_example`：2015-04-25 至 2015-05-22 的训练期实例，展示与预测期相近季节的 28 天事件和各州 SNAP 标记。
   - `03_forecast_28_days`：真正要预测的 `d_1914` 至 `d_1941`，即 2016-04-25 至 2016-05-22 的逐日事件和 SNAP 标记。两张 28 天图使用同一布局，便于比较。

## 文件保存位置

```text
5054minproject/
├─ data/
│  ├─ m5-forecasting-accuracy/       原始 CSV，保持不变
│  └─ processed/                     第 2 步生成
│     ├─ sales_matrix.npy            销量整数矩阵
│     ├─ sales_metadata.csv          矩阵逐行的商品与门店身份
│     ├─ calendar_clean.csv          日期、事件、SNAP，增加 d_num/has_event
│     ├─ sell_prices_clean.csv.gz    商品×门店×周价格，gzip 压缩
│     └─ manifest.json               数据规模、顺序和来源
├─ code_GAOJiashen/data_prep/                   本目录，五个脚本、共用样式与此说明
└─ outputs/data_prep/
   ├─ reports/
   │  ├─ raw_audit.json              第 1 步生成
   │  ├─ cleaning_summary.json       第 2 步生成
│  └─ eda_summary.json            第 3 步生成
   ├─ tables/                        第 3 步生成的 CSV 汇总表
   ├─ figures/                       第 4 步生成的 PNG 与 SVG 图表
   └─ calendar/figures/              第 5 步生成的日历 PNG 与 SVG
```

## 清洗边界与后续使用

- 销量中的 `0` 是有效观测，不能直接认作缺失值。
- 缺少某商品、门店、周的价格记录可能表示尚未开售；当前阶段不插补。
- 事件名空白保留为空。`has_event` 只是方便查看某日是否记录了事件。
- 不使用 `sales_train_evaluation.csv` 的后续真实销量，以免混入预测目标。
- 探索图使用完整训练历史；正式回测须先按时间切出验证期。任何滞后特征、均值或标准化参数，都只能从预测时点之前计算。
- 脚本会自动建立目标文件夹。若某一步中途失败，不应使用它可能留下的半成品；修复问题后重跑该步及后续步骤。
