# Quadrant-Based Analysis

按 idiom 的 `transparency` 与 `imageability`（默认阈值 `0.5, 0.5`）划分四个象限，并在每个象限内比较：

- `S_pot`
- `S_fid`
- `Entity+Action Avg` (`entity_action_avg`)
- `(Fig+Lit)/2` (`fig_lit_avg`)

计算指标：

- `RBO` (Standard)
- `RBO` (Tie-Aware)
- `ICC`
- `Pearson r`
- `MAE`

## 在 T2IMVI 环境运行

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate T2IMVI
```

### 1) 全部 idiom

```bash
python reliability_analysis/quadrant_analysis/run_quadrant_analysis.py \
  --model qwen3_vl_2b_T2IMVI
```

### 2) 指定 idiom（CLI）

```bash
python reliability_analysis/quadrant_analysis/run_quadrant_analysis.py \
  --model qwen3_vl_2b_T2IMVI \
  --idiom-ids 1,15,54
```

### 3) 调整阈值（模块化可调）

```bash
python reliability_analysis/quadrant_analysis/run_quadrant_analysis.py \
  --model qwen3_vl_2b_T2IMVI \
  --transparency-threshold 0.5 \
  --imageability-threshold 0.5
```

结果默认保存到：`data/reliability_analysis/results/quadrant_analysis/`。
