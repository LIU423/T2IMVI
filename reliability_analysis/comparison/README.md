# Comparison Experiments

## Direct VLM Scoring Baseline

Run direct idiom-image scoring with a single prompt:

```bash
python reliability_analysis/comparison/direct_vlm_scoring_baseline.py \
  --model qwen3-vl-2b \
  --batch-size 4
```

Prompt file can be:

- `.txt` template (supports `<idiom>` or `{idiom}` placeholder)
- `.py` file with `build_prompt(idiom)` or `PROMPT_TEMPLATE`

Output directory:

`data/reliability_analysis/comparison/direct_vlm_scoring_baseline/<model_dir>/idiom_<id>/image_<id>/`

Each image folder contains:

- `total_score.json` (compatible with `reliability_analysis/data_loader.py`)
- `direct_vlm_baseline.json` (direct output details)
- `raw_response.txt` (verbatim model text)

Run-level timing summary:

- `timing_direct_vlm_baseline.json` (average seconds per image, includes per-idiom stats)
- `checkpoint_direct_vlm_baseline.json` (resume metadata for completed/failed items)

Resume and progress:

- By default, existing outputs are skipped and the run resumes from checkpoint/files.
- Use `--overwrite` to force re-run all items from scratch.
- Progress is shown with `tqdm` when available (otherwise periodic log updates).

Model dirs:

- `qwen3-vl-2b` -> `qwen3_vl_2b`
- `qwen3-vl-30b-a3b-instruct` -> `qwen3_vl_30b_a3b_instruct`

Registered reliability model keys:

- `comparison_direct_vlm_baseline_qwen3_vl_2b`
- `comparison_direct_vlm_baseline_qwen3_vl_30b_a3b_instruct`

## Time Comparison Experiment

Compare per-image average time between pipeline and direct baseline:

```bash
python reliability_analysis/comparison/time_per_image_comparison.py \
  --model qwen3-vl-30b-a3b-instruct \
  --idiom-id 43 \
  --batch-size 1 \
  --run-direct-baseline
```

To force direct-baseline rerun instead of resume:

```bash
python reliability_analysis/comparison/time_per_image_comparison.py \
  --model qwen3-vl-30b-a3b-instruct \
  --idiom-id 43 \
  --run-direct-baseline \
  --overwrite-direct-baseline
```

Also supports batch idiom IDs:

```bash
python reliability_analysis/comparison/time_per_image_comparison.py \
  --model qwen3-vl-30b-a3b-instruct \
  --idiom-ids 489 47 553 \
  --run-direct-baseline
```

Output:

`data/reliability_analysis/comparison/timing_experiment/<model_dir>/idiom_<id>/time_per_image_comparison_report.json`

## Quadrant-Style Strategy vs T2IMVI Comparison

Compare

- `data/reliability_analysis/comparison/<strategy>/<model_dir>`
- `data/output/IRFL/<model_dir>_T2IMVI`

with the same quadrant-style aggregation idea used in
`reliability_analysis/quadrant_analysis/run_quadrant_analysis.py`:

```bash
python reliability_analysis/comparison/quadrant_strategy_comparison.py \
  --strategy direct_vlm_scoring_baseline \
  --model-dir qwen3_vl_2b
```

该比较脚本支持“1 对 4”模式：

- comparison（direct baseline）使用 1 个分数（默认 `figurative_score`）
- T2IMVI 使用 4 个分数（默认 `s_pot,s_fid,entity_action_avg,fig_lit_avg`）

对 direct baseline，comparison 侧建议使用 `figurative_score`（即 direct 总分）：

```bash
--comparison-score-field figurative_score
```

T2IMVI 侧四个分数可配置：

```bash
--t2imvi-score-fields s_pot,s_fid,entity_action_avg,fig_lit_avg
```

默认终端输出为直观并排分数（`comparison / T2IMVI`），不是差值。
如需额外显示差值，再加：

```bash
--show-delta
```

新增 RBO 指标（可配置阈值）：

- 先把 IRFL 人类总分 `<= 阈值` 置为 `0`
- 再按 tie-aware 逻辑计算 RBO（输出列 `RBO_tie<=thr(c/t)`）

默认阈值是 `10`，可改为 `5`、`15` 等：

```bash
--low-score-zero-threshold 10
```

Example for a specific idiom subset:

```bash
python reliability_analysis/comparison/quadrant_strategy_comparison.py \
  --strategy direct_vlm_scoring_baseline \
  --model-dir qwen3_vl_2b \
  --comparison-score-field figurative_score \
  --t2imvi-score-fields s_pot,s_fid,entity_action_avg,fig_lit_avg \
  --idiom-ids 1 43 489
```

Output JSON is saved to:

`data/reliability_analysis/results/comparison_quadrant_analysis/`
