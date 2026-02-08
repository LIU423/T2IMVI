# Comparison Experiments

## Direct VLM Scoring Baseline

Run direct idiom-image scoring with a single prompt:

```bash
python reliability_analysis/comparison/direct_vlm_scoring_baseline.py \
  --model qwen3-vl-2b
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
  --run-direct-baseline
```

Output:

`data/reliability_analysis/comparison/timing_experiment/<model_dir>/idiom_<id>/time_per_image_comparison_report.json`
