# quick-polars-bench

Local, fast evaluation harness for text→Polars systems using a public/alternative benchmark JSON that includes `gold_code`.

## What this is
- Loads `benchmark_final.json`
- Loads the referenced `data/*.parquet`
- Executes `gold_code` to compute the expected result
- Executes your predicted code (adapter hook)
- Counts **N_correct** and measures total time (**T_total_s**)

> This is for rapid iteration. It is not the official polarsbench.net evaluation runner.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data / benchmark files
- Put your benchmark at: `./benchmark_final.json` (or pass `--bench path`)
- Put parquet files under `./data/` as referenced by `"file_name"` in the benchmark.

We do **not** commit parquet files to git (often large). Keep them local or distribute them separately.

## Run (sanity check)
This runs the harness with `pred_code = gold_code` (should be 100% correct):
```bash
python scripts/quick_bench.py --use-gold-as-pred
```

## Integrate a participant repo / model
Edit `generate_pred_code(question, datasets)` in `scripts/quick_bench.py` to call the participant system and return a Python code string that sets:

```python
result = ...
```

Then run:
```bash
python scripts/quick_bench.py
```
