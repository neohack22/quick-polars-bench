import argparse
import json
import time
from pathlib import Path

import polars as pl


def load_one_dataset(path: Path, fmt: str) -> pl.DataFrame:
    fmt = (fmt or "").lower()
    if fmt == "parquet" or path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if fmt == "csv" or path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {fmt} for {path}")


def load_datasets(datasets: dict, base_dir: Path) -> dict:
    env = {}
    for var_name, spec in datasets.items():
        rel = spec["file_name"]
        fmt = spec.get("format", "")
        path = (base_dir / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file for {var_name}: {path}")
        env[var_name] = load_one_dataset(path, fmt)
    return env


def eval_code(code: str, env: dict) -> pl.DataFrame:
    """
    Expects executed code to set `result = <DataFrame or LazyFrame>`.
    """
    local_env = dict(env)
    safe_globals = {"__builtins__": {}}
    exec(code, safe_globals, local_env)

    if "result" not in local_env:
        raise RuntimeError("Code did not set `result`")

    res = local_env["result"]
    if isinstance(res, pl.LazyFrame):
        res = res.collect()
    if not isinstance(res, pl.DataFrame):
        raise TypeError(f"`result` must be a Polars DataFrame/LazyFrame, got {type(res)}")

    return res


def normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    # Normalize column order
    df = df.select(sorted(df.columns))
    # Normalize row order (best-effort)
    try:
        df = df.sort(df.columns)
    except Exception:
        pass
    return df


def approx_equal(a: pl.DataFrame, b: pl.DataFrame, float_tol: float) -> bool:
    if a.columns != b.columns or a.shape != b.shape:
        return False

    import numpy as np

    for c in a.columns:
        sa = a.get_column(c)
        sb = b.get_column(c)

        if sa.dtype in (pl.Float32, pl.Float64):
            da = sa.to_numpy()
            db = sb.to_numpy()
            if da.shape != db.shape:
                return False
            nan_mask = np.isnan(da) & np.isnan(db)
            diff = np.abs(da - db)
            diff[nan_mask] = 0.0
            if not np.all(diff <= float_tol):
                return False
        else:
            if not sa.series_equal(sb, null_equal=True):
                return False

    return True


# ---- ADAPTER (to integrate participant code) ----
def generate_pred_code(question: str, datasets: dict) -> str:
    """
    Replace this function to call a participant system.

    MUST return a Python code string that sets:
        result = <polars dataframe expression>

    Example output:
        result = nw_products.filter(pl.col("unit_price") > 100).select("product_name")
    """
    raise NotImplementedError("Implement generate_pred_code(...) to run a participant model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmark_final.json")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--float-tol", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--use-gold-as-pred", action="store_true", help="Sanity check: pred_code = gold_code")
    args = ap.parse_args()

    bench_path = Path(args.bench)
    base_dir = Path(args.base_dir)

    bench = json.loads(bench_path.read_text())
    questions = bench["questions"]
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    total = len(questions)
    correct = 0
    failures = []

    t0 = time.perf_counter()
    for q in questions:
        qid = q["id"]
        question = q["question"]
        datasets = q["datasets"]
        gold_code = q["gold_code"]

        try:
            env = {"pl": pl}
            env.update(load_datasets(datasets, base_dir=base_dir))

            gold = normalize_df(eval_code(gold_code, env))

            if args.use_gold_as_pred:
                pred_code = gold_code
            else:
                pred_code = generate_pred_code(question, datasets)

            pred = normalize_df(eval_code(pred_code, env))

            ok = approx_equal(pred, gold, float_tol=args.float_tol)
            correct += int(ok)
            if not ok:
                failures.append((qid, "mismatch"))
        except Exception as e:
            failures.append((qid, f"error: {type(e).__name__}: {e}"))

    t1 = time.perf_counter()
    elapsed = t1 - t0

    print("=== QUICK BENCH ===")
    print(f"bench: {bench_path}")
    print(f"questions: {total}")
    print(f"N_correct: {correct}/{total}")
    print(f"accuracy: {correct/total:.3f}" if total else "accuracy: n/a")
    print(f"T_total_s: {elapsed:.3f}")

    if failures:
        print("\nFailures:")
        for qid, msg in failures[:50]:
            print(f"- {qid}: {msg}")
        if len(failures) > 50:
            print(f"... +{len(failures) - 50} more")


if __name__ == "__main__":
    main()
