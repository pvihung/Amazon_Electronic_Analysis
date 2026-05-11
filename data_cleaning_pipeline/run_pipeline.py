"""
run_pipeline.py — Orchestrator
Runs all three pipeline steps in order:
  1. ML filter (classify digital devices from metadata)
  2. BigQuery SQL (link reviews ↔ products, deduplicate)
  3. EDA data prep (clean, categorize, sentiment, upload)

Usage:
  python pipeline/run_pipeline.py                # all steps
  python pipeline/run_pipeline.py --steps 2 3   # specific steps only
"""

import argparse
import time

from pipeline import step1_ml_filter, step2_bq_queries, step3_eda_data

STEPS = {
    1: ("ML Filter",       step1_ml_filter.run),
    2: ("BQ SQL Queries",  step2_bq_queries.run),
    3: ("EDA Data Prep",   step3_eda_data.run),
}


def main():
    parser = argparse.ArgumentParser(description="Run the EDA data pipeline")
    parser.add_argument(
        "--steps", nargs="+", type=int,
        default=list(STEPS.keys()),
        help="Which steps to run (default: all). Example: --steps 2 3",
    )
    args = parser.parse_args()

    total_start = time.time()
    for step_num in sorted(args.steps):
        label, fn = STEPS[step_num]
        print(f"\n{'─' * 60}")
        print(f"  Running Step {step_num}: {label}")
        print(f"{'─' * 60}")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  ✓  Step {step_num} finished in {elapsed:.1f}s")

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete — total time: {total:.1f}s")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
