"""Output file checks for both workloads (Part B6; M1 4.2, 6).

Runs at the end of run.py, or standalone on a finished run:
    python -m src.checks.outputs runs/<run_id>
"""
import json
import sys
from pathlib import Path

import pandas as pd

UC03_COLUMNS = ["store", "department", "date", "weekly_sales"]
UC08_COLUMNS = ["o_order_id", "trip_type"]
FILES = ["predictions.csv", "scoring_predictions.csv"]


def check_outputs(run_dir: Path, check) -> None:
    """`check(name, value, expected=None, ok=None, note="")` is RunRecorder.check or a printer."""
    for f in FILES:
        tag = f.removesuffix(".csv")
        p3 = run_dir / "uc03" / f
        if p3.exists():
            d = pd.read_csv(p3)
            check(f"output.uc03.{tag}.columns", list(d.columns), expected=UC03_COLUMNS)
            check(f"output.uc03.{tag}.duplicate_keys",
                  int(d.duplicated(["store", "department", "date"]).sum()), expected=0)
            check(f"output.uc03.{tag}.negative_forecasts", int((d["weekly_sales"] < 0).sum()), expected=0)
            check(f"output.uc03.{tag}.missing_values", int(d.isna().sum().sum()), expected=0)
        p8 = run_dir / "uc08" / f
        if p8.exists():
            d = pd.read_csv(p8)
            check(f"output.uc08.{tag}.columns", list(d.columns), expected=UC08_COLUMNS)
            check(f"output.uc08.{tag}.duplicate_order_ids", int(d["o_order_id"].duplicated().sum()),
                  expected=0)
            check(f"output.uc08.{tag}.missing_values", int(d.isna().sum().sum()), expected=0)


def _print_check(name, value, expected=None, ok=None, note=""):
    if ok is None and expected is not None:
        ok = value == expected
    status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"{status:4}  {name} = {value}" + (f" (expected {expected})" if expected is not None else ""))
    return status


if __name__ == "__main__":
    run = Path(sys.argv[1])
    check_outputs(run, _print_check)
    v = run / "validation.json"
    if v.exists():
        print(f"\nvalidation.json: {json.loads(v.read_text())['failed']} failed check(s)")
