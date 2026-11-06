"""UC03 missing-weeks experiment: compare two runs (kit = no filling, fill = --fill-missing-weeks)
on the scoring labels, overall and on the pairs that have gaps in their training history.

    python tools/compare_uc03_fill.py <run_nofill> <run_fill> [warehouse] [sf_prefix]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(d: Path) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted(d.glob("*.parquet")))


def msle(df):
    y = np.clip(df["weekly_sales_true"].to_numpy(float), 0, None)
    p = df["weekly_sales_pred"].to_numpy(float)
    return float(np.mean((np.log1p(y) - np.log1p(p)) ** 2)) if len(df) else float("nan")


def main(run_a, run_b, warehouse="warehouse", sf="sf1"):
    wh = Path(warehouse)
    truth = read_table(wh / f"{sf}_labels.db" / "score_store_dept_labels")
    truth["date"] = pd.to_datetime(truth["date"]).dt.date
    weekly = read_table(wh / f"{sf}_features.db" / "uc03_weekly_sales")
    g = weekly.groupby(["store", "department"])["week"].agg(["min", "max", "count"])
    g["span"] = ((pd.to_datetime(g["max"]) - pd.to_datetime(g["min"])).dt.days // 7 + 1)
    g["missing"] = g["span"] - g["count"]
    gap_pairs = set(g[g["missing"] > 0].index)

    res = {"gap_pairs": len(gap_pairs), "missing_weeks": int(g["missing"].sum())}
    per_pair = {}
    for name, run in (("no_fill", run_a), ("fill", run_b)):
        p = pd.read_csv(Path(run) / "uc03" / "scoring_predictions.csv")
        p["date"] = pd.to_datetime(p["date"]).dt.date
        m = truth.merge(p, on=["store", "department", "date"], suffixes=("_true", "_pred"))
        key = list(zip(m["store"], m["department"]))
        is_gap = np.array([k in gap_pairs for k in key])
        res[name] = {"msle_all": round(msle(m), 4), "msle_gap_pairs": round(msle(m[is_gap]), 4),
                     "msle_other_pairs": round(msle(m[~is_gap]), 4),
                     "rows_gap_pairs": int(is_gap.sum()), "rows_all": len(m)}
        per_pair[name] = m[is_gap].groupby(["store", "department"]).apply(msle)
    both = pd.DataFrame(per_pair)
    res["gap_pairs_better_with_fill"] = int((both["fill"] < both["no_fill"]).sum())
    res["gap_pairs_worse_with_fill"] = int((both["fill"] > both["no_fill"]).sum())
    print(json.dumps(res, indent=2))
    out = Path(run_b) / "uc03" / "fill_comparison.json"
    out.write_text(json.dumps(res, indent=2))
    both.round(4).to_csv(Path(run_b) / "uc03" / "fill_comparison_gap_pairs.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
