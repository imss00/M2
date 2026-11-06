"""UC03 with the kit's library versions (python 3.9, statsmodels 0.12.2, pandas 1.5.2), done
exactly like UseCase03.train / serve on our weekly table (same aggregation rule as the kit).
Reports how many seasonal fits fail (the kit run would stop there), the MSLE on the scoring labels
and the difference to our predictions.

    <py39-env>/bin/python tools/kit_versions_uc03.py <warehouse> <run_dir> <score_dir>

Runs on Python 3.9, so it does not import src/.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def read_table(d: Path) -> pd.DataFrame:
    return pd.concat(pd.read_parquet(f) for f in sorted(d.glob("*.parquet")))


def main(warehouse, run_dir, score_dir):
    wh, run = Path(warehouse), Path(run_dir)
    weekly = read_table(wh / "sf1_features.db" / "uc03_weekly_sales")
    weekly["week"] = pd.to_datetime(weekly["week"])
    t = time.time()
    models, failed = {}, []
    for (store, dept), g in weekly.groupby(["store", "department"]):
        ts = g.set_index("week")["weekly_sales"].sort_index()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                models[(store, dept)] = (ExponentialSmoothing(ts, seasonal="add", seasonal_periods=52).fit(),
                                         ts.index.max())
            except Exception as e:  # noqa: BLE001 - the kit would stop here
                failed.append([int(store), dept, len(ts), type(e).__name__])
    train_s = time.time() - t

    req = pd.read_csv(Path(score_dir) / "store_dept.csv")
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in req.itertuples(index=False):
            if (r.store, r.department) not in models:
                continue
            m, ts_max = models[(r.store, r.department)]
            fc = np.clip(np.asarray(m.forecast(int(r.periods))), 0.0, None)
            start = pd.date_range(ts_max, periods=2)[1]
            idx = pd.date_range(start, periods=int(r.periods), freq="W-FRI")
            rows.append(pd.DataFrame({"store": r.store, "department": r.department,
                                      "date": idx.date, "weekly_sales": fc}))
    kit = pd.concat(rows)
    kit.to_csv(run / "uc03" / "kit_versions_scoring_predictions.csv", index=False)

    truth = pd.read_csv(Path(score_dir) / "store_dept_labels.csv")
    truth["date"] = pd.to_datetime(truth["date"]).dt.date
    ours = pd.read_csv(run / "uc03" / "scoring_predictions.csv")
    ours["date"] = pd.to_datetime(ours["date"]).dt.date
    key = ["store", "department", "date"]
    m = truth.merge(kit, on=key, suffixes=("", "_kit")).merge(ours, on=key, suffixes=("", "_ours"))
    y = np.log1p(np.clip(m["weekly_sales"], 0, None))
    res = {"statsmodels": statsmodels.__version__, "pairs_trained": len(models),
           "seasonal_fit_failures": len(failed), "failures": failed[:10],
           "train_seconds": round(train_s, 1), "scored_rows": len(m),
           "msle_kit_versions": round(float(np.mean((y - np.log1p(m["weekly_sales_kit"])) ** 2)), 4),
           "msle_ours": round(float(np.mean((y - np.log1p(m["weekly_sales_ours"])) ** 2)), 4),
           "max_abs_diff_vs_ours": round(float((m["weekly_sales_kit"] - m["weekly_sales_ours"]).abs().max()), 3),
           "median_abs_diff_vs_ours": round(float((m["weekly_sales_kit"] - m["weekly_sales_ours"]).abs().median()), 3)}
    (run / "uc03" / "kit_versions_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
