"""UC03 model step, written like UseCase03.train / serve (kit v2.0.0).

    python -m src.jobs.uc03_job train   <job.json>   files, model_file, seasonal_periods, fill_missing
    python -m src.jobs.uc03_job predict <job.json>   request_files, model_file, default_periods, out_file

train : per (store, department): Series indexed by week date, sorted;
        ExponentialSmoothing(seasonal="add", seasonal_periods=52).fit(). If that raises (the kit
        run would stop), fall back to simple exponential smoothing, then the last value.
predict: for each store_dept row with a model: forecast(periods), clip at 0, dates = Fridays
        after the pair's last training week. Pairs without a model are skipped.
"""
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _read(files, columns=None):
    return pd.concat([pd.read_parquet(f, columns=columns) for f in sorted(files)], ignore_index=True)


def train(job):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    weekly = _read(job["files"])
    weekly["week"] = pd.to_datetime(weekly["week"])
    sp = job["seasonal_periods"]
    models, kinds = {}, {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (store, dept), g in weekly.groupby(["store", "department"]):
            ts = g.set_index("week")["weekly_sales"].sort_index()
            ts_max = ts.index.max()
            if job["fill_missing"]:
                ts = ts.reindex(pd.date_range(ts.index.min(), ts_max, freq="W-FRI"), fill_value=0.0)
            fit, kind = None, "naive_last"
            for k, make in (("ets_seasonal", lambda: ExponentialSmoothing(ts, seasonal="add", seasonal_periods=sp)),
                            ("ets_level", lambda: ExponentialSmoothing(ts))):
                try:
                    fit, kind = make().fit(), k
                    break
                except Exception:  # noqa: BLE001
                    continue
            models[(int(store), dept)] = (fit if fit is not None else float(ts.iloc[-1]), ts_max)
            kinds[kind] = kinds.get(kind, 0) + 1
    Path(job["model_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(job["model_file"], "wb") as f:
        pickle.dump(models, f)
    return {"pairs": len(models), "model_kinds": kinds}


def predict(job):
    with open(job["model_file"], "rb") as f:
        models = pickle.load(f)
    req = _read(job["request_files"])
    if "periods" not in req.columns:
        req["periods"] = job["default_periods"]
    out, requested, expected = [], len(req), 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in req.itertuples(index=False):
            key = (int(r.store), r.department)
            if key not in models:
                continue
            obj, ts_max = models[key]
            periods = int(r.periods)
            expected += periods
            values = np.asarray(obj.forecast(periods)) if hasattr(obj, "forecast") else np.full(periods, obj)
            dates = pd.date_range(pd.Timestamp(ts_max) + pd.Timedelta(days=1), periods=periods, freq="W-FRI")
            out.append(pd.DataFrame({"store": key[0], "department": key[1], "date": dates.date,
                                     "weekly_sales": np.clip(values, 0.0, None)}))
    pred = pd.concat(out, ignore_index=True) if out else \
        pd.DataFrame(columns=["store", "department", "date", "weekly_sales"])
    pred = pred.sort_values(["store", "department", "date"]).reset_index(drop=True)
    pred.to_csv(job["out_file"], index=False)
    return {"rows": len(pred), "requested_pairs": requested,
            "pairs_with_model": int(pred[["store", "department"]].drop_duplicates().shape[0]),
            "expected_rows": expected}


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    result = {"train": train, "predict": predict}[cmd](json.loads(path.read_text()))
    path.with_suffix(".result.json").write_text(json.dumps(result))
