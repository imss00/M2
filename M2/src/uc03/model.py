"""UC03 per-pair forecasting (Part A6; M1 3.2, 3.3), split into train and serve like the kit.

train : one ExponentialSmoothing(seasonal="add", seasonal_periods=52) per store-department pair,
        fitted in parallel with groupBy().applyInPandas(); fitted models are pickled into the
        Hive table <sf>_features.uc03_models together with the pair's last week (ts_max).
serve : store_dept rows (store, department, periods) -> forecast(periods) starting the Friday after
        the pair's own ts_max; negatives clipped to 0; pairs without a model are skipped.

Same as UseCase03.py except one extension: if the seasonal fit raises (the kit run would stop),
we fall back to simple exponential smoothing, then to the last value, and record it per pair.
"""
import pickle
import warnings

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from src.common.config import RunConfig
from src.common.jobs import run_job, table_files
from src.common.metrics import RunRecorder

MODEL_SCHEMA = "store int, department string, ts_min date, ts_max date, n_weeks int, " \
               "model_kind string, model binary"
PRED_SCHEMA = "store int, department string, date date, weekly_sales double"


def models_table(cfg: RunConfig) -> str:
    return f"{cfg.db('features')}.uc03_models"


def make_trainer(sp: int, fill_missing: bool):
    def train_pair(pdf: pd.DataFrame) -> pd.DataFrame:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        pdf = pdf.sort_values("week")
        idx = pd.DatetimeIndex(pd.to_datetime(pdf["week"]))
        s = pd.Series(pdf["weekly_sales"].to_numpy(dtype=float), index=idx)
        ts_min, ts_max = idx.min(), idx.max()
        if fill_missing:
            s = s.reindex(pd.date_range(ts_min, ts_max, freq="W-FRI"), fill_value=0.0)
        # The kit passes a date index without freq, so statsmodels treats the observations as
        # consecutive steps. A plain array does the same and still allows forecast().
        s = pd.Series(s.to_numpy())
        fit, kind = None, "naive_last"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit, kind = ExponentialSmoothing(s, seasonal="add", seasonal_periods=sp).fit(), \
                    "ets_seasonal"
            except Exception:  # noqa: BLE001
                try:
                    fit, kind = ExponentialSmoothing(s).fit(), "ets_level"
                except Exception:  # noqa: BLE001
                    fit = None
        payload = pickle.dumps(fit if fit is not None else float(s.iloc[-1]))
        return pd.DataFrame({"store": [pdf["store"].iloc[0]], "department": [pdf["department"].iloc[0]],
                             "ts_min": [ts_min.date()], "ts_max": [ts_max.date()], "n_weeks": [len(s)],
                             "model_kind": [kind], "model": [payload]})
    return train_pair


def predict_pair(pdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in pdf.itertuples(index=False):
            periods = int(r.periods)
            obj = pickle.loads(r.model)
            values = obj.forecast(periods) if hasattr(obj, "forecast") else [obj] * periods
            dates = pd.date_range(pd.Timestamp(r.ts_max) + pd.Timedelta(days=1), periods=periods,
                                  freq="W-FRI")
            rows.append(pd.DataFrame({"store": r.store, "department": r.department, "date": dates.date,
                                      "weekly_sales": [max(float(v), 0.0) for v in values]}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["store", "department", "date", "weekly_sales"])


def train(spark, cfg: RunConfig, rec: RunRecorder, weekly: DataFrame, model_dir) -> None:
    if cfg.model_python:
        return _train_job(cfg, rec, weekly, model_dir)
    fn = make_trainer(cfg.seasonal_periods, cfg.fill_missing_weeks)
    with rec.stage("uc03_train", "train", "train") as st:
        weekly.groupBy("store", "department").applyInPandas(fn, MODEL_SCHEMA) \
            .write.mode("overwrite").format("parquet").saveAsTable(models_table(cfg))
        models = spark.table(models_table(cfg))
        st["rows_out"] = models.count()
    kinds = {r["model_kind"]: r["count"] for r in models.groupBy("model_kind").count().collect()}
    rec.check("uc03.train.models", st["rows_out"])
    rec.check("uc03.train.model_kinds", kinds,
              note="ets_level / naive_last = seasonal fit failed; the kit has no fallback")


def _train_job(cfg, rec, weekly, model_dir):
    """--model-python: train in a child process (e.g. statsmodels 0.12.2 like the kit)."""
    with rec.stage("uc03_train", "train", "train") as st:
        res = run_job(cfg, "uc03_job", "train",
                      {"files": table_files(weekly), "model_file": str(model_dir / "models.pkl"),
                       "seasonal_periods": cfg.seasonal_periods, "fill_missing": cfg.fill_missing_weeks},
                      model_dir / "train_job.json")
        st["rows_out"] = res["pairs"]
    rec.check("uc03.train.models", res["pairs"])
    rec.check("uc03.train.model_kinds", res["model_kinds"],
              note="ets_level / naive_last = seasonal fit failed; the kit has no fallback")


def _serve_job(cfg, rec, store_dept, pipeline, out_file):
    model_dir = out_file.parent
    with rec.stage(f"uc03_{pipeline}_predict", pipeline, "predict") as st:
        res = run_job(cfg, "uc03_job", "predict",
                      {"request_files": table_files(store_dept), "model_file": str(model_dir / "models.pkl"),
                       "default_periods": cfg.default_periods, "out_file": str(out_file)},
                      model_dir / f"{pipeline}_job.json")
        pred = pd.read_csv(out_file)
        st["rows_out"], st["cols_out"] = len(pred), 4
    rec.check(f"uc03.{pipeline}.requested_pairs", res["requested_pairs"])
    rec.check(f"uc03.{pipeline}.pairs_without_model", res["requested_pairs"] - res["pairs_with_model"],
              note="no training history: skipped, as in the kit")
    rec.check(f"uc03.{pipeline}.rows", len(pred), expected=res["expected_rows"],
              note="sum of periods over pairs with a model")
    return pred


def serve(spark, cfg: RunConfig, rec: RunRecorder, store_dept: DataFrame, pipeline: str,
          out_file) -> pd.DataFrame:
    """Forecast every requested pair that has a model (serving or scoring folder)."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if cfg.model_python:
        return _serve_job(cfg, rec, store_dept, pipeline, out_file)
    if "periods" not in store_dept.columns:
        store_dept = store_dept.withColumn("periods", F.lit(cfg.default_periods))
    req = store_dept.select("store", "department", "periods")
    with rec.stage(f"uc03_{pipeline}_predict", pipeline, "predict") as st:
        todo = req.join(spark.table(models_table(cfg)).select("store", "department", "ts_max", "model"),
                        ["store", "department"], "inner")
        pred = todo.groupBy("store", "department").applyInPandas(predict_pair, PRED_SCHEMA).toPandas()
        pred = pred.sort_values(["store", "department", "date"]).reset_index(drop=True)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out_file, index=False)
        st["rows_out"], st["cols_out"] = len(pred), 4

    n_req = req.count()
    expected = todo.agg(F.sum("periods")).first()[0] or 0
    rec.check(f"uc03.{pipeline}.requested_pairs", n_req)
    rec.check(f"uc03.{pipeline}.pairs_without_model", n_req - todo.count(),
              note="no training history: skipped, as in the kit")
    rec.check(f"uc03.{pipeline}.rows", len(pred), expected=int(expected),
              note="sum of periods over pairs with a model")
    return pred
