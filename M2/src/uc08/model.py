"""UC08 trip-type classifier (Part B4, B5; M1 4.1, 4.2), matching UseCase08.train / serve.

- Labels come only from the labels tables, joined by order_id (no leakage into features).
- Training uses every labelled training order except trip_type 14 (kit: "remove tiny classes").
  No validation split, as in the kit; quality is measured on the scoring folder.
- The model step (src/jobs/uc08_job.py) runs XGBClassifier(tree_method="hist",
  objective="multi:softprob", n_estimators=100) exactly like the kit, in a child process whose
  interpreter is --model-python (kit versions: xgboost 1.7.3) or the current one.

Memory (SF1: 5.2M orders x 77 columns, 37 classes, 7 GB container): run.py finishes all Spark
work first, ends the Spark JVM, then runs the child process, which reads the Parquet files
directly. The JVM and XGBoost never hold memory at the same time.
"""
import shutil
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, functions as F

from src.common.config import RunConfig, UC08_LABELS, UC08_ACCURACY_THRESHOLD
from src.common.jobs import localize, run_job, table_files
from src.common.metrics import RunRecorder
from src.prep.ingest import labels_table
from src.uc08.features import FEATURE_COLUMNS


def _dist(share: dict) -> dict:
    share = {int(k): v for k, v in share.items()}
    top = max(share, key=share.get)
    return {"top_class": top, "top_share": share[top], "classes": len(share), "share": share}


# ---- Spark side -------------------------------------------------------------------------------

def prepare_training(spark, cfg: RunConfig, rec: RunRecorder, feats: DataFrame) -> list[str]:
    """Attach labels, drop excluded types, save the labelled table; return its Parquet files."""
    labels = spark.table(labels_table(cfg, "train")).select(
        F.col("o_order_id").alias("order_id"), "trip_type")
    table = f"{cfg.db('features')}.uc08_train_labelled"
    with rec.stage("uc08_attach_labels", "train", "preprocess") as st:
        data = feats.join(labels, "order_id", "inner")
        if cfg.exclude_trip_types:
            data = data.where(~F.col("trip_type").isin(list(cfg.exclude_trip_types)))
        data.select("order_id", *FEATURE_COLUMNS, "trip_type").write.mode("overwrite") \
            .format("parquet").saveAsTable(table)
        st["rows_out"] = spark.table(table).count()
        st["cols_out"] = len(FEATURE_COLUMNS)

    with rec.stage("check_uc08_labels_after_load", "train", "check"):
        counts = {r["trip_type"]: r["count"] for r in labels.groupBy("trip_type").count().collect()}
        total = sum(counts.values())
        after_load = _dist({k: round(v / total, 4) for k, v in sorted(counts.items())})
        outside = sum(v for k, v in counts.items() if k not in UC08_LABELS + [14])
    rec.check("uc08.train.labels_outside_kit_set", outside, expected=0)
    rec.check("uc08.labels.after_load", after_load)
    return table_files(spark.table(table))


def score_truth_files(spark, cfg: RunConfig) -> list[str]:
    return table_files(spark.table(labels_table(cfg, "score", "order_labels")))


# ---- child process (Spark already stopped) ------------------------------------------------------

def train(cfg: RunConfig, rec: RunRecorder, labelled_files: list[str], out_dir):
    with rec.stage("uc08_train", "train", "train") as st:
        res = run_job(cfg, "uc08_job", "train",
                      {"files": labelled_files, "columns": FEATURE_COLUMNS, "out_dir": str(out_dir),
                       "estimators": cfg.xgb_estimators, "seed": cfg.seed}, out_dir / "train_job.json")
        st["rows_out"], st["cols_out"] = res["rows"], len(FEATURE_COLUMNS)
    rec.check("uc08.train.rows", res["rows"])
    rec.check("uc08.train.classes", len(res["classes"]))
    rec.check("uc08.train.xgboost_version", res["xgboost"])
    rec.check("uc08.train.excluded_trip_types", list(cfg.exclude_trip_types))
    rec.check("uc08.labels.after_preprocess", _dist(res["share"]),
              note="Tolovski et al. 4.2: label shares shift between stages")
    return out_dir


def predict(cfg: RunConfig, rec: RunRecorder, model_dir, files: list[str], pipeline: str,
            out_file) -> pd.DataFrame:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with rec.stage(f"uc08_{pipeline}_predict", pipeline, "predict") as st:
        run_job(cfg, "uc08_job", "predict",
                {"files": files, "columns": FEATURE_COLUMNS, "out_dir": str(model_dir),
                 "out_file": str(out_file)}, model_dir / f"{pipeline}_job.json")
        out = pd.read_csv(out_file)
        st["rows_out"], st["cols_out"] = len(out), 2
    rec.check(f"uc08.{pipeline}.rows", len(out), expected=int(out["o_order_id"].nunique()),
              note="one prediction per order_id")
    return out


def score(cfg: RunConfig, rec: RunRecorder, pred: pd.DataFrame, truth_files: list[str]) -> float | None:
    """Kit driver: merge on o_order_id, sklearn accuracy_score, threshold 0.65."""
    with rec.stage("uc08_score", "score", "score") as st:
        tmp = Path(cfg.runs_dir) / cfg.run_id / "uc08" / "_truth"
        truth = pd.concat(pd.read_parquet(f) for f in localize(truth_files, tmp))
        shutil.rmtree(tmp, ignore_errors=True)
        m = truth.merge(pred, on="o_order_id", suffixes=("_true", "_pred"))
        acc = float((m["trip_type_true"] == m["trip_type_pred"]).mean()) if len(m) else None
        st["rows_out"] = len(m)
    share = truth["trip_type"].value_counts(normalize=True).sort_index()
    rec.check("uc08.score.label_rows", len(truth))
    rec.check("uc08.score.matched_rows", len(m))
    rec.check("uc08.labels.at_score", _dist({k: round(float(v), 4) for k, v in share.items()}))
    rec.check("uc08.score.accuracy", round(acc, 4) if acc is not None else None,
              ok=(acc is not None and acc >= UC08_ACCURACY_THRESHOLD),
              note=f"TPCx-AI threshold >= {UC08_ACCURACY_THRESHOLD}")
    return acc
