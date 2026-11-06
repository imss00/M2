"""UC08 order-level features (Part B3; M1 4.1, Table 3), matching UseCase08.pre_process.

One row per order_id, columns in the kit's featureColumns order:
  scan_count      = sum(quantity)
  scan_count_abs  = sum(|quantity|)
  wd_<Weekday>    = 1 if the order's date falls on that weekday (from the date, as in the kit)
  dept_NNN        = sum(quantity) of the order's items in UC08_DEPARTMENTS[NNN], 0 if absent
Departments outside the kit's list are ignored (the kit selects featureColumns only).
Spark column names cannot hold commas, so departments are dept_000..dept_067; the mapping is
written to uc08/feature_columns.json.
"""
import json

from pyspark.sql import DataFrame, functions as F

from src.common.config import RunConfig, WEEKDAYS, UC08_DEPARTMENTS
from src.common.metrics import RunRecorder

DEPT_COLS = [f"dept_{i:03d}" for i in range(len(UC08_DEPARTMENTS))]
WD_COLS = [f"wd_{d}" for d in WEEKDAYS]
FEATURE_COLUMNS = ["scan_count", "scan_count_abs"] + WD_COLS + DEPT_COLS


def features_table(cfg: RunConfig, split: str) -> str:
    return f"{cfg.db('features')}.uc08_{split}_order_features"


def build_features(joined: DataFrame) -> DataFrame:
    """Single groupBy over order_id with conditional sums (one shuffle, no pivot discovery).
    The kit runs these aggregations as chained single-threaded pandas steps (Tolovski et al. 3.3)."""
    q = F.col("quantity")
    aggs = [F.sum(q).alias("scan_count"), F.sum(F.abs(q)).alias("scan_count_abs"),
            F.min(F.date_format("date", "EEEE")).alias("weekday")]
    aggs += [F.sum(F.when(F.col("department") == d, q).otherwise(F.lit(0))).alias(c)
             for c, d in zip(DEPT_COLS, UC08_DEPARTMENTS)]
    per_order = joined.groupBy("order_id").agg(*aggs)
    wd = [F.when(F.col("weekday") == d, 1).otherwise(0).alias(c) for c, d in zip(WD_COLS, WEEKDAYS)]
    return per_order.select("order_id", "scan_count", "scan_count_abs", *wd, *DEPT_COLS)


def write_features(spark, cfg, rec: RunRecorder, joined: DataFrame, split: str, out_dir) -> DataFrame:
    with rec.stage(f"uc08_features_{split}", split, "preprocess") as st:
        build_features(joined).write.mode("overwrite").format("parquet") \
            .saveAsTable(features_table(cfg, split))
        feats = spark.table(features_table(cfg, split))
        st["rows_out"], st["cols_out"] = feats.count(), len(FEATURE_COLUMNS)

    with rec.stage(f"check_uc08_features_{split}", split, "check"):
        n_ids = feats.select("order_id").distinct().count()
        n_orders = joined.select("order_id").distinct().count()
        unknown_dept = joined.where(~F.col("department").isin(UC08_DEPARTMENTS)).count()
    p = f"uc08.features.{split}"
    rec.check(f"{p}.rows", st["rows_out"], expected=n_orders, note="one row per order with items")
    rec.check(f"{p}.duplicate_order_ids", st["rows_out"] - n_ids, expected=0)
    rec.check(f"{p}.items_outside_kit_departments", unknown_dept,
              note="ignored, as in the kit (featureColumns only)")
    rec.check(f"{p}.feature_columns", len(FEATURE_COLUMNS))
    if split == "train":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_columns.json").write_text(json.dumps(
            {"columns": FEATURE_COLUMNS, "department_of_column": dict(zip(DEPT_COLS, UC08_DEPARTMENTS))},
            indent=2, ensure_ascii=False))
    return feats
