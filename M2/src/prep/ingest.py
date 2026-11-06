"""Raw -> clean layer (Part A3; M1 2.3, 5.1).

raw    : external Hive tables over the generated CSV files (no copy, all columns string)
clean  : typed Parquet Hive tables; rows with missing required values or duplicate keys removed
labels : order_id -> trip_type (training order.csv, scoring order_labels.csv) and the UC03
         scoring truth (store_dept_labels.csv), kept apart from model inputs (M1 4.1)
"""
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from src.common.config import (RunConfig, SOURCE_SCHEMA, OPTIONAL_COLUMNS, LABEL_FILES,
                               TABLE_KEYS, SPLIT_TABLES, SPLIT_LABEL_FILES, is_uri, join_path)
from src.common.metrics import RunRecorder
from src.common.spark import table_exists


def raw_table(cfg: RunConfig, split: str, table: str) -> str:
    return f"{cfg.db('raw')}.{split}_{table}"


def clean_table(cfg: RunConfig, split: str, table: str) -> str:
    return f"{cfg.db('clean')}.{split}_{table}"


def labels_table(cfg: RunConfig, split: str, name: str = "order_labels") -> str:
    return f"{cfg.db('labels')}.{split}_{name}"


def _csv_path(base: str, name: str) -> str:
    if is_uri(base):                        # hdfs://...: Spark reports a missing file itself
        return join_path(base, f"{name}.csv")
    p = Path(base) / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing source file: {p}")
    return str(p.resolve())


def _files(split: str) -> list[str]:
    return SPLIT_TABLES[split] + SPLIT_LABEL_FILES.get(split, [])


def register_raw(spark: SparkSession, cfg: RunConfig, split: str):
    """Expose each CSV as an external Hive table in <sf>_raw (string columns, no inference)."""
    base = cfg.split_dir(split)
    for t in _files(split):
        name = raw_table(cfg, split, t)
        spark.sql(f"DROP TABLE IF EXISTS {name}")
        spark.catalog.createTable(name, path=_csv_path(base, t), source="csv",
                                  header="true", sep=cfg.sep)


def _typed(raw, cols: dict):
    return raw.select([F.trim(F.col(c)).try_cast(t).alias(c) if t != "string"
                       else F.trim(F.col(c)).alias(c) for c, t in cols.items()])


def ingest_split(spark: SparkSession, cfg: RunConfig, rec: RunRecorder, split: str):
    pipeline = split
    for t in SPLIT_TABLES[split]:
        _ingest_table(spark, cfg, rec, split, t, pipeline)
    for name in SPLIT_LABEL_FILES.get(split, []):
        _ingest_label_file(spark, cfg, rec, split, name, pipeline)
    mine = [c for c in rec.checks if c["check"].startswith(f"{split}.")]
    _ingest_log(cfg, split).write_text(json.dumps(
        {"checks": mine, "date_range": rec.info.get(f"{split}_date_range")}, default=str))


def _ingest_table(spark, cfg, rec, split, table, pipeline):
    raw = spark.table(raw_table(cfg, split, table))
    schema = SOURCE_SCHEMA[table]
    missing = [c for c in schema if c not in raw.columns]
    if missing:
        raise ValueError(f"{split}/{table}: columns {missing} not found; got {raw.columns}. "
                         "Edit SOURCE_SCHEMA in src/common/config.py.")
    optional = {c: t for c, t in OPTIONAL_COLUMNS.get(table, {}).items() if c in raw.columns}
    cols = {**schema, **optional}
    typed = _typed(raw, cols)

    required_ok = F.lit(True)
    for c in schema:
        required_ok = required_ok & F.col(c).isNotNull()
    aggs = [F.count(F.lit(1)).alias("rows_raw"),
            F.sum(required_ok.cast("int")).alias("rows_complete")]
    aggs += [F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in schema]
    if table == "lineitem":
        aggs += [F.sum((F.col("price") < 0).cast("int")).alias("price_negative"),
                 F.sum((F.col("quantity") == 0).cast("int")).alias("quantity_zero"),
                 F.sum((F.col("quantity") < 0).cast("int")).alias("quantity_negative")]
    if table == "order":
        aggs += [F.min("date").alias("date_min"), F.max("date").alias("date_max"),
                 # weekday column vs the date's weekday (UC08 uses the date, as the kit does)
                 F.sum((F.date_format("date", "EEEE") != F.col("weekday")).cast("int"))
                  .alias("weekday_mismatch")]

    with rec.stage(f"ingest_{split}_{table}", pipeline, "load") as st:
        stats = typed.agg(*aggs).first().asDict()
        clean = typed.where(required_ok)
        if table in TABLE_KEYS:
            clean = clean.dropDuplicates(TABLE_KEYS[table])
        out_cols = [c for c in cols if c != "trip_type"]
        clean.select(out_cols).write.mode("overwrite").format("parquet") \
            .saveAsTable(clean_table(cfg, split, table))
        if "trip_type" in optional:
            clean.select("o_order_id", "trip_type").where(F.col("trip_type").isNotNull()) \
                .write.mode("overwrite").format("parquet").saveAsTable(labels_table(cfg, split))
        n_clean = spark.table(clean_table(cfg, split, table)).count()
        st["rows_in"], st["rows_out"], st["cols_out"] = stats["rows_raw"], n_clean, len(out_cols)

    p = f"{split}.{table}"
    rec.check(f"{p}.rows_raw", stats["rows_raw"])
    for c in schema:
        if stats[f"null_{c}"]:
            rec.check(f"{p}.missing_or_unparsable.{c}", stats[f"null_{c}"], ok=False,
                      note="rows dropped from clean layer")
    rec.check(f"{p}.rows_dropped_incomplete", stats["rows_raw"] - stats["rows_complete"])
    if table in TABLE_KEYS:
        rec.check(f"{p}.duplicate_keys_removed", stats["rows_complete"] - n_clean, expected=0,
                  note="duplicate keys would multiply joined rows (M1 5.1)")
    rec.check(f"{p}.rows_clean", n_clean)
    if table == "lineitem":
        rec.check(f"{p}.price_negative", stats["price_negative"], expected=0)
        rec.check(f"{p}.quantity_zero", stats["quantity_zero"], expected=0)
        rec.check(f"{p}.quantity_negative", stats["quantity_negative"],
                  note="kept: returns; UC08 also uses abs(quantity)")
    if table == "order":
        rec.check(f"{p}.date_range", f"{stats['date_min']}..{stats['date_max']}")
        rec.info[f"{split}_date_range"] = [str(stats["date_min"]), str(stats["date_max"])]
        rec.check(f"{p}.weekday_mismatch", stats["weekday_mismatch"] or 0, expected=0,
                  note="weekday column vs weekday of date")
        rec.check(f"{p}.has_trip_type", "trip_type" in optional)
    if table == "store_dept":
        rec.check(f"{p}.has_periods", "periods" in optional)


def _ingest_label_file(spark, cfg, rec, split, name, pipeline):
    raw = spark.table(raw_table(cfg, split, name))
    cols = LABEL_FILES[name]
    with rec.stage(f"ingest_{split}_{name}", pipeline, "load") as st:
        typed = _typed(raw, cols).dropna()
        typed.write.mode("overwrite").format("parquet").saveAsTable(labels_table(cfg, split, name))
        st["rows_out"] = spark.table(labels_table(cfg, split, name)).count()
        st["cols_out"] = len(cols)
    rec.check(f"{split}.{name}.rows", st["rows_out"])


def _ingest_log(cfg: RunConfig, split: str) -> Path:
    d = Path(cfg.state_dir) / cfg.master
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{cfg.db_prefix}_{split}_ingest_checks.json"


def reuse_split(cfg: RunConfig, rec: RunRecorder, split: str):
    """Clean tables already exist: copy the data-quality results of the earlier ingest."""
    log = _ingest_log(cfg, split)
    rec.check(f"{split}.ingest", "skipped: clean tables reused (use --reingest to rebuild)")
    if log.exists():
        saved = json.loads(log.read_text())
        for c in saved["checks"]:
            rec.checks.append({**c, "note": (c.get("note") or "") + " [from earlier ingest]"})
        if saved.get("date_range"):
            rec.info[f"{split}_date_range"] = saved["date_range"]


def clean_ready(spark, cfg: RunConfig, split: str) -> bool:
    ok = all(table_exists(spark, clean_table(cfg, split, t)) for t in SPLIT_TABLES[split])
    return ok and all(table_exists(spark, labels_table(cfg, split, n))
                      for n in SPLIT_LABEL_FILES.get(split, []))
