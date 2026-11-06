"""Order-Lineitem-Product join (Part A4; M1 2.2, 5.2).

Result = one row per line item with the columns in JOINED_COLUMNS. trip_type is never joined.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F

from src.common.config import RunConfig, JOINED_COLUMNS
from src.common.metrics import RunRecorder
from src.prep.ingest import clean_table


def joined_table(cfg: RunConfig, split: str) -> str:
    return f"{cfg.db('joined')}.{split}_lineitem_joined"


def order_lineitem_table(cfg: RunConfig, split: str) -> str:
    return f"{cfg.db('joined')}.{split}_order_lineitem"


def _inputs(spark, cfg, split):
    o = spark.table(clean_table(cfg, split, "order")).select(
        F.col("o_order_id").alias("order_id"), "date", "store", "weekday")
    li = spark.table(clean_table(cfg, split, "lineitem")).select(
        F.col("li_order_id").alias("order_id"), F.col("li_product_id").alias("product_id"),
        "quantity", "price")
    p = spark.table(clean_table(cfg, split, "product")).select(
        F.col("p_product_id").alias("product_id"), "department")
    return o, li, p


def build_joined(spark: SparkSession, cfg: RunConfig, split: str) -> DataFrame:
    """Lazy join. Product (~700 rows at SF1) is broadcast; Order-Lineitem is a shuffle join."""
    o, li, p = _inputs(spark, cfg, split)
    return li.join(o, "order_id").join(F.broadcast(p), "product_id").select(JOINED_COLUMNS)


def persist_joined(spark, cfg, rec: RunRecorder, split: str) -> DataFrame:
    """Shared mode: join once and save as a Hive table that UC03 and UC08 both read."""
    if cfg.split_join_timing:
        return _persist_joined_split(spark, cfg, rec, split)
    with rec.stage(f"join_{split}_persist", split, "preprocess") as st:
        build_joined(spark, cfg, split).write.mode("overwrite").format("parquet") \
            .saveAsTable(joined_table(cfg, split))
        st["rows_out"] = spark.table(joined_table(cfg, split)).count()
        st["cols_out"] = len(JOINED_COLUMNS)
    return spark.table(joined_table(cfg, split))


def _persist_joined_split(spark, cfg, rec, split):
    """--split-join-timing: materialise after each join so the two joins are timed apart (M1 5.3).
    The extra write adds cost, so do not mix these times with an unsplit shared run."""
    o, li, p = _inputs(spark, cfg, split)
    with rec.stage(f"join_{split}_order_lineitem", split, "preprocess") as st:
        li.join(o, "order_id").write.mode("overwrite").format("parquet") \
            .saveAsTable(order_lineitem_table(cfg, split))
        st["rows_out"] = spark.table(order_lineitem_table(cfg, split)).count()
        st["cols_out"] = len(spark.table(order_lineitem_table(cfg, split)).columns)
    with rec.stage(f"join_{split}_product", split, "preprocess") as st:
        spark.table(order_lineitem_table(cfg, split)).join(F.broadcast(p), "product_id") \
            .select(JOINED_COLUMNS).write.mode("overwrite").format("parquet") \
            .saveAsTable(joined_table(cfg, split))
        st["rows_out"] = spark.table(joined_table(cfg, split)).count()
        st["cols_out"] = len(JOINED_COLUMNS)
    n_li = li.count()
    no_order = li.join(o.select("order_id"), "order_id", "left_anti").count()
    rec.check(f"{split}.join.order_lineitem_rows",
              spark.table(order_lineitem_table(cfg, split)).count(), expected=n_li - no_order)
    return spark.table(joined_table(cfg, split))


def get_joined(spark, cfg: RunConfig, split: str) -> DataFrame:
    """Separate mode: a fresh lazy join every call. Shared mode: read the saved table."""
    if cfg.mode == "shared":
        return spark.table(joined_table(cfg, split))
    return build_joined(spark, cfg, split)


def validate_join(spark, cfg, rec: RunRecorder, split: str, joined: DataFrame):
    """Row counts around the joins and unmatched keys (M1 2.2, 2.3)."""
    o, li, p = _inputs(spark, cfg, split)
    with rec.stage(f"check_join_{split}", split, "check"):
        n_li = li.count()
        no_order = li.join(o.select("order_id"), "order_id", "left_anti").count()
        with_order = li.join(o.select("order_id"), "order_id", "left_semi")
        no_product = with_order.join(F.broadcast(p.select("product_id")), "product_id",
                                     "left_anti").count()
        n_joined = joined.count()
        n_orders = o.count()
        orders_with_items = joined.select("order_id").distinct().count()
    s = f"{split}.join"
    rec.check(f"{s}.lineitem_rows", n_li)
    rec.check(f"{s}.unmatched_order_id", no_order, note="line items without an order")
    rec.check(f"{s}.unmatched_product_id", no_product, note="line items without a product")
    rec.check(f"{s}.joined_rows", n_joined, expected=n_li - no_order - no_product,
              note="inner join must not add rows (unique keys) or lose matched ones")
    rec.check(f"{s}.orders_without_items", n_orders - orders_with_items)
    return n_joined
