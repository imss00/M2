"""UC03 weekly sales (Part A5; M1 2.2, 2.3, 3.1).

sales = quantity * price, summed per (store, department, week).
week  = ISO week (Mon..Sun); its Friday is the week's date, as in the kit's UseCase03.pre_process
        (pandas dt.week + strptime('%G-W%V-%u') with weekday 5).

Known kit difference: the kit corrects the year only for January dates in ISO week > 50, not for
December dates in ISO week 1, so those sales land in week 1 of the *same* calendar year. We use the
true ISO-week Friday and count the affected line items (check uc03.weekly.kit_year_bug_rows).
"""
from pyspark.sql import DataFrame, functions as F

from src.common.config import RunConfig
from src.common.metrics import RunRecorder


def weekly_table(cfg: RunConfig) -> str:
    return f"{cfg.db('features')}.uc03_weekly_sales"


def iso_week_friday(col):
    # date_trunc('week') = Monday of the ISO week; +4 days = Friday
    return F.date_add(F.to_date(F.date_trunc("week", col)), 4)


def build_weekly(joined: DataFrame) -> DataFrame:
    # Rounded to 6 decimals: prices have 2 decimals and quantities are integers, so this keeps the
    # exact value but removes floating-point noise from the summation order, which otherwise
    # differs between separate and shared mode and changes the ETS optimum for some pairs.
    return (joined.withColumn("week", iso_week_friday(F.col("date")))
            .groupBy("store", "department", "week")
            .agg(F.round(F.sum(F.col("quantity") * F.col("price")), 6).alias("weekly_sales")))


def write_weekly(spark, cfg, rec: RunRecorder, joined: DataFrame) -> DataFrame:
    with rec.stage("uc03_aggregate_weekly", "train", "preprocess") as st:
        build_weekly(joined).write.mode("overwrite").format("parquet") \
            .saveAsTable(weekly_table(cfg))
        st["rows_out"] = spark.table(weekly_table(cfg)).count()
        st["cols_out"] = 4
    return spark.table(weekly_table(cfg))


def _quantiles(df, col):
    r = df.agg(F.expr(f"percentile_approx({col}, array(0.05, 0.5, 0.95))").alias("q"),
               F.avg((F.col(col) < 0).cast("double")).alias("neg")).first()
    return {"p5": round(r["q"][0], 2), "p50": round(r["q"][1], 2), "p95": round(r["q"][2], 2),
            "negative_share": round(r["neg"], 4)}


def validate_weekly(rec: RunRecorder, joined: DataFrame, weekly: DataFrame):
    """Sales reconciliation, gaps, and target distribution (M1 2.3; Tolovski et al. 4.2)."""
    with rec.stage("check_uc03_weekly", "train", "check"):
        line = joined.groupBy("store", "department").agg(
            F.sum(F.col("quantity") * F.col("price")).alias("a"))
        week = weekly.groupBy("store", "department").agg(
            F.sum("weekly_sales").alias("b"), F.count(F.lit(1)).alias("weeks"),
            F.min("week").alias("first"), F.max("week").alias("last"))
        week = week.withColumn("span", (F.datediff("last", "first") / 7 + 1).cast("int")) \
                   .withColumn("missing", F.col("span") - F.col("weeks"))
        cmp = line.join(week, ["store", "department"], "full_outer").agg(
            F.sum("a").alias("total_line"), F.sum("b").alias("total_weekly"),
            F.max(F.abs(F.coalesce("a", F.lit(0.0)) - F.coalesce("b", F.lit(0.0)))).alias("max_pair_diff"),
            F.count(F.lit(1)).alias("pairs"),
            F.min("weeks").alias("weeks_min"), F.expr("percentile_approx(weeks, 0.5)").alias("weeks_median"),
            F.max("weeks").alias("weeks_max"),
            F.sum((F.col("weeks") < 104).cast("int")).alias("pairs_under_2y"),
            F.sum((F.col("missing") > 0).cast("int")).alias("pairs_with_gaps"),
            F.sum("missing").alias("missing_weeks"),
            F.min("first").alias("first_week"), F.max("last").alias("last_week"),
        ).first()
        n_pairs_ending_early = week.where(F.col("last") < F.lit(cmp["last_week"])).count()
        non_friday = weekly.where(F.dayofweek("week") != 6).count()
        dup = weekly.groupBy("store", "department", "week").count().where("count > 1").count()
        kit_bug = joined.where((F.month("date") == 12) & (F.weekofyear("date") == 1)).count()
        q_line = _quantiles(joined.select((F.col("quantity") * F.col("price")).alias("s")), "s")
        q_week = _quantiles(weekly, "weekly_sales")

    tol = 1e-6 * max(abs(cmp["total_line"] or 0), 1.0)
    rec.check("uc03.weekly.total_sales_line_items", round(cmp["total_line"], 2))
    rec.check("uc03.weekly.total_sales_weekly", round(cmp["total_weekly"], 2),
              ok=abs(cmp["total_line"] - cmp["total_weekly"]) <= tol,
              note="sum of line-item sales must equal sum of weekly sales")
    rec.check("uc03.weekly.max_pair_difference", round(cmp["max_pair_diff"], 6),
              ok=cmp["max_pair_diff"] <= tol, note="per store-department reconciliation")
    rec.check("uc03.weekly.duplicate_keys", dup, expected=0)
    rec.check("uc03.weekly.week_not_friday", non_friday, expected=0)
    rec.check("uc03.weekly.pairs_with_sales", cmp["pairs"])
    rec.check("uc03.weekly.observed_weeks_min_median_max",
              f"{cmp['weeks_min']}/{cmp['weeks_median']}/{cmp['weeks_max']}")
    rec.check("uc03.weekly.pairs_under_104_weeks", cmp["pairs_under_2y"],
              note="fewer than two 52-week cycles")
    rec.check("uc03.weekly.pairs_with_gaps", cmp["pairs_with_gaps"],
              note="weeks without sales inside a pair's history; ETS treats them as consecutive")
    rec.check("uc03.weekly.missing_weeks", cmp["missing_weeks"])
    rec.check("uc03.weekly.pairs_ending_before_last_week", n_pairs_ending_early,
              note="kit forecasts from each pair's own last week; dates may miss the scoring labels")
    rec.check("uc03.weekly.week_range", f"{cmp['first_week']}..{cmp['last_week']}")
    rec.check("uc03.weekly.kit_year_bug_rows", kit_bug,
              note="December dates in ISO week 1; the kit assigns them to the wrong year")
    rec.check("uc03.target.line_item_sales_quantiles", q_line)
    rec.check("uc03.target.weekly_sales_quantiles", q_week,
              note="aggregation changes the target distribution (Tolovski et al. 2026, 3.3)")
    return cmp["last_week"]
