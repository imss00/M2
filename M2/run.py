"""TPCx-AI UC03 + UC08 pipeline on PySpark with embedded Hive tables.

    # kit data generated into <kit>/output/raw_data/{training,serving,scoring}
    python run.py --data-dir <kit>/output/raw_data --sf 1 --mode shared --workers 4
    # or each folder explicitly
    python run.py --train-dir .../training --serve-dir .../serving --score-dir .../scoring --sf 1

Modes (M1 6):
  separate : UC03 and UC08 each join Order-Lineitem-Product themselves (join runs twice)
  shared   : join once per folder, save <sf>_joined.*_lineitem_joined, both workloads read it
Pipelines (TPCx-AI): train (training folder) -> serve (serving folder) -> score (scoring folder,
compared with store_dept_labels.csv / order_labels.csv: UC03 MSLE <= 5.4, UC08 accuracy >= 0.65)
Outputs: runs/<run_id>/{uc03,uc08}/{predictions,scoring_predictions}.csv, metrics.json,
         stages.csv, validation.json
"""
import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pyspark

from src.common.config import RunConfig, PROJECT_ROOT, is_uri, join_path
from src.common.metrics import RunRecorder
from src.common.jobs import library_versions, table_files
from src.common.spark import get_spark, shutdown_jvm
from src.prep import ingest, join
from src.uc03 import weekly as uc03_weekly, model as uc03_model, evaluate as uc03_eval
from src.uc08 import features as uc08_features, model as uc08_model
from src.checks.outputs import check_outputs


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", help="folder holding training/, serving/, scoring/ (kit raw_data)")
    ap.add_argument("--train-dir")
    ap.add_argument("--serve-dir")
    ap.add_argument("--score-dir")
    ap.add_argument("--sf", default="1", help="scale-factor label; also names the Hive databases")
    ap.add_argument("--mode", choices=["separate", "shared"], default="shared")
    ap.add_argument("--workloads", choices=["both", "uc03", "uc08"], default="both")
    ap.add_argument("--workers", type=int, default=2, help="Spark local[N] cores")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--shuffle-partitions", type=int)
    ap.add_argument("--sep", default=",", help="CSV delimiter of the source files")
    ap.add_argument("--reingest", action="store_true", help="rebuild clean tables even if present")
    ap.add_argument("--split-join-timing", action="store_true",
                    help="shared mode: time Order-Lineitem and Product joins separately")
    ap.add_argument("--fill-missing-weeks", action="store_true",
                    help="UC03: fill weeks without sales with 0 (kit: off)")
    ap.add_argument("--default-periods", type=int, default=52,
                    help="UC03 forecast weeks if store_dept has no periods column")
    ap.add_argument("--xgb-estimators", type=int, default=100)
    ap.add_argument("--model-python",
                    help="Python for the model steps, e.g. an env with the kit's versions "
                         "(statsmodels 0.12.2, xgboost 1.7.3); default: this Python")
    # cluster execution (docker/): Spark on YARN, data and warehouse on HDFS, metastore in PostgreSQL
    ap.add_argument("--master", choices=["local", "yarn"], default=os.environ.get("TPCXAI_MASTER", "local"))
    ap.add_argument("--executors", type=int, default=2, help="yarn: executor count")
    ap.add_argument("--executor-cores", type=int, default=2, help="yarn: cores per executor")
    ap.add_argument("--executor-memory", default="3g", help="yarn: memory per executor")
    ap.add_argument("--warehouse", default=os.environ.get("TPCXAI_WAREHOUSE"),
                    help="Hive warehouse dir (yarn: hdfs://namenode:9000/user/hive/warehouse)")
    ap.add_argument("--metastore-url", default=os.environ.get("TPCXAI_METASTORE_URL") or None,
                    help="yarn: JDBC URL of the Hive metastore database")
    ap.add_argument("--run-id")
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    a = ap.parse_args()

    train_dir, serve_dir, score_dir = a.train_dir, a.serve_dir, a.score_dir
    if a.data_dir:
        if is_uri(a.data_dir):      # hdfs://...: the three folders are expected to exist
            train_dir = train_dir or join_path(a.data_dir, "training")
            serve_dir = serve_dir or join_path(a.data_dir, "serving")
            score_dir = score_dir or join_path(a.data_dir, "scoring")
        else:
            d = Path(a.data_dir)
            train_dir = train_dir or str(d / "training")
            serve_dir = serve_dir or (str(d / "serving") if (d / "serving").exists() else None)
            score_dir = score_dir or (str(d / "scoring") if (d / "scoring").exists() else None)
    if not train_dir:
        ap.error("give --data-dir or --train-dir")
    extra = {}
    if a.warehouse:
        extra["warehouse_dir"] = a.warehouse
    if a.master == "yarn" and not a.warehouse:
        ap.error("--master yarn needs --warehouse (or TPCXAI_WAREHOUSE)")
    return RunConfig(**extra, master=a.master, executors=a.executors, executor_cores=a.executor_cores,
                     executor_memory=a.executor_memory, metastore_url=a.metastore_url,
                     train_dir=train_dir, serve_dir=serve_dir, score_dir=score_dir, sf=a.sf,
                     mode=a.mode, workloads=a.workloads, workers=a.workers,
                     driver_memory=a.driver_memory, shuffle_partitions=a.shuffle_partitions,
                     sep=a.sep, reingest=a.reingest, split_join_timing=a.split_join_timing,
                     fill_missing_weeks=a.fill_missing_weeks, default_periods=a.default_periods,
                     xgb_estimators=a.xgb_estimators, model_python=a.model_python,
                     run_id=a.run_id, runs_dir=a.runs_dir)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    cfg = parse_args()
    do03 = cfg.workloads in ("both", "uc03")
    do08 = cfg.workloads in ("both", "uc08")
    splits = ["train"] + [s for s in ("serve", "score") if cfg.split_dir(s)]
    if len(splits) == 1:
        print("note: no serving/scoring folder; only training runs")
    if cfg.split_join_timing and cfg.mode != "shared":
        print("warning: --split-join-timing applies to shared mode only; ignored")
        cfg.split_join_timing = False

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.master == "yarn":        # report parallelism as total executor cores
        cfg.workers = cfg.executors * cfg.executor_cores
    cfg.run_id = cfg.run_id or f"{stamp}_sf{cfg.sf}_{cfg.mode}_{cfg.master}_w{cfg.workers}"
    run_dir = Path(cfg.runs_dir) / cfg.run_id
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; pick another --run-id")
    run_dir.mkdir(parents=True)

    spark = get_spark(cfg, event_log_dir=run_dir / "spark-events")
    rec = RunRecorder(run_dir, {
        "run_id": cfg.run_id, "sf": cfg.sf, "mode": cfg.mode, "master": cfg.master,
        "workers": cfg.workers,
        "executors": cfg.executors if cfg.master == "yarn" else None,
        "driver_memory": cfg.driver_memory, "workloads": cfg.workloads, "splits": splits,
        "spark_version": spark.version, "pyspark_version": pyspark.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "code_version": git_commit(), "config": cfg.as_dict(),
        "model_libraries": library_versions(cfg),
    })
    where = f"yarn {cfg.executors}x{cfg.executor_cores} cores" if cfg.master == "yarn" else f"local[{cfg.workers}]"
    print(f"run {cfg.run_id}  (Spark {spark.version}, {where}, mode={cfg.mode}, "
          f"splits={splits})")

    try:
        # load: raw + clean layers (shared by both modes)
        for split in splits:
            ingest.register_raw(spark, cfg, split)
            if cfg.reingest or not ingest.clean_ready(spark, cfg, split):
                ingest.ingest_split(spark, cfg, rec, split)
            else:
                ingest.reuse_split(cfg, rec, split)

        # shared mode: one join per folder that a workload joins (UC03 serve/score read store_dept only)
        join_splits = ["train"] + ([s for s in splits if s != "train"] if do08 else [])
        if cfg.mode == "shared":
            for split in join_splits:
                join.persist_joined(spark, cfg, rec, split)
        join.validate_join(spark, cfg, rec, "train", join.get_joined(spark, cfg, "train"))

        if do03:
            joined = join.get_joined(spark, cfg, "train")      # separate: joins again here
            weekly = uc03_weekly.write_weekly(spark, cfg, rec, joined)
            uc03_weekly.validate_weekly(rec, joined, weekly)
            uc03_model.train(spark, cfg, rec, weekly, run_dir / "uc03")
            for split in [s for s in splits if s != "train"]:
                store_dept = spark.table(ingest.clean_table(cfg, split, "store_dept"))
                name = "predictions.csv" if split == "serve" else "scoring_predictions.csv"
                pred = uc03_model.serve(spark, cfg, rec, store_dept, split, run_dir / "uc03" / name)
                if split == "score":
                    uc03_eval.score(spark, cfg, rec, pred)

        if do08:
            # 1) all Spark work: features for every folder, labelled training table, file lists
            feats = {split: uc08_features.write_features(spark, cfg, rec,
                                                         join.get_joined(spark, cfg, split),
                                                         split, run_dir / "uc08")
                     for split in splits}
            labelled_files = uc08_model.prepare_training(spark, cfg, rec, feats["train"])
            files = {s: table_files(feats[s]) for s in splits if s != "train"}
            truth_files = uc08_model.score_truth_files(spark, cfg) if "score" in files else []
            # 2) stop Spark and end its JVM so XGBoost gets the memory (SF1 OOM otherwise)
            shutdown_jvm(spark)
            spark = None
            # 3) XGBoost in a child process, reading the Parquet files directly
            model_dir = uc08_model.train(cfg, rec, labelled_files, run_dir / "uc08")
            for split, fs in files.items():
                name = "predictions.csv" if split == "serve" else "scoring_predictions.csv"
                pred = uc08_model.predict(cfg, rec, model_dir, fs, split, run_dir / "uc08" / name)
                if split == "score":
                    uc08_model.score(cfg, rec, pred, truth_files)

        with rec.stage("check_outputs", "serve", "check"):
            check_outputs(run_dir, rec.check)
    finally:
        failed = rec.save()
        if spark is not None:
            spark.stop()

    print(f"\ndone: {run_dir}")
    print(f"work time {rec.info['work_seconds']}s, total {rec.info['total_seconds']}s")
    for k, v in rec.info["phase_seconds"].items():
        print(f"  {k:20} {v:8.2f}s")
    print(f"checks failed: {failed}  -> see validation.json")


if __name__ == "__main__":
    main()
