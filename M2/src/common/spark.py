"""SparkSession with Hive tables.

local : Spark local[N], embedded Hive metastore (Derby file), warehouse on the local disk.
yarn  : Spark on YARN (Hadoop cluster, docker/), warehouse on HDFS, Hive metastore schema in
        PostgreSQL (--metastore-url). Executors use the same image, Python and project mount.
"""
import os
from pathlib import Path

from pyspark.sql import SparkSession

from .config import PROJECT_ROOT, RunConfig


def _base(cfg: RunConfig, parallelism: int):
    return (SparkSession.builder.appName(f"tpcxai-m2-sf{cfg.sf}-{cfg.mode}")
            .config("spark.driver.memory", cfg.driver_memory)
            .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions or max(parallelism * 4, 8)))
            .config("spark.ui.showConsoleProgress", "false"))


def _local(cfg: RunConfig):
    warehouse = Path(cfg.warehouse_dir).resolve()
    metastore = Path(cfg.metastore_dir).resolve()
    warehouse.mkdir(parents=True, exist_ok=True)
    return (_base(cfg, cfg.workers).master(f"local[{cfg.workers}]")
            .config("spark.sql.warehouse.dir", str(warehouse))
            # Absolute metastore path: tables are found no matter which folder you run from.
            .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                    f"jdbc:derby:;databaseName={metastore};create=true")
            .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={metastore.parent}"))


def _yarn(cfg: RunConfig):
    java17 = os.environ.get("SPARK_JAVA_HOME", os.environ.get("JAVA_HOME", ""))
    pyexec = os.environ.get("PYSPARK_PYTHON", "python3")
    b = (_base(cfg, cfg.executors * cfg.executor_cores).master("yarn")
         .config("spark.submit.deployMode", "client")
         .config("spark.executor.instances", str(cfg.executors))
         .config("spark.executor.cores", str(cfg.executor_cores))
         .config("spark.executor.memory", cfg.executor_memory)
         .config("spark.sql.warehouse.dir", cfg.warehouse_dir)
         # executors: same Java 17 / Python / project as the driver (all containers share the image)
         .config("spark.yarn.appMasterEnv.JAVA_HOME", java17)
         .config("spark.executorEnv.JAVA_HOME", java17)
         .config("spark.executorEnv.PYTHONPATH", str(PROJECT_ROOT))
         .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", pyexec)
         .config("spark.executorEnv.PYSPARK_PYTHON", pyexec))
    if os.environ.get("SPARK_YARN_ARCHIVE"):          # Spark jars pre-uploaded to HDFS (docker/init)
        b = b.config("spark.yarn.archive", os.environ["SPARK_YARN_ARCHIVE"])
    if cfg.metastore_url:
        # Hive metastore schema in PostgreSQL, created by Spark's built-in Hive client on first use.
        # Caution: with DataNucleus auto-creation the first CREATE TABLE can block on PostgreSQL
        # locks; the default below (embedded Derby on the client) is the tested setup.
        b = (b.config("spark.hadoop.javax.jdo.option.ConnectionURL", cfg.metastore_url)
             .config("spark.hadoop.javax.jdo.option.ConnectionDriverName", "org.postgresql.Driver")
             .config("spark.hadoop.javax.jdo.option.ConnectionUserName", os.environ.get("METASTORE_USER", "hive"))
             .config("spark.hadoop.javax.jdo.option.ConnectionPassword", os.environ.get("METASTORE_PASSWORD", "hive"))
             .config("spark.hadoop.datanucleus.schema.autoCreateAll", "true")
             .config("spark.hadoop.hive.metastore.schema.verification", "false"))
    else:
        # Hive metastore embedded in the driver (Derby files in the project folder);
        # table data stays on HDFS (spark.sql.warehouse.dir)
        metastore = (Path(cfg.metastore_dir).parent / "metastore_cluster").resolve()
        b = (b.config("spark.hadoop.javax.jdo.option.ConnectionURL",
                      f"jdbc:derby:;databaseName={metastore};create=true")
             .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={metastore.parent}"))
    return b


def get_spark(cfg: RunConfig, event_log_dir: Path | None = None) -> SparkSession:
    builder = (_yarn(cfg) if cfg.master == "yarn" else _local(cfg)).enableHiveSupport()
    if event_log_dir is not None:
        event_log_dir.mkdir(parents=True, exist_ok=True)
        # Spark event log: shuffle bytes, spill, per-task time for M3 (Spark History Server).
        builder = (builder.config("spark.eventLog.enabled", "true")
                   .config("spark.eventLog.dir", event_log_dir.resolve().as_uri()))

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    for layer in ("raw", "clean", "labels", "joined", "features"):
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {cfg.db(layer)}")
    return spark


def shutdown_jvm(spark: SparkSession) -> None:
    """Stop Spark AND end its JVM process. spark.stop() alone leaves the py4j gateway JVM alive
    with its heap (3.5 GB at SF1), which then competes with XGBoost for memory."""
    from pyspark import SparkContext

    gateway = SparkContext._gateway
    spark.stop()
    if gateway is not None:
        try:
            gateway.shutdown()
        except Exception:  # noqa: BLE001
            pass
        proc = getattr(gateway, "proc", None)
        if proc is not None:
            proc.kill()
            proc.wait(timeout=30)
    SparkContext._gateway = None
    SparkContext._jvm = None


def table_exists(spark: SparkSession, name: str) -> bool:
    return spark.catalog.tableExists(name)
