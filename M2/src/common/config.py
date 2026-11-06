"""Run configuration and the shared table contract.

Column names, label sets and the department list follow TPCx-AI Tools v2.0.0
(data-gen/config/tpcxai-generation.xml, workload/python/workload/UseCase03.py, UseCase08.py).
"""
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source tables: column -> Spark type after cleaning. Columns marked optional may be absent
# (e.g. store_dept.periods exists only in the serving/scoring folders).
SOURCE_SCHEMA = {
    "order": {"o_order_id": "bigint", "date": "date", "store": "int", "weekday": "string"},
    "lineitem": {"li_order_id": "bigint", "li_product_id": "bigint",
                 "quantity": "bigint", "price": "double"},
    "product": {"p_product_id": "bigint", "department": "string"},
    "store_dept": {"store": "int", "department": "string"},
}
OPTIONAL_COLUMNS = {
    "order": {"trip_type": "int"},      # training folder only; goes to the labels table
    "store_dept": {"periods": "int"},   # serving and scoring folders
}
# Ground-truth files of the scoring folder (includeLabels=2 in the generator).
LABEL_FILES = {
    "order_labels": {"o_order_id": "bigint", "trip_type": "int"},
    "store_dept_labels": {"store": "int", "department": "string", "date": "date",
                          "weekly_sales": "double"},
}

# Logical unique keys (M1 5.1).
TABLE_KEYS = {
    "order": ["o_order_id"],
    "product": ["p_product_id"],
    "store_dept": ["store", "department"],
}

# Tables per folder. train = <kit>/output/raw_data/training, serve = .../serving, score = .../scoring
SPLIT_TABLES = {
    "train": ["order", "lineitem", "product"],
    "serve": ["order", "lineitem", "product", "store_dept"],
    "score": ["order", "lineitem", "product", "store_dept"],
}
SPLIT_LABEL_FILES = {"score": ["order_labels", "store_dept_labels"]}

# Shared joined contract: one row per line item.
JOINED_COLUMNS = ["order_id", "date", "store", "weekday", "product_id",
                  "department", "quantity", "price"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# UC08 department feature columns, in the kit's order (UseCase08.department_columns).
# Departments outside this list are ignored, as in the kit.
UC08_DEPARTMENTS = [
    "FINANCIAL SERVICES", "SHOES", "PERSONAL CARE", "PAINT AND ACCESSORIES", "DSD GROCERY",
    "MEAT - FRESH & FROZEN", "DAIRY", "PETS AND SUPPLIES", "HOUSEHOLD CHEMICALS/SUPP",
    "IMPULSE MERCHANDISE", "PRODUCE", "CANDY, TOBACCO, COOKIES", "GROCERY DRY GOODS", "BOYS WEAR",
    "FABRICS AND CRAFTS", "JEWELRY AND SUNGLASSES", "MENS WEAR", "ACCESSORIES", "HOME MANAGEMENT",
    "FROZEN FOODS", "SERVICE DELI", "INFANT CONSUMABLE HARDLINES", "PRE PACKED DELI",
    "COOK AND DINE", "PHARMACY OTC", "LADIESWEAR", "COMM BREAD", "BAKERY", "HOUSEHOLD PAPER GOODS",
    "CELEBRATION", "HARDWARE", "BEAUTY", "AUTOMOTIVE", "BOOKS AND MAGAZINES", "SEAFOOD",
    "OFFICE SUPPLIES", "LAWN AND GARDEN", "SHEER HOSIERY", "WIRELESS", "BEDDING", "BATH AND SHOWER",
    "HORTICULTURE AND ACCESS", "HOME DECOR", "TOYS", "INFANT APPAREL", "LADIES SOCKS",
    "PLUS AND MATERNITY", "ELECTRONICS", "GIRLS WEAR, 4-6X  AND 7-14", "BRAS & SHAPEWEAR",
    "LIQUOR,WINE,BEER", "SLEEPWEAR/FOUNDATIONS", "CAMERAS AND SUPPLIES", "SPORTING GOODS",
    "PLAYERS AND ELECTRONICS", "PHARMACY RX", "MENSWEAR", "OPTICAL - FRAMES", "SWIMWEAR/OUTERWEAR",
    "OTHER DEPARTMENTS", "MEDIA AND GAMING", "FURNITURE", "OPTICAL - LENSES", "SEASONAL",
    "LARGE HOUSEHOLD GOODS", "1-HR PHOTO", "CONCEPT STORES", "HEALTH AND BEAUTY AIDS",
]
# UC08 label set of the kit (UseCase08.label_range; 14 removed there as a tiny class).
UC08_LABELS = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
               31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]

# Quality thresholds of the kit (driver/dat/thresholds.yaml).
UC03_MSLE_THRESHOLD = 5.4      # mean_squared_log_error, lower is better
UC08_ACCURACY_THRESHOLD = 0.65  # accuracy_score, higher is better


def is_uri(path: str) -> bool:
    """hdfs://..., file://... (anything with a scheme) as opposed to a local path."""
    return "://" in str(path)


def join_path(base: str, *parts: str) -> str:
    return "/".join([str(base).rstrip("/"), *parts]) if is_uri(base) else str(Path(base, *parts))


@dataclass
class RunConfig:
    train_dir: str
    serve_dir: str | None = None
    score_dir: str | None = None
    sf: str = "1"
    mode: str = "shared"            # separate | shared
    workloads: str = "both"         # both | uc03 | uc08
    workers: int = 2
    driver_memory: str = "4g"
    shuffle_partitions: int | None = None
    sep: str = ","
    reingest: bool = False
    split_join_timing: bool = False
    runs_dir: str = str(PROJECT_ROOT / "runs")
    warehouse_dir: str = str(PROJECT_ROOT / "warehouse")
    metastore_dir: str = str(PROJECT_ROOT / "metastore_db")   # embedded Derby metastore
    run_id: str | None = None
    # UC03
    default_periods: int = 52       # used only when store_dept has no periods column
    seasonal_periods: int = 52
    fill_missing_weeks: bool = False   # kit: no filling
    # UC08
    exclude_trip_types: tuple = (14,)
    # Execution: local = Spark local[workers]; yarn = Spark on a Hadoop cluster (docker/)
    master: str = "local"
    executors: int = 2
    executor_cores: int = 2
    executor_memory: str = "3g"
    metastore_url: str | None = None   # yarn: jdbc:postgresql://metastore-db:5432/metastore
    state_dir: str = str(PROJECT_ROOT / "state")   # local bookkeeping (ingest checks)
    xgb_estimators: int = 100       # kit driver passes --num-rounds 100
    # Interpreter for the model steps (src/jobs). None = current Python (UC03 trains inside Spark).
    # A path = child processes with that Python for UC03 and UC08, e.g. an env with the kit's
    # versions (statsmodels 0.12.2, xgboost 1.7.3); results depend on these versions.
    model_python: str | None = None
    seed: int = 42

    @property
    def db_prefix(self) -> str:
        return "sf" + str(self.sf).replace(".", "_")

    def db(self, layer: str) -> str:
        """Hive database per data layer, e.g. sf1_clean (M1 Table 4)."""
        return f"{self.db_prefix}_{layer}"

    def split_dir(self, split: str) -> str | None:
        return {"train": self.train_dir, "serve": self.serve_dir, "score": self.score_dir}[split]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["exclude_trip_types"] = list(self.exclude_trip_types)
        return d
