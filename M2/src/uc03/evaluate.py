"""UC03 scoring with the kit's ground truth (scoring/store_dept_labels.csv).

Same as the TPCx-AI driver: merge predictions and labels on (store, department, date), then
sklearn mean_squared_log_error; threshold 5.4 (driver/dat/thresholds.yaml).
"""
import numpy as np
import pandas as pd

from src.common.config import RunConfig, UC03_MSLE_THRESHOLD
from src.common.metrics import RunRecorder
from src.prep.ingest import labels_table


def score(spark, cfg: RunConfig, rec: RunRecorder, pred: pd.DataFrame) -> float | None:
    with rec.stage("uc03_score", "score", "score") as st:
        truth = spark.table(labels_table(cfg, "score", "store_dept_labels")).toPandas()
        truth["date"] = pd.to_datetime(truth["date"]).dt.date
        p = pred.copy()
        p["date"] = pd.to_datetime(p["date"]).dt.date
        m = truth.merge(p, on=["store", "department", "date"], suffixes=("_true", "_pred"))
        y = m["weekly_sales_true"].to_numpy(dtype=float)
        yhat = m["weekly_sales_pred"].to_numpy(dtype=float)
        n_neg = int((y < 0).sum())
        # sklearn's MSLE rejects negatives; clip and report how many were clipped
        y = np.clip(y, 0, None)
        msle = float(np.mean((np.log1p(y) - np.log1p(yhat)) ** 2)) if len(m) else None
        per_pair = (m.assign(e=(np.log1p(y) - np.log1p(yhat)) ** 2)
                    .groupby(["store", "department"])["e"].mean().sort_values(ascending=False))
        st["rows_out"] = len(m)

    rec.check("uc03.score.label_rows", len(truth))
    rec.check("uc03.score.matched_rows", len(m))
    rec.check("uc03.score.matched_share_of_labels", round(len(m) / max(len(truth), 1), 4),
              note="labels without a matching forecast date are not scored")
    rec.check("uc03.score.matched_share_of_predictions", round(len(m) / max(len(pred), 1), 4))
    rec.check("uc03.score.negative_labels_clipped", n_neg)
    rec.check("uc03.score.msle", round(msle, 4) if msle is not None else None,
              ok=(msle is not None and msle <= UC03_MSLE_THRESHOLD),
              note=f"TPCx-AI threshold <= {UC03_MSLE_THRESHOLD}")
    rec.check("uc03.score.worst_pairs",
              {f"{s}|{d}": round(v, 3) for (s, d), v in per_pair.head(5).items()})
    return msle
