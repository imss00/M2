"""Reference check: the kit's UC08 logic (UseCase08.py v2.0.0: load_data, pre_process, train,
serve), reproduced with pandas, run on the same folders as our pipeline. Compares accuracy and
per-order agreement with our predictions.

    python tools/kit_uc08_reference.py <train_dir> <score_dir> <our scoring_predictions.csv>

Only for verification; it loads everything into pandas and is slow/memory hungry at SF1.
The department list and label set are taken from src/common/config.py (same as the kit).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import UC08_DEPARTMENTS, UC08_LABELS, WEEKDAYS  # noqa: E402

FEATURES = ["scan_count", "scan_count_abs"] + WEEKDAYS + UC08_DEPARTMENTS


def load(d):
    o = pd.read_csv(f"{d}/order.csv", parse_dates=["date"])
    li = pd.read_csv(f"{d}/lineitem.csv")
    p = pd.read_csv(f"{d}/product.csv")
    x = o.merge(li, left_on="o_order_id", right_on="li_order_id") \
         .merge(p, left_on="li_product_id", right_on="p_product_id")
    cols = ["o_order_id", "date", "department", "quantity"] + (["trip_type"] if "trip_type" in x else [])
    return x[cols]


def pre_process(raw):
    has_labels = "trip_type" in raw.columns
    raw["scan_count"] = raw["quantity"]
    raw["weekday"] = raw["date"].dt.day_name()
    g = raw.groupby("o_order_id")
    feats = pd.DataFrame({"scan_count": g["scan_count"].sum(),
                          "scan_count_abs": g["scan_count"].apply(lambda s: np.abs(s).sum())})
    if has_labels:
        feats["trip_type"] = g["trip_type"].min()
    wd = raw.pivot_table(index="o_order_id", columns="weekday", values="scan_count",
                         aggfunc=lambda s: int(s.count() > 0)).fillna(0.0)
    for c in set(WEEKDAYS) - set(wd.columns):
        wd[c] = 0.0
    dep = raw.pivot_table(index="o_order_id", columns="department", values="scan_count", aggfunc="sum")
    for c in set(UC08_DEPARTMENTS) - set(dep.columns):
        dep[c] = 0.0
    final = feats.join(wd).join(dep).fillna(0.0)
    if has_labels:
        final = final[final["trip_type"] != 14]
        return final["trip_type"].to_numpy(), final[FEATURES]
    return None, final[FEATURES]


def main(train_dir, score_dir, ours_csv):
    sorted_labels = sorted(UC08_LABELS, key=str)
    t = time.time()
    y_raw, X = pre_process(load(train_dir))
    present = sorted(np.unique(y_raw).tolist(), key=str)   # xgboost>=2 needs 0..K-1
    enc = {c: i for i, c in enumerate(present)}
    clf = XGBClassifier(tree_method="hist", objective="multi:softprob", n_estimators=100)
    clf.fit(csr_matrix(X[FEATURES].to_numpy()), np.array([enc[v] for v in y_raw]))
    print(f"kit-logic train: {len(X)} orders, {len(present)} classes, {time.time() - t:.0f}s")

    _, Xs = pre_process(load(score_dir))
    pred = np.array(present)[clf.predict(csr_matrix(Xs.to_numpy()))]
    kit = pd.DataFrame({"o_order_id": Xs.index, "trip_type": pred})
    truth = pd.read_csv(f"{score_dir}/order_labels.csv")
    ours = pd.read_csv(ours_csv)
    m = truth.merge(kit, on="o_order_id", suffixes=("", "_kit")).merge(ours, on="o_order_id",
                                                                       suffixes=("", "_ours"))
    print(f"accuracy kit-logic : {(m.trip_type == m.trip_type_kit).mean():.4f}")
    print(f"accuracy ours      : {(m.trip_type == m.trip_type_ours).mean():.4f}")
    print(f"same prediction    : {(m.trip_type_kit == m.trip_type_ours).mean():.4f} of {len(m)} orders")
    _ = sorted_labels


if __name__ == "__main__":
    main(*sys.argv[1:4])
