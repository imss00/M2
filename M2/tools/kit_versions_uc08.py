"""UC08 with the kit's library versions (tools/python/python.yaml: python 3.9, xgboost 1.7.3,
pandas 1.5.2, scikit-learn 1.2.2, numpy 1.23.5), trained exactly like UseCase08.train:
XGBClassifier(tree_method='hist', objective='multi:softprob', n_estimators=100) on a CSR matrix,
labels encoded with the kit's label_to_index.

Features come from our Spark feature tables, which match the kit's pre_process output
(tools/kit_uc08_reference.py: 100% identical predictions). So only the library versions change.

    <py39-env>/bin/python tools/kit_versions_uc08.py <warehouse> <run_dir>
      warehouse : e.g. warehouse (contains sf1_features.db, sf1_labels.db)
      run_dir   : a finished run (reads uc08/feature_columns.json, writes uc08/kit_versions_*.csv)

Runs on Python 3.9, so it does not import src/ (which needs 3.10+).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

# UseCase08.label_range / sorted_labels / label_to_index
LABEL_RANGE = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]
SORTED_LABELS = sorted(LABEL_RANGE, key=str)
LABEL_TO_INDEX = {k: v for v, k in enumerate(SORTED_LABELS)}


def read(table_dir: Path, cols, extra=()):
    files = sorted(table_dir.glob("*.parquet"))
    ids, ex, rows, cs, vals, off = [], {c: [] for c in extra}, [], [], [], 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=500_000, columns=["order_id", *cols, *extra]):
            ids.append(b.column("order_id").to_numpy())
            for c in extra:
                ex[c].append(b.column(c).to_numpy())
            for j, c in enumerate(cols):
                v = b.column(c).to_numpy().astype(np.float64)        # kit: float64 DataFrame
                nz = np.flatnonzero(v)
                rows.append(nz + off)
                cs.append(np.full(len(nz), j, dtype=np.int32))
                vals.append(v[nz])
            off += b.num_rows
    X = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cs))),
                   shape=(off, len(cols)))
    oid = np.concatenate(ids)
    perm = np.argsort(oid, kind="stable")
    return oid[perm], X[perm], {c: np.concatenate(v)[perm] for c, v in ex.items()}


def main(warehouse, run_dir):
    wh, run = Path(warehouse), Path(run_dir)
    cols = json.loads((run / "uc08" / "feature_columns.json").read_text())["columns"]
    print("versions:", sys.version.split()[0], "xgboost", xgboost.__version__, "pandas", pd.__version__)

    t = time.time()
    _, X, ex = read(wh / "sf1_features.db" / "uc08_train_labelled", cols, ["trip_type"])
    y = np.array([LABEL_TO_INDEX[v] for v in ex["trip_type"]])
    print(f"train matrix {X.shape}, nnz {X.nnz}, classes present {len(np.unique(y))}, {time.time() - t:.0f}s")

    t = time.time()
    clf = XGBClassifier(tree_method="hist", objective="multi:softprob", n_estimators=100)
    clf.fit(X, y)
    train_s = time.time() - t
    print(f"train {train_s:.0f}s")
    del X

    oid, Xs, _ = read(wh / "sf1_features.db" / "uc08_score_order_features", cols)
    t = time.time()
    pred = np.array(SORTED_LABELS)[clf.predict(Xs)]
    print(f"predict {time.time() - t:.0f}s")
    out = pd.DataFrame({"o_order_id": oid, "trip_type": pred})
    out.to_csv(run / "uc08" / "kit_versions_scoring_predictions.csv", index=False)

    truth = pd.concat(pd.read_parquet(f) for f in sorted((wh / "sf1_labels.db" / "score_order_labels").glob("*.parquet")))
    ours = pd.read_csv(run / "uc08" / "scoring_predictions.csv")
    m = truth.merge(out, on="o_order_id", suffixes=("", "_kit")).merge(ours, on="o_order_id", suffixes=("", "_ours"))
    res = {"xgboost": xgboost.__version__, "train_seconds": round(train_s, 1), "orders": len(m),
           "accuracy_kit_versions": round(float((m.trip_type == m.trip_type_kit).mean()), 4),
           "accuracy_ours": round(float((m.trip_type == m.trip_type_ours).mean()), 4),
           "same_prediction": round(float((m.trip_type_kit == m.trip_type_ours).mean()), 4)}
    (run / "uc08" / "kit_versions_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
