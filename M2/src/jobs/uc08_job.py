"""UC08 model step, written like UseCase08.train / serve (kit v2.0.0).

    python -m src.jobs.uc08_job train   <job.json>   files, columns, out_dir, estimators, seed
    python -m src.jobs.uc08_job predict <job.json>   files, columns, out_dir, out_file

train : CSR matrix of the feature columns (zeros absent, as csr_matrix of the kit's DataFrame),
        labels encoded in the kit's string-sorted order,
        XGBClassifier(tree_method="hist", objective="multi:softprob", n_estimators=100).
predict: model.predict -> decoded trip_type, one row per o_order_id.

The Parquet files are read batch by batch straight into the sparse matrix, so the dense
5.2M x 77 frame of SF1 is never built.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def read_matrix(files, columns, extra=()):
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix

    ids, ex = [], {c: [] for c in extra}
    rows, cols, vals = [], [], []
    offset = 0
    for path in sorted(files):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500000,
                                                       columns=["order_id"] + list(columns) + list(extra)):
            ids.append(batch.column("order_id").to_numpy())
            for c in extra:
                ex[c].append(batch.column(c).to_numpy())
            for j, c in enumerate(columns):
                v = batch.column(c).to_numpy().astype(np.float32)
                nz = np.flatnonzero(v)
                rows.append((nz + offset).astype(np.int64))
                cols.append(np.full(len(nz), j, dtype=np.int32))
                vals.append(v[nz])
            offset += batch.num_rows
    X = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(offset, len(columns)), dtype=np.float32)
    del rows, cols, vals
    order_id = np.concatenate(ids)
    extras = {c: np.concatenate(v) for c, v in ex.items()}
    perm = np.argsort(order_id, kind="stable")
    if not np.all(perm == np.arange(len(perm))):
        X, order_id = X[perm], order_id[perm]
        extras = {c: v[perm] for c, v in extras.items()}
    return order_id, X, extras


def train(job):
    import xgboost
    from xgboost.sklearn import XGBClassifier

    _, X, ex = read_matrix(job["files"], job["columns"], ["trip_type"])
    trip = ex["trip_type"]
    classes = sorted(np.unique(trip).tolist(), key=str)            # kit: sorted(key=str)
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[v] for v in trip])
    clf = XGBClassifier(tree_method="hist", objective="multi:softprob",
                        n_estimators=job["estimators"], random_state=job["seed"])
    clf.fit(X, y)
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out / "model.json"))
    (out / "classes.json").write_text(json.dumps([int(c) for c in classes]))
    share = pd.Series(trip).value_counts(normalize=True).sort_index()
    return {"rows": int(X.shape[0]), "classes": [int(c) for c in classes], "xgboost": xgboost.__version__,
            "share": {int(k): round(float(v), 4) for k, v in share.items()}}


def predict(job):
    from xgboost.sklearn import XGBClassifier

    out = Path(job["out_dir"])
    clf = XGBClassifier()
    clf.load_model(str(out / "model.json"))
    classes = np.array(json.loads((out / "classes.json").read_text()))
    order_id, X, _ = read_matrix(job["files"], job["columns"])
    pred = classes[np.asarray(clf.predict(X)).astype(int)]
    pd.DataFrame({"o_order_id": order_id.astype("int64"), "trip_type": pred}).to_csv(job["out_file"], index=False)
    return {"rows": int(len(order_id))}


if __name__ == "__main__":
    cmd, path = sys.argv[1], Path(sys.argv[2])
    result = {"train": train, "predict": predict}[cmd](json.loads(path.read_text()))
    path.with_suffix(".result.json").write_text(json.dumps(result))
