"""Fake data in the TPCx-AI kit's folder layout, for development without PDGF (Part B1).

Writes <out>/training, <out>/serving, <out>/scoring with the kit's file names and headers
(data-gen/config/tpcxai-generation.xml):
  training : order.csv (+trip_type), lineitem.csv, product.csv, store_dept.csv (store, department)
  serving  : order.csv (no trip_type), lineitem.csv, product.csv, store_dept.csv (+periods)
  scoring  : same as serving + order_labels.csv, store_dept_labels.csv (ground truth)
Department names and trip types come from the kit lists in src/common/config.py.
Deliberate anomalies in training (unmatched keys, a duplicate product, unparsable dates)
exercise the checks.

    python tools/gen_fake_data.py --out data/fake --orders 20000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import UC08_DEPARTMENTS  # noqa: E402

TRIP_TYPES = [3, 5, 7, 8, 9, 14, 39, 40]
TRIP_P = np.array([.08, .12, .15, .10, .15, .01, .24, .15])


def make_products(rng, n_products, depts):
    dept_of = rng.integers(0, len(depts), n_products)
    return pd.DataFrame({"p_product_id": np.arange(1, n_products + 1),
                         "name": [f"product {i}" for i in range(1, n_products + 1)],
                         "department": [depts[d] for d in dept_of]}), dept_of


def trip_profiles(rng, n_depts):
    pref = {t: rng.dirichlet(np.full(n_depts, 0.3)) for t in TRIP_TYPES}
    size = {t: 1 + i for i, t in enumerate(TRIP_TYPES)}
    return pref, size


def make_orders(rng, n, start, weeks, n_stores, first_id, n_products, dept_of, n_depts, profiles):
    # dates with yearly seasonality so weekly sales have a 52-week pattern (day index from start)
    days = np.arange(weeks * 7)
    abs_day = (pd.Timestamp(start) - pd.Timestamp("2019-01-01")).days + days
    w = 1 + 0.35 * np.sin(2 * np.pi * abs_day / 364.0) + 0.15 * (days % 7 >= 5)
    day = rng.choice(days, size=n, p=w / w.sum())
    dates = pd.Timestamp(start) + pd.to_timedelta(day, unit="D")
    trip = rng.choice(TRIP_TYPES, size=n, p=TRIP_P)
    orders = pd.DataFrame({"o_order_id": np.arange(first_id, first_id + n),
                           "o_customer_sk": rng.integers(1, 5000, n), "weekday": dates.day_name(),
                           "date": dates.strftime("%Y-%m-%d"), "store": rng.integers(1, n_stores + 1, n),
                           "trip_type": trip})
    pref, size = profiles
    prods_by_dept = [np.where(dept_of == d)[0] + 1 for d in range(n_depts)]
    rows_o, rows_p = [], []
    for oid, t in zip(orders.o_order_id.to_numpy(), trip):
        for d in rng.choice(n_depts, size=rng.poisson(size[t]) + 1, p=pref[t]):
            cand = prods_by_dept[d] if len(prods_by_dept[d]) else np.arange(1, n_products + 1)
            rows_o.append(oid)
            rows_p.append(rng.choice(cand))
    m = len(rows_o)
    qty = rng.integers(1, 4, m)
    qty[rng.random(m) < 0.02] *= -1                     # returns
    base_price = 1 + (np.arange(n_products) * 7.31) % 40
    lineitem = pd.DataFrame({"li_order_id": rows_o, "li_product_id": rows_p, "quantity": qty,
                             "price": np.round(base_price[np.array(rows_p) - 1], 2)})
    return orders, lineitem


def weekly_truth(orders, lineitem, products):
    """store_dept_labels.csv: true weekly sales by ISO-week Friday."""
    j = lineitem.merge(orders, left_on="li_order_id", right_on="o_order_id") \
                .merge(products, left_on="li_product_id", right_on="p_product_id")
    d = pd.to_datetime(j["date"])
    j["date"] = (d - pd.to_timedelta(d.dt.weekday, unit="D") + pd.Timedelta(days=4)).dt.strftime("%Y-%m-%d")
    j["weekly_sales"] = j["quantity"] * j["price"]
    return j.groupby(["store", "department", "date"], as_index=False)["weekly_sales"].sum()


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--orders", type=int, default=20000, help="training orders")
    ap.add_argument("--serve-orders", type=int, default=4000)
    ap.add_argument("--score-orders", type=int, default=20000)
    ap.add_argument("--stores", type=int, default=8)
    ap.add_argument("--departments", type=int, default=15)
    ap.add_argument("--products", type=int, default=300)
    ap.add_argument("--weeks", type=int, default=156, help="training history length")
    ap.add_argument("--periods", type=int, default=52, help="forecast weeks requested")
    ap.add_argument("--no-anomalies", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    depts = UC08_DEPARTMENTS[:a.departments]

    products, dept_of = make_products(rng, a.products, depts)
    profiles = trip_profiles(rng, len(depts))
    start = pd.Timestamp("2019-01-07")                     # a Monday
    next_start = start + pd.Timedelta(weeks=a.weeks)
    o_tr, li_tr = make_orders(rng, a.orders, start, a.weeks, a.stores, 1, a.products, dept_of,
                              len(depts), profiles)
    o_sv, li_sv = make_orders(rng, a.serve_orders, next_start, a.periods, a.stores, a.orders + 1,
                              a.products, dept_of, len(depts), profiles)
    o_sc, li_sc = make_orders(rng, a.score_orders, next_start, a.periods, a.stores,
                              a.orders + a.serve_orders + 1, a.products, dept_of, len(depts), profiles)

    pairs = pd.DataFrame([(s, d) for s in range(1, a.stores + 1) for d in depts],
                         columns=["store", "department"])
    pairs = pd.concat([pairs, pd.DataFrame({"store": [a.stores + 1], "department": [depts[0]]})])
    sd_req = pairs.assign(periods=a.periods)               # one pair has no history

    prod_tr, o_tr_out = products, o_tr.copy()
    if not a.no_anomalies:
        li_tr = pd.concat([li_tr, pd.DataFrame({            # 3 items without an order
            "li_order_id": [10**9, 10**9 + 1, 10**9 + 2], "li_product_id": [1, 2, 3],
            "quantity": [1, 1, 1], "price": [1.0, 1.0, 1.0]}),
            pd.DataFrame({"li_order_id": [3, 4], "li_product_id": [99999, 99998],  # 2 unknown products
                          "quantity": [1, 1], "price": [1.0, 1.0]})])
        prod_tr = pd.concat([products, products.iloc[[0]]])  # 1 duplicate product id
        o_tr_out.loc[o_tr_out.index[:2], "date"] = "N/A"     # 2 unparsable dates

    write(o_tr_out, out / "training/order.csv")
    write(li_tr, out / "training/lineitem.csv")
    write(prod_tr, out / "training/product.csv")
    write(pairs, out / "training/store_dept.csv")
    write(o_sv.drop(columns="trip_type"), out / "serving/order.csv")
    write(li_sv, out / "serving/lineitem.csv")
    write(products, out / "serving/product.csv")
    write(sd_req, out / "serving/store_dept.csv")
    write(o_sc.drop(columns="trip_type"), out / "scoring/order.csv")
    write(li_sc, out / "scoring/lineitem.csv")
    write(products, out / "scoring/product.csv")
    write(sd_req, out / "scoring/store_dept.csv")
    write(o_sc[["o_order_id", "trip_type"]], out / "scoring/order_labels.csv")
    write(weekly_truth(o_sc, li_sc, products), out / "scoring/store_dept_labels.csv")
    print(f"training {len(o_tr)} orders / {len(li_tr)} items | serving {len(o_sv)} | "
          f"scoring {len(o_sc)} orders | {len(pairs)} store-dept pairs -> {out}")


if __name__ == "__main__":
    main()
