"""Compare finished runs: phase totals, stage times and whether predictions match (M1 6).

    python tools/compare_runs.py runs/<separate_run> runs/<shared_run>
    python tools/compare_runs.py runs/*            # also writes runs/all_stages.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

FILES = {"uc03": ["store", "department", "date"], "uc08": ["o_order_id"]}


def load(run: Path):
    m = json.loads((run / "metrics.json").read_text())
    st = pd.DataFrame(m["stages"])
    st["run"] = run.name
    return m["run"], st


def same_predictions(a: Path, b: Path) -> dict:
    out = {}
    for uc, keys in FILES.items():
        for f in ("predictions.csv", "scoring_predictions.csv"):
            pa, pb = a / uc / f, b / uc / f
            if not (pa.exists() and pb.exists()):
                continue
            da = pd.read_csv(pa).sort_values(keys).reset_index(drop=True)
            db = pd.read_csv(pb).sort_values(keys).reset_index(drop=True)
            tag = f"{uc}/{f}"
            if da.shape != db.shape or not da[keys].equals(db[keys]):
                out[tag] = "DIFFERENT keys/rows"
            elif uc == "uc03":
                diff = (da["weekly_sales"] - db["weekly_sales"]).abs().max()
                out[tag] = "same" if diff < 1e-6 else f"max abs diff {diff:.6g}"
            else:
                n = int((da["trip_type"] != db["trip_type"]).sum())
                out[tag] = "same" if n == 0 else f"{n} orders differ"
    return out


def main(paths):
    runs = [Path(p) for p in paths if (Path(p) / "metrics.json").exists()]
    infos, stages = zip(*(load(r) for r in runs))
    allst = pd.concat(stages)
    pd.set_option("display.width", 200)

    work = allst[allst["kind"] == "work"]
    phase = work.pivot_table(index=["pipeline", "phase"], columns="run", values="seconds",
                             aggfunc="sum", sort=False)
    print("== work seconds by pipeline/phase ==")
    print(phase.round(2).fillna("-").to_string())
    print("\n== stage seconds ==")
    table = allst.pivot_table(index=["pipeline", "phase", "stage"], columns="run", values="seconds",
                              aggfunc="sum", sort=False)
    print(table.round(2).fillna("-").to_string())
    print()
    for i in infos:
        print(f"{i['run_id']:40} mode={i['mode']:8} workers={i['workers']} "
              f"work={i['work_seconds']}s checks_failed={i['checks_failed']}")
    if len(runs) == 2:
        print("\npredictions:", same_predictions(*runs))
    if len(runs) > 2:
        out = runs[0].parent / "all_stages.csv"
        pd.concat(pd.read_csv(r / "stages.csv") for r in runs).to_csv(out, index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
