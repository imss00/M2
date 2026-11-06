"""Stage timer and validation log written to runs/<run_id>/ (M1 5.3, 6).

Every stage carries
  pipeline : train | serve | score        (TPCx-AI training / serving / scoring)
  phase    : load | preprocess | train | predict | score | check
phase "check" is validation only and is excluded from work time.
"""
import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PHASES = ("load", "preprocess", "train", "predict", "score", "check")


class RunRecorder:
    def __init__(self, run_dir: Path, info: dict):
        self.run_dir = run_dir
        self.info = dict(info)
        self.stages: list[dict] = []
        self.checks: list[dict] = []
        self._t0 = time.perf_counter()
        self.info["started_at"] = datetime.now().isoformat(timespec="seconds")

    @contextmanager
    def stage(self, name: str, pipeline: str, phase: str):
        assert phase in PHASES, phase
        rec = {"stage": name, "pipeline": pipeline, "phase": phase,
               "kind": "check" if phase == "check" else "work"}
        t0 = time.perf_counter()
        print(f"[{pipeline}/{phase}] {name} ...", flush=True)
        try:
            yield rec              # caller may add rows_in / rows_out / cols_out
        finally:
            rec["seconds"] = round(time.perf_counter() - t0, 3)
            self.stages.append(rec)
            extra = {k: v for k, v in rec.items()
                     if k not in ("stage", "pipeline", "phase", "kind", "seconds")}
            print(f"[{pipeline}/{phase}] {name}: {rec['seconds']}s {extra or ''}", flush=True)

    def check(self, name: str, value, expected=None, ok: bool | None = None, note: str = ""):
        if ok is None and expected is not None:
            ok = value == expected
        status = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        self.checks.append({"check": name, "value": value, "expected": expected,
                            "status": status, "note": note})
        mark = {"PASS": "ok", "FAIL": "FAIL", "INFO": "--"}[status]
        print(f"  [{mark}] {name} = {value}" +
              (f" (expected {expected})" if expected is not None else ""), flush=True)

    def save(self):
        self.info["finished_at"] = datetime.now().isoformat(timespec="seconds")
        self.info["total_seconds"] = round(time.perf_counter() - self._t0, 3)
        work = [s for s in self.stages if s["kind"] == "work"]
        self.info["work_seconds"] = round(sum(s["seconds"] for s in work), 3)
        phase_seconds = defaultdict(float)
        for s in work:
            phase_seconds[f"{s['pipeline']}/{s['phase']}"] += s["seconds"]
        self.info["phase_seconds"] = {k: round(v, 3) for k, v in phase_seconds.items()}
        n_fail = sum(c["status"] == "FAIL" for c in self.checks)
        self.info["checks_failed"] = n_fail
        shapes = [{"stage": s["stage"], "rows_out": s.get("rows_out"), "cols_out": s.get("cols_out")}
                  for s in self.stages if "rows_out" in s]

        (self.run_dir / "metrics.json").write_text(json.dumps(
            {"run": self.info, "stages": self.stages, "shapes": shapes}, indent=2, default=str))
        (self.run_dir / "validation.json").write_text(json.dumps(
            {"run_id": self.info.get("run_id"), "failed": n_fail, "checks": self.checks},
            indent=2, default=str))
        # Flat file so several runs (several people in M3) can be concatenated into one table.
        keys = ["run_id", "sf", "mode", "workers", "driver_memory"]
        with open(self.run_dir / "stages.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys + ["stage", "pipeline", "phase", "kind", "seconds", "rows_out", "cols_out"])
            for s in self.stages:
                w.writerow([self.info.get(k) for k in keys] +
                           [s["stage"], s["pipeline"], s["phase"], s["kind"], s["seconds"],
                            s.get("rows_out", ""), s.get("cols_out", "")])
        return n_fail
