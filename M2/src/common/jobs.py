"""Run a model step (src/jobs/*) in a child Python process.

The interpreter is cfg.model_python (e.g. a Python 3.9 env with the kit's pinned versions) or the
current one. The child gets <job>.json and writes <job>.result.json. Input files on HDFS are first
copied to the local run folder with `hdfs dfs -get`, so the child only reads local Parquet files.
"""
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import PROJECT_ROOT, RunConfig


def interpreter(cfg: RunConfig) -> str:
    return cfg.model_python or sys.executable


def table_files(table_df) -> list[str]:
    """Data files behind a Hive table DataFrame: local paths, or hdfs:// URIs on a cluster."""
    out = []
    for f in table_df.inputFiles():
        u = urlparse(f)
        out.append(unquote(u.path) if u.scheme in ("", "file") else f)
    return sorted(out)


def localize(files: list[str], dest: Path) -> list[str]:
    """Copy hdfs:// files into dest (one `hdfs dfs -get` per directory); local paths pass through."""
    remote = [f for f in files if f.startswith("hdfs://")]
    if not remote:
        return files
    by_dir = defaultdict(list)
    for f in remote:
        by_dir[f.rsplit("/", 1)[0]].append(f)
    local = [f for f in files if not f.startswith("hdfs://")]
    for i, (d, fs) in enumerate(sorted(by_dir.items())):
        target = dest / f"part{i:02d}_{d.rsplit('/', 1)[-1]}"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        subprocess.run(["hdfs", "dfs", "-get", *fs, str(target)], check=True)
        local += [str(target / f.rsplit("/", 1)[-1]) for f in fs]
    return sorted(local)


def run_job(cfg: RunConfig, module: str, cmd: str, job: dict, job_file: Path) -> dict:
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job = dict(job)
    for key, value in job.items():
        if isinstance(value, list) and value and isinstance(value[0], str) and "://" in value[0]:
            job[key] = localize(value, job_file.parent / f"_input_{job_file.stem}_{key}")
    job_file.write_text(json.dumps(job))
    subprocess.run([interpreter(cfg), "-W", "ignore", "-m", f"src.jobs.{module}", cmd, str(job_file)],
                   cwd=PROJECT_ROOT, check=True)
    for key in job:                         # drop copied inputs; results stay
        tmp = job_file.parent / f"_input_{job_file.stem}_{key}"
        if tmp.exists():
            shutil.rmtree(tmp)
    return json.loads(job_file.with_suffix(".result.json").read_text())


def library_versions(cfg: RunConfig) -> dict:
    """Versions of the model libraries in the job interpreter (recorded in metrics.json)."""
    code = ("import json,sys,pandas,numpy,statsmodels,xgboost;print(json.dumps({'python':sys.version.split()[0],"
            "'pandas':pandas.__version__,'numpy':numpy.__version__,'statsmodels':statsmodels.__version__,"
            "'xgboost':xgboost.__version__}))")
    out = subprocess.run([interpreter(cfg), "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])
