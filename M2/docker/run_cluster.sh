#!/bin/bash
# Inside the client container: run the pipeline on YARN in both modes with the kit's model versions.
#   docker compose exec client docker/run_cluster.sh [SF] [executors] [cores] [memory]
set -euo pipefail
SF="${1:-1}"; EX="${2:-2}"; CORES="${3:-3}"; MEM="${4:-4g}"
COMMON="--data-dir hdfs://namenode:9000/tpcxai/sf${SF} --sf ${SF} --master yarn \
  --executors ${EX} --executor-cores ${CORES} --executor-memory ${MEM} --driver-memory 3g \
  --model-python /opt/kitenv/bin/python"
TAG="sf${SF}_yarn_e${EX}c${CORES}"

python3 run.py $COMMON --mode shared --reingest --run-id "${TAG}_kit_shared"   2>&1 | tee "runs/${TAG}_kit_shared.log"
python3 run.py $COMMON --mode separate          --run-id "${TAG}_kit_separate" 2>&1 | tee "runs/${TAG}_kit_separate.log"
python3 tools/compare_runs.py "runs/${TAG}_kit_separate" "runs/${TAG}_kit_shared"
