#!/bin/bash
# Inside the client container: Spark History Server over the saved event logs of finished runs,
# so the Spark UI (Jobs, Stages, Executors) can be browsed after a run at http://localhost:4040
#   docker compose exec client docker/history_server.sh [run-name-pattern]    (Ctrl+C to stop)
set -euo pipefail
PATTERN="${1:-sf1_yarn_*}"
LOGS=/tmp/spark-history
rm -rf "$LOGS" && mkdir -p "$LOGS"
for d in /workspace/runs/${PATTERN}/spark-events/*; do
  [ -e "$d" ] && cp -r "$d" "$LOGS/" && echo "event log: $d"
done
SPARK_HOME="$(python3 -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')"
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=file://$LOGS -Dspark.history.ui.port=4040"
echo "open http://localhost:4040  (Ctrl+C to stop)"
exec "$SPARK_HOME/bin/spark-class" org.apache.spark.deploy.history.HistoryServer
