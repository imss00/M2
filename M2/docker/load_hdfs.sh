#!/bin/bash
# Inside the client container: put generated data on HDFS and prepare Spark on YARN.
#   docker compose exec client docker/load_hdfs.sh [SF]
set -euo pipefail
SF="${1:-1}"
SRC="/workspace/data/sf${SF}"
DST="/tpcxai/sf${SF}"

echo "waiting for 2 live DataNodes ..."
until [ "$(hdfs dfsadmin -report 2>/dev/null | grep -c '^Name:')" -ge 2 ]; do sleep 3; done
hdfs dfs -mkdir -p "$DST" /user/hive/warehouse /user/root /spark

echo "uploading $SRC -> hdfs://namenode:9000$DST"
time hdfs dfs -put -f "$SRC/training" "$SRC/serving" "$SRC/scoring" "$DST/"

# Spark jars for the YARN containers, uploaded once instead of on every run
J="$(python3 -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')/jars"
(cd "$J" && zip -q -r /tmp/spark-jars.zip .)
hdfs dfs -put -f /tmp/spark-jars.zip /spark/spark-jars.zip

hdfs dfs -du -h "$DST"
hdfs dfsadmin -report | grep -E "^Name:|DFS Used:" | head -6
echo "done. block placement: hdfs fsck $DST -files -blocks -locations | head"
