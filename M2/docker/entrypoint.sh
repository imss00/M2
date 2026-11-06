#!/bin/bash
# Role of this container: namenode | resourcemanager | worker | client
set -e
role="${1:-client}"
case "$role" in
  namenode)
    if [ ! -d /hadoop/dfs/name/current ]; then
      hdfs namenode -format -nonInteractive -force
    fi
    exec hdfs namenode
    ;;
  resourcemanager)
    exec yarn resourcemanager
    ;;
  worker)
    # one worker = one HDFS DataNode + one YARN NodeManager (data and compute on the same node)
    hdfs datanode &
    exec yarn nodemanager
    ;;
  client)
    exec sleep infinity
    ;;
  *)
    exec "$@"
    ;;
esac
