#!/bin/bash
# Generate TPCx-AI data for UC03/UC08 with the kit's PDGF, on the host (macOS or Linux).
#   docker/gen_data.sh <path to tpcx-ai-v2.0.0> [SF]      -> data/sf<SF>/{training,serving,scoring}
# PDGF needs x86 + Java 8, so it runs in an amd64 Java 8 container (Rosetta on Apple Silicon).
# Running this accepts the PDGF license (lib/pdgf/LICENSE.txt), like the kit's first start.
# Seeds and options are those of the kit driver (driver/tpcxai-driver/data_generation.py).
set -euo pipefail
KIT="$(cd "${1:?usage: docker/gen_data.sh <kit dir> [SF]}" && pwd)"
SF="${2:-1}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/sf${SF}"
mkdir -p "$OUT"
echo "kit=$KIT  SF=$SF  -> $OUT"

docker run --rm --platform linux/amd64 \
  -v "$KIT":/kit:ro -v "$OUT":/out -e SF="$SF" \
  eclipse-temurin:8-jre bash -c '
set -euo pipefail
# PDGF loads plugins with native GUI libraries (FaceImage/scalismo); the slim JRE image lacks them
apt-get update -qq && apt-get install -y -qq --no-install-recommends libx11-6 libxext6 libxrender1 libxtst6 libxi6 libgl1 libfontconfig1 >/dev/null
mkdir -p /tmp/k/lib && cp -r /kit/data-gen /tmp/k/ && cp -r /kit/lib/pdgf /tmp/k/lib/   # kit stays read-only
cd /tmp/k/data-gen
O="-l config/tpcxai-schema.xml -l config/tpcxai-generation.xml -ns"
T="order lineitem product store_dept"
run() { printf "\nYES\n" | java -Djava.awt.headless=true -Xmx4g -jar ../lib/pdgf/pdgf.jar $O "$@" | tail -3; }
mkdir -p /out/training /out/serving /out/scoring
echo "training ..."; run -sf $SF -sp MY_SEED 1234 -sp includeLabels 1.0 -sp TTVF 1.0 -s $T -output "\"/out/training/\""
echo "serving ...";  run -sf $SF -sp SF_TRAINING $SF -sp MY_SEED 1235 -sp includeLabels 0.0 -sp TTVF 0.1 -s $T -output "\"/out/serving/\""
echo "scoring ...";  run -sf 1 -sp SF_TRAINING $SF -sp MY_SEED $((1234 + ${SF%.*})) -sp includeLabels 2.0 -sp TTVF 0.1 -s $T order_weekly_sales -output "\"/out/scoring/\""
'
wc -l "$OUT"/*/*.csv
