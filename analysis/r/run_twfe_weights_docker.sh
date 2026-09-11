#!/usr/bin/env bash
# Runs the TwoWayFEWeights negative-weights diagnostic (analysis/r/promotion_twfe_weights.R)
# in a disposable rocker/r-ver container -- avoids installing R locally for one script.
# Requires Docker and Python dependencies from requirements.txt. The panel export
# and Docker package-cache volume are removed automatically when the run finishes.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

panel_export="reports/promotion_panel_export.csv"
volume_name="twfe_weights_r_lib"
cleanup() {
  rm -f -- "$panel_export"
  docker volume rm "$volume_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python analysis/export_promotion_panel.py
docker volume create "$volume_name" >/dev/null

docker run --rm \
  -v "$(pwd):/work" \
  -v "${volume_name}:/usr/local/lib/R/site-library" \
  -w /work \
  rocker/r-ver:4.4.1 \
  bash -c "apt-get update -qq && apt-get install -y -qq zlib1g-dev libcurl4-openssl-dev libssl-dev libxml2-dev >/dev/null && Rscript analysis/r/promotion_twfe_weights.R"
