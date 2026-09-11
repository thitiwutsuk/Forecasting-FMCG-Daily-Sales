#!/usr/bin/env bash
# Runs the TwoWayFEWeights negative-weights diagnostic (analysis/r/promotion_twfe_weights.R)
# in a disposable rocker/r-ver container -- avoids installing R locally for one script.
# Requires: Docker, and reports/promotion_panel_export.csv already generated (see
# analysis/promotion_robustness.py or the Phase 8 robustness notebook).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

docker volume create twfe_weights_r_lib >/dev/null

docker run --rm \
  -v "$(pwd):/work" \
  -v twfe_weights_r_lib:/usr/local/lib/R/site-library \
  -w /work \
  rocker/r-ver:4.4.1 \
  bash -c "apt-get update -qq && apt-get install -y -qq zlib1g-dev libcurl4-openssl-dev libssl-dev libxml2-dev >/dev/null && Rscript analysis/r/promotion_twfe_weights.R"
