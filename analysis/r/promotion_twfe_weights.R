#!/usr/bin/env Rscript
# Negative-weights diagnostic for the Phase 8 promotion TWFE estimate
# (de Chaisemartin & D'Haultfoeuille 2020, AER 110(9), 2964-2996).
#
# Run inside the rocker/r-ver container set up by analysis/r/run_twfe_weights_docker.sh
# (or ...ps1 on Windows) -- installs TwoWayFEWeights from CRAN, reads the exported
# panel, and writes the weight-decomposition summary back out as CSV/JSON.

# Pre-built Ubuntu-jammy binaries (Posit Package Manager) instead of CRAN source --
# avoids compiling fixest/vroom/etc. from source, which takes a very long time.
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"))

if (!requireNamespace("TwoWayFEWeights", quietly = TRUE)) {
  install.packages("TwoWayFEWeights")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

library(TwoWayFEWeights)
library(jsonlite)

panel <- read.csv("/work/reports/promotion_panel_export.csv")

weights_result <- twowayfeweights(
  panel,
  Y = "log_units",
  G = "group_id",
  T = "time_id",
  D = "promotion_flag",
  type = "feTR",
  controls = "price_unit",
  summary_measures = TRUE
)

print(weights_result)

out <- list(
  beta = weights_result$beta,
  n_obs = nrow(panel),
  n_groups = length(unique(panel$group_id)),
  weights_summary = as.list(weights_result)
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = 10), "/work/reports/promotion_twfe_weights_result.json")
write.csv(weights_result$weights, "/work/reports/promotion_twfe_weights_detail.csv", row.names = FALSE)

cat("\nDone. Wrote reports/promotion_twfe_weights_result.json and _detail.csv\n")
