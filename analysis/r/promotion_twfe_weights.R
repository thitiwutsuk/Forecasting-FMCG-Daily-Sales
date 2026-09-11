#!/usr/bin/env Rscript
# Negative-weights diagnostic for the Phase 8 promotion TWFE estimate
# (de Chaisemartin & D'Haultfoeuille 2020, AER 110(9), 2964-2996).
#
# Run inside the rocker/r-ver container set up by analysis/r/run_twfe_weights_docker.sh
# (or ...ps1 on Windows) -- installs the expected TwoWayFEWeights version, reads the exported
# panel, and writes the weight-decomposition summary back out as CSV/JSON.

# Pre-built Ubuntu-jammy binaries (Posit Package Manager) instead of CRAN source --
# avoids compiling fixest/vroom/etc. from source, which takes a very long time.
options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"))

expected_twoway_version <- "2.1.0"
if (
  !requireNamespace("TwoWayFEWeights", quietly = TRUE) ||
  as.character(packageVersion("TwoWayFEWeights")) != expected_twoway_version
) {
  install.packages("TwoWayFEWeights")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite")
}

library(TwoWayFEWeights)
library(jsonlite)

twoway_version <- as.character(packageVersion("TwoWayFEWeights"))
if (twoway_version != expected_twoway_version) {
  stop(paste(
    "Expected TwoWayFEWeights", expected_twoway_version,
    "but the configured repository installed", twoway_version
  ))
}

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
  n_treated_cells = weights_result$tot_cells,
  n_positive_weights = weights_result$nr_plus,
  n_negative_weights = weights_result$nr_minus,
  sum_positive_weights = weights_result$sum_plus,
  sum_negative_weights = weights_result$sum_minus,
  sensitivity_to_zero_att = weights_result$sensibility,
  type = weights_result$type,
  twowayfeweights_version = twoway_version,
  r_version = R.version.string
)

writeLines(toJSON(out, auto_unbox = TRUE, digits = 10), "/work/reports/promotion_twfe_weights_result.json")
write.csv(weights_result$dat_result, "/work/reports/promotion_twfe_weights_detail.csv", row.names = FALSE)

cat("\nDone. Wrote reports/promotion_twfe_weights_result.json and _detail.csv\n")
