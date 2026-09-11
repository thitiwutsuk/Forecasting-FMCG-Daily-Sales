# Runs the official R TwoWayFEWeights diagnostic from a clean Windows checkout.
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$panelExport = Join-Path $repoRoot "reports\promotion_panel_export.csv"
$volumeName = "twfe_weights_r_lib"

Push-Location $repoRoot
try {
    & python "analysis\export_promotion_panel.py"
    if ($LASTEXITCODE -ne 0) { throw "Panel export failed with exit code $LASTEXITCODE" }

    & docker volume create $volumeName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker volume creation failed with exit code $LASTEXITCODE" }

    & docker run --rm `
        -v "${repoRoot}:/work" `
        -v "${volumeName}:/usr/local/lib/R/site-library" `
        -w /work `
        rocker/r-ver:4.4.1 `
        bash -c "apt-get update -qq && apt-get install -y -qq zlib1g-dev libcurl4-openssl-dev libssl-dev libxml2-dev >/dev/null && Rscript analysis/r/promotion_twfe_weights.R"
    if ($LASTEXITCODE -ne 0) { throw "R diagnostic failed with exit code $LASTEXITCODE" }
}
finally {
    Remove-Item -LiteralPath $panelExport -Force -ErrorAction SilentlyContinue
    & docker volume rm $volumeName 2>$null | Out-Null
    Pop-Location
}
