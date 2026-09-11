"""Export the validated promotion panel consumed by the R weight diagnostic."""

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from causal.promotion_robustness import build_group_panel, export_group_panel_csv


def main() -> None:
    source = REPO_ROOT / "data" / "processed" / "weekly_features.csv"
    destination = REPO_ROOT / "reports" / "promotion_panel_export.csv"
    panel = build_group_panel(pd.read_csv(source, parse_dates=["week"]))
    export_group_panel_csv(panel, str(destination))
    print(
        f"Exported {len(panel):,} rows and {panel['group_id'].nunique():,} groups "
        f"to {destination}"
    )


if __name__ == "__main__":
    main()
