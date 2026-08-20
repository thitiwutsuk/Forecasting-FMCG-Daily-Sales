"""Hypothesis-driven features (Phase 4).

Each feature only uses information available at the end of the current week — the
same temporal cutoff already used by `price_unit`/`promotion_flag` in the base weekly
table, since we're always predicting `target_next_week` from a week that has already
fully happened. None of these look into future weeks.
"""

import pandas as pd

GROUP_KEYS = ["sku", "channel", "region"]


def add_price_index(df: pd.DataFrame) -> pd.DataFrame:
    """SKU price relative to its category's average price in the same week (all channels/regions)."""
    df = df.copy()
    category_avg_price = df.groupby(["category", "week"])["price_unit"].transform("mean")
    df["price_index"] = df["price_unit"] / category_avg_price
    return df


def add_promo_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Weeks since the last promotion in this sku x channel x region series (0 = promo this week)."""
    df = df.sort_values(GROUP_KEYS + ["week"]).copy()

    def _recency(promo_flags: pd.Series) -> pd.Series:
        recency = []
        weeks_since = None
        for flag in promo_flags:
            if flag == 1:
                weeks_since = 0
            elif weeks_since is not None:
                weeks_since += 1
            recency.append(weeks_since)
        return pd.Series(recency, index=promo_flags.index)

    df["promo_recency"] = df.groupby(GROUP_KEYS)["promotion_flag"].transform(_recency)
    return df


def add_rolling_promo_rate(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Share of the trailing `window` weeks (inclusive of this week) that had a promotion."""
    df = df.sort_values(GROUP_KEYS + ["week"]).copy()
    df["rolling_promo_rate"] = df.groupby(GROUP_KEYS)["promotion_flag"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    return df


def add_cross_channel_demand_share(df: pd.DataFrame) -> pd.DataFrame:
    """This channel's share of the SKU's total units_sold across channels, same region x week."""
    df = df.copy()
    total_across_channels = df.groupby(["sku", "region", "week"])["units_sold"].transform("sum")
    df["cross_channel_demand_share"] = df["units_sold"] / total_across_channels.replace(0, pd.NA)
    return df


def build_hypothesis_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_price_index(df)
    df = add_promo_recency(df)
    df = add_rolling_promo_rate(df)
    df = add_cross_channel_demand_share(df)
    return df
