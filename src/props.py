"""
Prop registry: the single source of truth for every pitcher prop this pipeline
supports, so the pipeline is parameterized by prop rather than K-hardcoded.

Design: docs/design/specs/2026-06-29-prop-expansion-walks-earned-runs-design.md
("Design — a prop registry, not copy-paste").

Why a registry here rather than constants scattered across baseline_model /
game_logs / tiering / refresh: the pipeline has five stages that each need to
know SOMETHING about the prop (label column, Underdog stat key, threshold
range, feature prefix, label source).  A single Prop dataclass centralises
that coupling so adding a new prop is one block of config, not five
scattered edits.

Line source note (2026-08 migration): `underdog_stat` holds the Underdog
Fantasy stat KEY (e.g. "runs_allowed"), not its display name ("Earned Runs
Allowed") -- see src/data/underdog_lines.py's module docstring for the
key-vs-display-name trap. This field was previously `prizepicks_stat_type`
and held a PrizePicks display string; PrizePicks' endpoint is now
permanently 403'd (see docs/data_sources.md).

Label sources:
- "statcast": count(events == statcast_event) per pitcher-game, the same path
  game_logs.aggregate_pitcher_games already uses for strikeouts.
- "statsapi_boxscore": earned runs from MLB StatsAPI pitching boxscore
  (earnedRuns per pitcher per game_pk).  Statcast does not carry the
  earned/unearned distinction; the official scoring call lives in the boxscore.

Backward compatibility guarantee: the DEFAULT_PROP is "strikeouts", so every
caller that does not pass a prop= argument continues to resolve to the
strikeout pipeline with the same partition paths and model artifact as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Prop:
    """Configuration for one pitcher prop end-to-end."""

    # Unique key used in partition paths (prop={key}/), model filenames
    # ({key}_model.joblib), and API surfaces.
    key: str

    # The exact stat key Underdog Fantasy uses in its over_under_lines payload
    # (over_under.appearance_stat.stat) -- NOT the display name.
    underdog_stat: str

    # Column name for this prop's label in per-pitcher-game tables.
    label_column: str

    # "statcast"          -> count(events == statcast_event) via game_logs
    # "statsapi_boxscore" -> earnedRuns from MLB StatsAPI boxscore ingestion
    label_source: str

    # For statcast label_source only: the Statcast event string(s) to count.
    # None for non-statcast props (e.g. earned_runs).
    statcast_event: Optional[str] = None

    # Prefix for the rolling rate features (bb_rate_*, er_*, k_rate_*).
    rate_feature_prefix: str = "k_rate"

    # Integer thresholds to sweep: range(1, 11) for Ks, range(0, 6) for walks/ER.
    thresholds: range = field(default_factory=lambda: range(1, 11))

    # Count family for model selection: "poisson" (escalates to NB on evidence).
    count_family: str = "poisson"

    # v1 regressor allowlist for this prop's model.  None means use the
    # strikeout baseline_model defaults (CORE_PITCHER_FORM_COLUMNS + IMPUTE_COLUMNS).
    regressor_columns: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROP_REGISTRY: dict[str, Prop] = {
    "strikeouts": Prop(
        key="strikeouts",
        underdog_stat="strikeouts",
        label_column="strikeouts",
        label_source="statcast",
        statcast_event="strikeout",
        rate_feature_prefix="k_rate",
        thresholds=range(1, 11),
        count_family="poisson",
    ),
    "walks": Prop(
        key="walks",
        underdog_stat="walks_allowed",
        label_column="walks",
        label_source="statcast",
        statcast_event="walk",
        rate_feature_prefix="bb_rate",
        thresholds=range(0, 6),
        count_family="poisson",
    ),
    "earned_runs": Prop(
        key="earned_runs",
        underdog_stat="runs_allowed",
        label_column="earned_runs",
        label_source="statsapi_boxscore",
        statcast_event=None,
        rate_feature_prefix="er",
        thresholds=range(0, 6),
        count_family="poisson",
    ),
}

# The default prop for every pipeline stage that hasn't been updated to
# require an explicit prop= argument yet.  Existing callers omitting prop=
# continue to produce strikeout outputs under the same partition paths.
DEFAULT_PROP = "strikeouts"


def get_prop(key: str) -> Prop:
    """Look up a prop by key; raises KeyError with a helpful message if unknown."""
    if key not in PROP_REGISTRY:
        known = ", ".join(sorted(PROP_REGISTRY))
        raise KeyError(f"Unknown prop key {key!r}. Known props: {known}")
    return PROP_REGISTRY[key]
