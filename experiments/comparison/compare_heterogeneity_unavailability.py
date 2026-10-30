from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HETEROGENEITY_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "heterogeneity"
)

UNAVAILABILITY_RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "unavailability"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "comparison"
    / "heterogeneity_vs_unavailability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Scenario mapping
# ---------------------------------------------------------------------

RETENTION_MAP = {
    "h5_retention_100": 100,
    "h5_retention_75": 75,
    "h5_retention_50": 50,
    "h5_retention_25": 25,
    "h5_retention_0": 0,
}


# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------

heterogeneity_exact = pd.read_csv(
    HETEROGENEITY_RESULT_DIR
    / "heterogeneity_exact_statistics.csv"
)

heterogeneity_dp = pd.read_csv(
    HETEROGENEITY_RESULT_DIR
    / "heterogeneity_dp_summary.csv"
)

unavailability_exact = pd.read_csv(
    UNAVAILABILITY_RESULT_DIR
    / "unavailability_exact_shift.csv"
)

unavailability_dp = pd.read_csv(
    UNAVAILABILITY_RESULT_DIR
    / "unavailability_dp_summary.csv"
)


# ---------------------------------------------------------------------
# Add numerical retention level
# ---------------------------------------------------------------------

for df in [
    heterogeneity_exact,
    heterogeneity_dp,
    unavailability_exact,
    unavailability_dp,
]:
    df["h5_retention_percent"] = (
        df["scenario"].map(
            RETENTION_MAP
        )
    )


# ---------------------------------------------------------------------
# 1. Consolidated exact-statistic table
# ---------------------------------------------------------------------

exact_comparison = pd.merge(
    heterogeneity_exact[
        [
            "scenario",
            "h5_retention_percent",
            "q006_global_count",
            "q007_global_count",
            "q007_global_sum",
            "q007_global_mean",
        ]
    ],
    unavailability_exact[
        [
            "scenario",
            "q006_exact_count",
            "q006_availability_error",
            "q006_availability_error_percent",
            "q007_exact_count",
            "q007_exact_sum",
            "q007_exact_mean",
            "q007_availability_error",
            "q007_availability_error_percent",
        ]
    ],
    on="scenario",
    how="inner",
)

exact_comparison = exact_comparison.rename(
    columns={
        "q006_global_count":
            "redistribution_q006_count",

        "q007_global_count":
            "redistribution_q007_count",

        "q007_global_sum":
            "redistribution_q007_sum",

        "q007_global_mean":
            "redistribution_q007_mean",

        "q006_exact_count":
            "unavailability_q006_count",

        "q007_exact_count":
            "unavailability_q007_count",

        "q007_exact_sum":
            "unavailability_q007_sum",

        "q007_exact_mean":
            "unavailability_q007_mean",
    }
)

exact_comparison = (
    exact_comparison
    .sort_values(
        "h5_retention_percent",
        ascending=False,
    )
)


# ---------------------------------------------------------------------
# 2. Consolidated DP table
# ---------------------------------------------------------------------

heterogeneity_dp_selected = (
    heterogeneity_dp[
        [
            "scenario",
            "h5_retention_percent",
            "query_id",
            "epsilon",
            "exact_value",
            "mae",
            "rmse",
        ]
    ]
    .rename(
        columns={
            "exact_value":
                "redistribution_exact_value",

            "mae":
                "redistribution_dp_mae",

            "rmse":
                "redistribution_dp_rmse",
        }
    )
)


unavailability_dp_selected = (
    unavailability_dp[
        [
            "scenario",
            "query_id",
            "epsilon",
            "full_data_value",
            "scenario_exact_value",
            "availability_error",
            "availability_absolute_error",
            "dp_mae",
            "dp_rmse",
            "total_mae",
            "total_rmse",
        ]
    ]
    .rename(
        columns={
            "dp_mae":
                "unavailability_dp_mae",

            "dp_rmse":
                "unavailability_dp_rmse",

            "total_mae":
                "unavailability_total_mae",

            "total_rmse":
                "unavailability_total_rmse",
        }
    )
)


dp_comparison = pd.merge(
    heterogeneity_dp_selected,
    unavailability_dp_selected,
    on=[
        "scenario",
        "query_id",
        "epsilon",
    ],
    how="inner",
)

dp_comparison = (
    dp_comparison
    .sort_values(
        [
            "query_id",
            "epsilon",
            "h5_retention_percent",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )
)


# ---------------------------------------------------------------------
# Save comparison tables
# ---------------------------------------------------------------------

exact_output = (
    OUTPUT_DIR
    / "exact_redistribution_vs_unavailability.csv"
)

dp_output = (
    OUTPUT_DIR
    / "dp_redistribution_vs_unavailability.csv"
)

exact_comparison.to_csv(
    exact_output,
    index=False,
)

dp_comparison.to_csv(
    dp_output,
    index=False,
)


# ---------------------------------------------------------------------
# Helper for plot ordering
# ---------------------------------------------------------------------

def sorted_by_retention(df):
    return df.sort_values(
        "h5_retention_percent"
    )


# ---------------------------------------------------------------------
# Figure 1
# Q006 exact count:
# redistribution vs actual unavailability
# ---------------------------------------------------------------------

plot_df = sorted_by_retention(
    exact_comparison
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    plot_df["h5_retention_percent"],
    plot_df["redistribution_q006_count"],
    marker="o",
    label="Redistribution",
)

plt.plot(
    plot_df["h5_retention_percent"],
    plot_df["unavailability_q006_count"],
    marker="s",
    label="Unavailability",
)

plt.xlabel(
    "H5 target-subgroup availability (%)"
)

plt.ylabel(
    "Global Q006 patient count"
)

plt.title(
    "Q006: Site Redistribution vs Data Unavailability"
)

plt.xticks(
    [0, 25, 50, 75, 100]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "q006_exact_redistribution_vs_unavailability.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# Figure 2
# Q007 exact bounded mean:
# redistribution vs actual unavailability
# ---------------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    plot_df["h5_retention_percent"],
    plot_df["redistribution_q007_mean"],
    marker="o",
    label="Redistribution",
)

plt.plot(
    plot_df["h5_retention_percent"],
    plot_df["unavailability_q007_mean"],
    marker="s",
    label="Unavailability",
)

plt.xlabel(
    "H5 target-subgroup retention (%)"
)

plt.ylabel(
    "Global bounded mean LOS (days)"
)

plt.title(
    "Q007: Site Redistribution vs Data Unavailability"
)

plt.xticks(
    [0, 25, 50, 75, 100]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "q007_exact_redistribution_vs_unavailability.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# Figure 3
# Q007 DP-only MAE at epsilon = 1
# ---------------------------------------------------------------------

q007_eps1 = dp_comparison[
    (dp_comparison["query_id"] == "Q007")
    &
    (dp_comparison["epsilon"] == 1.0)
].copy()

q007_eps1 = sorted_by_retention(
    q007_eps1
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    q007_eps1["h5_retention_percent"],
    q007_eps1["redistribution_dp_mae"],
    marker="o",
    label="Redistribution",
)

plt.plot(
    q007_eps1["h5_retention_percent"],
    q007_eps1["unavailability_dp_mae"],
    marker="s",
    label="Unavailability",
)

plt.xlabel(
    "H5 target-subgroup retention (%)"
)

plt.ylabel(
    "DP MAE (days)"
)

plt.title(
    "Q007 DP Utility under Site Redistribution and Data Unavailability "
    "(ε = 1)"
)

plt.xticks(
    [0, 25, 50, 75, 100]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "q007_dp_mae_eps1_redistribution_vs_unavailability.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# Figure 4
# Q006 total error at epsilon = 1
#
# Redistribution has no availability shift, so its DP MAE represents
# the complete error relative to the unchanged full-data statistic.
# ---------------------------------------------------------------------

q006_eps1 = dp_comparison[
    (dp_comparison["query_id"] == "Q006")
    &
    (dp_comparison["epsilon"] == 1.0)
].copy()

q006_eps1 = sorted_by_retention(
    q006_eps1
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    q006_eps1["h5_retention_percent"],
    q006_eps1["redistribution_dp_mae"],
    marker="o",
    label="Redistribution: total MAE",
)

plt.plot(
    q006_eps1["h5_retention_percent"],
    q006_eps1["unavailability_total_mae"],
    marker="s",
    label="Unavailability: total MAE",
)

plt.xlabel(
    "H5 target-subgroup retention (%)"
)

plt.ylabel(
    "MAE in patient count"
)

plt.title(
    "Q006 End-to-End Error under Redistribution and Unavailability "
    "(ε = 1)"
)

plt.xticks(
    [0, 25, 50, 75, 100]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "q006_total_mae_eps1_redistribution_vs_unavailability.png",
    dpi=300,
)

plt.close()


# ---------------------------------------------------------------------
# Print compact tables
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("EXACT REDISTRIBUTION VS UNAVAILABILITY")
print("=" * 78)

print(
    exact_comparison[
        [
            "h5_retention_percent",
            "redistribution_q006_count",
            "unavailability_q006_count",
            "redistribution_q007_mean",
            "unavailability_q007_mean",
            "q006_availability_error_percent",
            "q007_availability_error_percent",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 78)
print("Q007 DP MAE COMPARISON AT EPSILON = 1")
print("=" * 78)

print(
    q007_eps1[
        [
            "h5_retention_percent",
            "redistribution_dp_mae",
            "unavailability_dp_mae",
            "unavailability_total_mae",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 78)
print("Q006 TOTAL MAE COMPARISON AT EPSILON = 1")
print("=" * 78)

print(
    q006_eps1[
        [
            "h5_retention_percent",
            "redistribution_dp_mae",
            "unavailability_dp_mae",
            "unavailability_total_mae",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Saved comparison tables:")
print(exact_output)
print(dp_output)

print()
print("Saved figures:")
print(
    OUTPUT_DIR
    / "q006_exact_redistribution_vs_unavailability.png"
)
print(
    OUTPUT_DIR
    / "q007_exact_redistribution_vs_unavailability.png"
)
print(
    OUTPUT_DIR
    / "q007_dp_mae_eps1_redistribution_vs_unavailability.png"
)
print(
    OUTPUT_DIR
    / "q006_total_mae_eps1_redistribution_vs_unavailability.png"
)