"""
Create paper/notebook-ready CDP vs HE+DP equivalence outputs.

Inputs
------
results/he/equivalence/q006_cdp_vs_he_dp_summary.csv
results/he/equivalence/q007_cdp_vs_he_dp_summary.csv

Outputs
-------
results/he/equivalence/table_cdp_he_dp_equivalence.csv
results/he/equivalence/q007_he_induced_difference.png
results/he/equivalence/q007_he_induced_difference.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(
    "results/he/equivalence"
)

Q006_FILE = (
    RESULTS_DIR
    / "q006_cdp_vs_he_dp_summary.csv"
)

Q007_FILE = (
    RESULTS_DIR
    / "q007_cdp_vs_he_dp_summary.csv"
)

TABLE_FILE = (
    RESULTS_DIR
    / "table_cdp_he_dp_equivalence.csv"
)

FIGURE_PNG = (
    RESULTS_DIR
    / "q007_he_induced_difference.png"
)

FIGURE_PDF = (
    RESULTS_DIR
    / "q007_he_induced_difference.pdf"
)

HE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------

q006 = pd.read_csv(
    Q006_FILE
)

q007 = pd.read_csv(
    Q007_FILE
)


# ---------------------------------------------------------------------
# Build compact equivalence table
# ---------------------------------------------------------------------

q006_table = pd.DataFrame(
    {
        "query": "Q006",
        "epsilon": q006["epsilon"],
        "cdp_mae": q006["cdp_mae"],
        "he_dp_mae": q006["he_dp_mae"],
        "cdp_relative_error_percent": (
            q006[
                "cdp_relative_error_percent"
            ]
        ),
        "he_dp_relative_error_percent": (
            q006[
                "he_dp_relative_error_percent"
            ]
        ),
        "paired_max_absolute_difference": (
            q006[
                "paired_max_absolute_difference"
            ]
        ),
        "equivalence_rate_percent": (
            q006[
                "exact_match_rate_percent"
            ]
        ),
        "equivalence_definition": (
            "exact match"
        ),
    }
)

q007_table = pd.DataFrame(
    {
        "query": "Q007",
        "epsilon": q007["epsilon"],
        "cdp_mae": q007["cdp_mae"],
        "he_dp_mae": q007["he_dp_mae"],
        "cdp_relative_error_percent": (
            q007[
                "cdp_relative_error_percent"
            ]
        ),
        "he_dp_relative_error_percent": (
            q007[
                "he_dp_relative_error_percent"
            ]
        ),
        "paired_max_absolute_difference": (
            q007[
                "paired_max_absolute_difference"
            ]
        ),
        "equivalence_rate_percent": (
            q007[
                "within_tolerance_rate_percent"
            ]
        ),
        "equivalence_definition": (
            "within 1e-4 days"
        ),
    }
)

combined = pd.concat(
    [
        q006_table,
        q007_table,
    ],
    ignore_index=True,
)

combined.to_csv(
    TABLE_FILE,
    index=False,
)


# ---------------------------------------------------------------------
# Print compact table
# ---------------------------------------------------------------------

print()
print("=" * 88)
print("CDP VS HE+DP UTILITY-EQUIVALENCE TABLE")
print("=" * 88)

display_table = combined.copy()

display_table[
    "cdp_mae"
] = display_table[
    "cdp_mae"
].map(
    lambda x: f"{x:.6f}"
)

display_table[
    "he_dp_mae"
] = display_table[
    "he_dp_mae"
].map(
    lambda x: f"{x:.6f}"
)

display_table[
    "cdp_relative_error_percent"
] = display_table[
    "cdp_relative_error_percent"
].map(
    lambda x: f"{x:.6f}"
)

display_table[
    "he_dp_relative_error_percent"
] = display_table[
    "he_dp_relative_error_percent"
].map(
    lambda x: f"{x:.6f}"
)

display_table[
    "paired_max_absolute_difference"
] = display_table[
    "paired_max_absolute_difference"
].map(
    lambda x: f"{x:.3e}"
)

display_table[
    "equivalence_rate_percent"
] = display_table[
    "equivalence_rate_percent"
].map(
    lambda x: f"{x:.1f}"
)

print(
    display_table.to_string(
        index=False
    )
)


# ---------------------------------------------------------------------
# Q007 HE-induced difference figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 4.8)
)

ax.plot(
    q007["epsilon"],
    q007[
        "paired_mean_absolute_difference"
    ],
    marker="o",
    linewidth=1.8,
    label=(
        "Mean absolute CDP–HE+DP difference"
    ),
)

ax.plot(
    q007["epsilon"],
    q007[
        "paired_max_absolute_difference"
    ],
    marker="s",
    linewidth=1.8,
    label=(
        "Maximum absolute CDP–HE+DP difference"
    ),
)

ax.axhline(
    HE_TOLERANCE,
    linestyle="--",
    linewidth=1.5,
    label=r"HE validation tolerance ($10^{-4}$ days)",
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"Privacy budget $\epsilon$"
)

ax.set_ylabel(
    "Paired absolute difference (days)"
)

ax.set_title(
    "Q007: HE-Induced Difference Relative to CDP"
)

ax.grid(
    True,
    which="both",
    linestyle=":",
    linewidth=0.7,
)

ax.legend(
    frameon=False
)

fig.tight_layout()

fig.savefig(
    FIGURE_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

print()
print("=" * 88)
print("HE EQUIVALENCE SUMMARY")
print("=" * 88)

print(
    "Q006 maximum paired difference:",
    q006[
        "paired_max_absolute_difference"
    ].max(),
)

print(
    "Q006 minimum exact-match rate:",
    q006[
        "exact_match_rate_percent"
    ].min(),
)

print(
    "Q007 maximum paired difference:",
    q007[
        "paired_max_absolute_difference"
    ].max(),
)

print(
    "Q007 minimum within-tolerance rate:",
    q007[
        "within_tolerance_rate_percent"
    ].min(),
)

print(
    "HE tolerance:",
    HE_TOLERANCE,
)

print()

print(
    "Saved table:",
    TABLE_FILE,
)

print(
    "Saved PNG:",
    FIGURE_PNG,
)

print(
    "Saved PDF:",
    FIGURE_PDF,
)

print()

print("Status: COMPLETE")