from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
    / "q008_exact_rates.csv"
)

RESULTS_DIR = INPUT_FILE.parent


df = pd.read_csv(INPUT_FILE)

# ---------------------------------------------------------
# Basic checks
# ---------------------------------------------------------

print("=" * 70)
print("Q008 COHORT-SIZE INSPECTION")
print("=" * 70)

print(f"Total race × admission groups: {len(df)}")
print(f"Non-empty groups:              {(df['total'] > 0).sum()}")
print(f"Empty groups:                  {(df['total'] == 0).sum()}")

print()


# ---------------------------------------------------------
# Cohort-size buckets
# ---------------------------------------------------------

def cohort_bucket(n):
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 5:
        return "2-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    if n <= 500:
        return "101-500"
    if n <= 1000:
        return "501-1000"
    return ">1000"


df["cohort_bucket"] = df["total"].apply(cohort_bucket)

bucket_order = [
    "0",
    "1",
    "2-5",
    "6-10",
    "11-20",
    "21-50",
    "51-100",
    "101-500",
    "501-1000",
    ">1000",
]

bucket_counts = (
    df["cohort_bucket"]
    .value_counts()
    .reindex(bucket_order, fill_value=0)
)

print("GROUPS BY COHORT SIZE")
print("---------------------")

for bucket, count in bucket_counts.items():
    print(f"{bucket:>10}: {count}")

print()


# ---------------------------------------------------------
# Smallest non-empty groups
# ---------------------------------------------------------

nonempty = (
    df[df["total"] > 0]
    .sort_values(
        by=[
            "total",
            "race",
            "admission_type",
        ]
    )
)

print("20 SMALLEST NON-EMPTY GROUPS")
print("----------------------------")

display_columns = [
    "race",
    "admission_type",
    "total",
    "prolonged",
    "not_prolonged",
    "prolonged_rate",
]

print(
    nonempty[display_columns]
    .head(20)
    .to_string(index=False)
)

print()


# ---------------------------------------------------------
# Exact 0% and 100% rates
# ---------------------------------------------------------

zero_rate = nonempty[
    nonempty["prolonged"] == 0
]

full_rate = nonempty[
    nonempty["not_prolonged"] == 0
]

print("EXACT EXTREME RATES")
print("-------------------")

print(
    f"Groups with exact 0% prolonged rate:   "
    f"{len(zero_rate)}"
)

print(
    f"Groups with exact 100% prolonged rate: "
    f"{len(full_rate)}"
)

print()


# ---------------------------------------------------------
# Small groups with potentially revealing exact rates
# ---------------------------------------------------------

small_groups = nonempty[
    nonempty["total"] <= 20
].copy()

small_groups["prolonged_rate_percent"] = (
    small_groups["prolonged_rate"] * 100
)

small_groups = small_groups.sort_values(
    by=[
        "total",
        "race",
        "admission_type",
    ]
)

output_path = (
    RESULTS_DIR
    / "q008_small_cohorts.csv"
)

small_groups.to_csv(
    output_path,
    index=False,
)

print(
    f"Groups with <=20 patients: "
    f"{len(small_groups)}"
)

print()
print(f"Saved: {output_path}")