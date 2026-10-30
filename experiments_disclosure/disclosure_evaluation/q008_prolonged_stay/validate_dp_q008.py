"""
Basic validation of the Q008 DP mechanism.
"""

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.dp import (
    privatize_prolonged_histogram,
)


EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]


# Example controlled cohort:
#
# 20 patients
# 4 prolonged
# 16 not prolonged

EXACT_NOT_PROLONGED = 16
EXACT_PROLONGED = 4


print("=" * 72)
print("Q008 DP MECHANISM SANITY CHECK")
print("=" * 72)

print()
print(
    f"Exact histogram: "
    f"[not_prolonged={EXACT_NOT_PROLONGED}, "
    f"prolonged={EXACT_PROLONGED}]"
)

print()


for epsilon in EPSILONS:

    print(f"epsilon = {epsilon}")

    for repetition in range(5):

        released = (
            privatize_prolonged_histogram(
                not_prolonged=EXACT_NOT_PROLONGED,
                prolonged=EXACT_PROLONGED,
                epsilon=epsilon,
            )
        )

        print(
            f"  rep {repetition + 1}: "
            f"not={released['not_prolonged']:3d}, "
            f"prolonged={released['prolonged']:3d}, "
            f"total={released['total']:3d}, "
            f"rate={released['prolonged_rate']}"
        )

    print()


print("=" * 72)
print("Q008 DP SANITY CHECK COMPLETE")
print("=" * 72)