"""Benchmark the two governed HE+DP release architectures.

Architectures compared
----------------------
1. he_central_dp
   HE aggregation -> trusted exact decryption -> DP noise -> release

2. he_encrypted_noise_dp
   HE aggregation -> encrypted DP-noise addition -> decrypt already-noisy
   statistic -> DP-safe post-processing -> release

The runner deliberately uses only the public coordinator API. It does not
modify ledgers, reset accounting, or bypass governance. Before the first DP
request it computes the total epsilon required by the requested experiment
matrix and refuses to start when the coordinator's remaining benchmark budget
is insufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:9100"
DEFAULT_QUERIES = tuple(f"Q{i:03d}" for i in range(1, 8))
DEFAULT_EPSILONS = (0.1, 0.5, 1.0, 5.0, 10.0)
DEFAULT_REPETITIONS = 10
DEFAULT_ARCHITECTURES = (
    "he_central_dp",
    "he_encrypted_noise_dp",
)

ARCHITECTURE_PATHS = {
    "he_central_dp": "/query/{query}/dp",
    "he_encrypted_noise_dp": "/query/{query}/encrypted-noise/dp",
}


class BenchmarkError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def http_get_json(url: str, *, timeout: int) -> tuple[dict, float, int]:
    request = Request(url, method="GET")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raw = exc.read()
        elapsed = time.perf_counter() - start
        text = raw.decode("utf-8", errors="replace")
        raise BenchmarkError(
            f"HTTP {exc.code} for {url} after {elapsed:.6f}s: {text}"
        ) from exc
    except URLError as exc:
        elapsed = time.perf_counter() - start
        raise BenchmarkError(
            f"Connection failure for {url} after {elapsed:.6f}s: {exc}"
        ) from exc

    elapsed = time.perf_counter() - start
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Non-JSON response from {url}") from exc

    if not isinstance(payload, dict):
        raise BenchmarkError(f"Expected a JSON object from {url}")

    return payload, elapsed, len(raw)


def load_registry_policy() -> tuple[list[str], list[float]]:
    """Load enabled queries and the common allowed epsilon set.

    The benchmark should be run from the project root so the local coordinator
    package is importable. A safe fallback retains the currently approved
    experiment matrix if the import is unavailable.
    """
    try:
        from coordinator.registry import list_queries  # type: ignore

        definitions = list_queries()
        query_ids = [str(item["query_id"]).upper() for item in definitions]
        if not query_ids:
            raise ValueError("registry contains no enabled queries")

        epsilon_sets = {
            tuple(float(v) for v in item["allowed_epsilons"])
            for item in definitions
        }
        if len(epsilon_sets) != 1:
            raise ValueError(
                "enabled queries do not share one common epsilon allowlist"
            )
        epsilons = list(next(iter(epsilon_sets)))
        return query_ids, epsilons
    except Exception as exc:
        print(
            "WARNING: could not import coordinator.registry; using the "
            f"known current policy. Reason: {exc}",
            file=sys.stderr,
        )
        return list(DEFAULT_QUERIES), list(DEFAULT_EPSILONS)


def parse_csv_values(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def validate_requested_matrix(
    *,
    requested_queries: list[str],
    requested_epsilons: list[float],
    registry_queries: list[str],
    registry_epsilons: list[float],
) -> None:
    unknown_queries = sorted(set(requested_queries) - set(registry_queries))
    if unknown_queries:
        raise BenchmarkError(
            f"Queries not enabled in registry: {unknown_queries}"
        )

    allowed = set(registry_epsilons)
    invalid_epsilons = [eps for eps in requested_epsilons if eps not in allowed]
    if invalid_epsilons:
        raise BenchmarkError(
            "Requested epsilon values are not allowed by the registry: "
            f"{invalid_epsilons}; allowed={registry_epsilons}"
        )


def required_epsilon(
    *,
    queries: list[str],
    epsilons: list[float],
    architectures: list[str],
    repetitions: int,
) -> float:
    return (
        len(queries)
        * len(architectures)
        * repetitions
        * sum(float(eps) for eps in epsilons)
    )


def collect_ciphertext_metadata(
    *,
    base_url: str,
    queries: list[str],
    timeout: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for query_id in queries:
        query_lower = query_id.lower()
        url = f"{base_url}/query/{query_lower}"
        payload, elapsed, response_bytes = http_get_json(url, timeout=timeout)

        aggregate_ciphertext_bytes = 0
        component_sizes: dict[str, int] = {}
        for key, value in payload.items():
            if key == "ciphertext_bytes" and isinstance(value, int):
                component_sizes[key] = value
                aggregate_ciphertext_bytes += value
            elif (
                key.endswith("_ciphertext_bytes")
                and key != "hospital_ciphertext_bytes"
                and isinstance(value, int)
            ):
                component_sizes[key] = value
                aggregate_ciphertext_bytes += value

        hospital_sizes = payload.get("hospital_ciphertext_bytes")

        rows.append(
            {
                "query_id": query_id,
                "aggregation_time_seconds": elapsed,
                "coordinator_response_bytes": response_bytes,
                "aggregate_ciphertext_bytes": aggregate_ciphertext_bytes,
                "aggregate_ciphertext_components_json": canonical_json(
                    component_sizes
                ),
                "hospital_ciphertext_bytes_json": canonical_json(
                    hospital_sizes
                ),
                "participating_hospitals": len(
                    payload.get("participating_hospitals", [])
                ),
            }
        )
        print(
            f"[metadata] {query_id}: aggregate_ciphertext_bytes="
            f"{aggregate_ciphertext_bytes}, aggregation_time={elapsed:.6f}s"
        )

    return rows


def extract_scalar(query_id: str, dp_result: Any) -> float | None:
    if query_id in {"Q001", "Q006"}:
        if isinstance(dp_result, bool):
            return None
        if isinstance(dp_result, (int, float)):
            return float(dp_result)
        return None

    if query_id in {"Q003", "Q007"} and isinstance(dp_result, dict):
        value = dp_result.get("bounded_mean")
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)

    return None


def architecture_order(repetition: int, architectures: list[str]) -> list[str]:
    """Alternate order to reduce systematic first-call/cache bias."""
    if len(architectures) != 2 or repetition % 2 == 1:
        return list(architectures)
    return list(reversed(architectures))


def run_matrix(
    *,
    base_url: str,
    queries: list[str],
    epsilons: list[float],
    architectures: list[str],
    repetitions: int,
    timeout: int,
    continue_on_error: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sequence = 0

    for epsilon in epsilons:
        for query_id in queries:
            for repetition in range(1, repetitions + 1):
                for architecture in architecture_order(
                    repetition, architectures
                ):
                    sequence += 1
                    query_lower = query_id.lower()
                    path = ARCHITECTURE_PATHS[architecture].format(
                        query=query_lower
                    )
                    params = urlencode(
                        {
                            "epsilon": epsilon,
                            "confirm_high_spend": "true",
                        }
                    )
                    url = f"{base_url}{path}?{params}"

                    row: dict[str, Any] = {
                        "timestamp_utc": utc_now(),
                        "sequence": sequence,
                        "query_id": query_id,
                        "epsilon": epsilon,
                        "repetition": repetition,
                        "architecture": architecture,
                        "status": "success",
                        "request_id": "",
                        "mechanism": "",
                        "participating_hospitals": "",
                        "exact_aggregate_exposed": "",
                        "dp_noise_applied_under_he": "",
                        "noise_service_private_key_available": "",
                        "release_stage": "",
                        "dp_scalar": "",
                        "dp_result_json": "",
                        "total_request_time_seconds": "",
                        "client_response_bytes": "",
                        "budget_spent_epsilon": "",
                        "budget_remaining_epsilon": "",
                        "error": "",
                    }

                    try:
                        payload, elapsed, response_bytes = http_get_json(
                            url, timeout=timeout
                        )
                        dp_result = payload.get("dp_result")
                        scalar = extract_scalar(query_id, dp_result)

                        row.update(
                            {
                                "request_id": payload.get("request_id", ""),
                                "mechanism": payload.get("mechanism", ""),
                                "participating_hospitals": payload.get(
                                    "participating_hospitals", ""
                                ),
                                "exact_aggregate_exposed": payload.get(
                                    "exact_aggregate_exposed", ""
                                ),
                                "dp_noise_applied_under_he": payload.get(
                                    "dp_noise_applied_under_he", False
                                    if architecture == "he_central_dp"
                                    else "",
                                ),
                                "noise_service_private_key_available": payload.get(
                                    "noise_service_private_key_available", ""
                                ),
                                "release_stage": payload.get(
                                    "release_stage", ""
                                ),
                                "dp_scalar": "" if scalar is None else scalar,
                                "dp_result_json": canonical_json(dp_result),
                                "total_request_time_seconds": elapsed,
                                "client_response_bytes": response_bytes,
                                "budget_spent_epsilon": payload.get(
                                    "budget", {}
                                )
                                .get("spent", {})
                                .get("epsilon", ""),
                                "budget_remaining_epsilon": payload.get(
                                    "budget", {}
                                )
                                .get("remaining", {})
                                .get("epsilon", ""),
                            }
                        )
                        print(
                            f"[{sequence}] {query_id} eps={epsilon:g} "
                            f"rep={repetition} {architecture}: "
                            f"{elapsed:.6f}s"
                        )
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = str(exc)
                        print(
                            f"[{sequence}] FAILED {query_id} eps={epsilon:g} "
                            f"rep={repetition} {architecture}: {exc}",
                            file=sys.stderr,
                        )
                        rows.append(row)
                        if not continue_on_error:
                            raise
                        continue

                    rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise BenchmarkError(f"No rows available for {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["query_id"]),
            float(row["epsilon"]),
            str(row["architecture"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (query_id, epsilon, architecture), group in sorted(groups.items()):
        successful = [row for row in group if row["status"] == "success"]
        failed = len(group) - len(successful)
        latencies = [
            float(row["total_request_time_seconds"])
            for row in successful
            if row["total_request_time_seconds"] != ""
        ]
        response_sizes = [
            int(row["client_response_bytes"])
            for row in successful
            if row["client_response_bytes"] != ""
        ]
        scalar_values = [
            float(row["dp_scalar"])
            for row in successful
            if row["dp_scalar"] != ""
        ]

        def mean_or_blank(values: list[float | int]) -> float | str:
            return statistics.mean(values) if values else ""

        summary.append(
            {
                "query_id": query_id,
                "epsilon": epsilon,
                "architecture": architecture,
                "runs": len(group),
                "successes": len(successful),
                "failures": failed,
                "mean_latency_seconds": mean_or_blank(latencies),
                "median_latency_seconds": (
                    statistics.median(latencies) if latencies else ""
                ),
                "stdev_latency_seconds": (
                    statistics.stdev(latencies) if len(latencies) >= 2 else ""
                ),
                "min_latency_seconds": min(latencies) if latencies else "",
                "max_latency_seconds": max(latencies) if latencies else "",
                "mean_client_response_bytes": mean_or_blank(response_sizes),
                "mean_dp_scalar": mean_or_blank(scalar_values),
                "stdev_dp_scalar": (
                    statistics.stdev(scalar_values)
                    if len(scalar_values) >= 2
                    else ""
                ),
            }
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HE+central-DP with encrypted-noise-before-decryption."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--queries",
        default=",".join(DEFAULT_QUERIES),
        help="Comma-separated query IDs (default Q001-Q007).",
    )
    parser.add_argument(
        "--epsilons",
        default=",".join(str(v) for v in DEFAULT_EPSILONS),
        help="Comma-separated epsilon values.",
    )
    parser.add_argument(
        "--architectures",
        default=",".join(DEFAULT_ARCHITECTURES),
        help="Comma-separated architectures.",
    )
    parser.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--output",
        default="results/he/comparison/he_dp_comparison_raw.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/he/comparison/he_dp_comparison_summary.csv",
    )
    parser.add_argument(
        "--ciphertext-output",
        default="results/he/comparison/he_ciphertext_metadata.csv",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed requests and continue instead of stopping.",
    )
    parser.add_argument(
        "--skip-ciphertext-metadata",
        action="store_true",
        help="Do not call the non-DP HE-only routes before the benchmark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate policy/budget and print the experiment size only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.repetitions <= 0:
        raise BenchmarkError("--repetitions must be greater than zero")

    base_url = args.base_url.rstrip("/")
    queries = [item.upper() for item in parse_csv_values(args.queries)]
    epsilons = [float(item) for item in parse_csv_values(args.epsilons)]
    architectures = parse_csv_values(args.architectures)

    invalid_architectures = [
        item for item in architectures if item not in ARCHITECTURE_PATHS
    ]
    if invalid_architectures:
        raise BenchmarkError(
            f"Unsupported architectures: {invalid_architectures}"
        )

    registry_queries, registry_epsilons = load_registry_policy()
    validate_requested_matrix(
        requested_queries=queries,
        requested_epsilons=epsilons,
        registry_queries=registry_queries,
        registry_epsilons=registry_epsilons,
    )

    health, _, _ = http_get_json(f"{base_url}/health", timeout=args.timeout)
    if health.get("status") != "ok":
        raise BenchmarkError("Coordinator health check did not return status=ok")

    budget, _, _ = http_get_json(
        f"{base_url}/privacy/budget", timeout=args.timeout
    )
    remaining = float(budget["remaining"]["epsilon"])
    ceiling = float(budget["ceiling"]["epsilon"])
    spent = float(budget["spent"]["epsilon"])

    required = required_epsilon(
        queries=queries,
        epsilons=epsilons,
        architectures=architectures,
        repetitions=args.repetitions,
    )
    request_count = (
        len(queries)
        * len(epsilons)
        * len(architectures)
        * args.repetitions
    )

    print("Experiment preflight")
    print("--------------------")
    print(f"Queries:        {queries}")
    print(f"Epsilons:       {epsilons}")
    print(f"Architectures:  {architectures}")
    print(f"Repetitions:    {args.repetitions}")
    print(f"DP requests:    {request_count}")
    print(f"Required eps:   {required:.12g}")
    print(f"Budget ceiling: {ceiling:.12g}")
    print(f"Already spent:  {spent:.12g}")
    print(f"Remaining eps:  {remaining:.12g}")

    # Tiny tolerance only for normal binary floating-point residue.
    if remaining + 1e-9 < required:
        raise BenchmarkError(
            "Insufficient benchmark privacy budget. This runner will not "
            "reset or bypass accounting. Restart the coordinator with a "
            "separate benchmark ledger and a benchmark-only ceiling at least "
            f"{required:.12g} (prefer some headroom)."
        )

    if args.dry_run:
        print("Dry run complete; no DP requests sent.")
        return 0

    if not args.skip_ciphertext_metadata:
        metadata_rows = collect_ciphertext_metadata(
            base_url=base_url,
            queries=queries,
            timeout=args.timeout,
        )
        write_csv(Path(args.ciphertext_output), metadata_rows)
        print(f"Wrote ciphertext metadata: {args.ciphertext_output}")

    rows = run_matrix(
        base_url=base_url,
        queries=queries,
        epsilons=epsilons,
        architectures=architectures,
        repetitions=args.repetitions,
        timeout=args.timeout,
        continue_on_error=args.continue_on_error,
    )
    write_csv(Path(args.output), rows)
    print(f"Wrote raw results: {args.output}")

    summary_rows = summarize(rows)
    write_csv(Path(args.summary_output), summary_rows)
    print(f"Wrote summary: {args.summary_output}")

    final_budget, _, _ = http_get_json(
        f"{base_url}/privacy/budget", timeout=args.timeout
    )
    print("Final benchmark budget:")
    print(json.dumps(final_budget, indent=2))

    failures = sum(1 for row in rows if row["status"] != "success")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
