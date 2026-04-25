#!/usr/bin/env python3
"""Post-hoc augmentation script: add new E2 metrics to existing result files.

Adds the following metrics to each record's ``e2`` block:
    - all0_concept_count : int
        Number of top-N concepts whose pairs are all zero across every other
        selected concept and every window (uses existing per-concept
        ``all_pairs_zero`` flag).
    - nonzero_frac_window_ratio : float | None
        E2_nonzero_frac(w=1000) / E2_nonzero_frac(w=100). Returns None if the
        denominator is 0 (record with no nonzero pairs at w=100), so downstream
        analysis must handle None.

The script:
    - Reads each ``e2_cooccurrence_*.json`` file listed (or discovered via glob).
    - Computes the two new metrics in-memory.
    - Writes back the updated JSON to the same path (in-place) or to an explicit
      output directory, preserving all existing fields.
    - Does NOT recompute or re-query infini-gram. It only derives new numbers
      from existing ``pairwise_cooccurrence`` and ``metrics_by_window`` data.

Idempotence: running the script twice on the same file yields the same result.

Usage:
    # In-place update for a single file
    python e2_augment_metrics.py --files results/olmo2_7b_instruct/pretraining/gpt-5-mini/e2_cooccurrence_standard_top10.json

    # Batch update via glob (dry-run first to see what will change)
    python e2_augment_metrics.py --glob 'results/olmo2_7b_instrct/pretraining/gpt-5-mini/e2_cooccurrence_standard_top*.json' --dry-run
    python e2_augment_metrics.py --glob 'results/olmo2_*/pretraining/gpt-5-mini/e2_cooccurrence_standard_top*.json'

    # Write to a separate output directory (non-destructive)
    python e2_augment_metrics.py --glob 'results/**/e2_cooccurrence_*.json' --out-suffix .augmented
"""

import argparse
import glob
import json
import logging
import os
import sys
from typing import Optional


# Windows used for the ratio. If either window is missing from the record, the
# ratio is set to None.
RATIO_NUMERATOR_W = "1000"
RATIO_DENOMINATOR_W = "100"


# ============================================================
# Logger
# ============================================================

def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("e2_augment_metrics")


# ============================================================
# Metric computation
# ============================================================

def compute_all0_concept_count(record: dict) -> int:
    """Count top-N concepts with ``all_pairs_zero`` flag set to True.

    The flag is set by ``mark_all_pairs_zero`` in the original pipeline:
    a concept is ``all_pairs_zero`` iff every pair it participates in
    returned count=0 across every window.

    Returns 0 if the ``ranked_concepts`` field is missing or empty.
    """
    concepts = record.get("e2", {}).get("ranked_concepts", []) or []
    return sum(1 for c in concepts if c.get("all_pairs_zero") is True)


def compute_nonzero_frac_window_ratio(record: dict) -> Optional[float]:
    """Compute E2_nonzero_frac(w=1000) / E2_nonzero_frac(w=100).

    Returns None if:
      - Either window is absent from ``metrics_by_window``.
      - The denominator (w=100) is 0.

    The caller must handle None explicitly; we do not silently substitute
    a placeholder value because that would distort downstream aggregation.
    """
    mbw = record.get("e2", {}).get("metrics_by_window", {}) or {}
    num = mbw.get(RATIO_NUMERATOR_W, {}).get("E2_nonzero_frac")
    den = mbw.get(RATIO_DENOMINATOR_W, {}).get("E2_nonzero_frac")

    if num is None or den is None:
        return None
    if den == 0:
        return None
    return round(num / den, 6)


def augment_record(record: dict) -> dict:
    """Add the two new metrics to one record's ``e2`` block in place.

    Returns the updated record (same object; also modified in place).
    Preserves all existing fields.
    """
    if "e2" not in record or "error" in record.get("e2", {}):
        # Records with errors keep their error payload untouched.
        return record

    e2 = record["e2"]
    e2["all0_concept_count"] = compute_all0_concept_count(record)
    e2["nonzero_frac_window_ratio"] = compute_nonzero_frac_window_ratio(record)
    return record


# ============================================================
# File processing
# ============================================================

def augment_file(input_path: str, output_path: str, dry_run: bool,
                 logger: logging.Logger) -> dict:
    """Augment one JSON file and write the result.

    Returns a stats dict:
        {
          "records_total": int,
          "records_augmented": int,
          "records_with_error": int,
          "all0_distribution": dict[int, int],
          "ratio_none_count": int,
        }
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list of records in {input_path}, "
            f"got {type(data).__name__}"
        )

    stats = {
        "records_total": len(data),
        "records_augmented": 0,
        "records_with_error": 0,
        "all0_distribution": {},
        "ratio_none_count": 0,
    }

    for rec in data:
        if "error" in rec.get("e2", {}):
            stats["records_with_error"] += 1
            continue

        augment_record(rec)
        stats["records_augmented"] += 1

        a0 = rec["e2"]["all0_concept_count"]
        stats["all0_distribution"][a0] = stats["all0_distribution"].get(a0, 0) + 1
        if rec["e2"]["nonzero_frac_window_ratio"] is None:
            stats["ratio_none_count"] += 1

    if dry_run:
        logger.info("[dry-run] would write %d records to %s",
                    stats["records_total"], output_path)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Wrote %d records to %s", stats["records_total"], output_path)

    return stats


def discover_files(args) -> list[str]:
    """Resolve the list of input files from --files or --glob."""
    if args.files and args.glob_pattern:
        raise ValueError("Use either --files or --glob, not both")
    if args.files:
        return list(args.files)
    if args.glob_pattern:
        # recursive=True so '**' works.
        return sorted(glob.glob(args.glob_pattern, recursive=True))
    raise ValueError("Must specify --files or --glob")


def resolve_output_path(input_path: str, out_suffix: str) -> str:
    """Compute the output path.

    If ``out_suffix`` is empty, overwrite input (in-place). Otherwise append
    the suffix before the extension: ``x.json`` + ``.augmented`` -> ``x.augmented.json``.
    """
    if not out_suffix:
        return input_path
    root, ext = os.path.splitext(input_path)
    ext = ext or ".json"
    return f"{root}{out_suffix}{ext}"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-hoc augmentation: add all0_concept_count and "
            "nonzero_frac_window_ratio to existing E2 result files."
        )
    )
    p.add_argument("--files", nargs="+", default=None,
                   help="Explicit list of JSON files to process.")
    p.add_argument("--glob", dest="glob_pattern", type=str, default=None,
                   help="Glob pattern (supports ** with recursive=True).")
    p.add_argument("--out-suffix", dest="out_suffix", type=str, default="",
                   help=("Suffix inserted before '.json' to form the output "
                         "path. Empty string (default) means in-place update."))
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Compute metrics and print stats, but don't write files.")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    try:
        files = discover_files(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(2)

    if not files:
        logger.error("No files matched.")
        sys.exit(1)

    logger.info("Found %d file(s) to process", len(files))
    if args.dry_run:
        logger.info("*** DRY RUN — no files will be written ***")

    grand_total = {
        "files": 0,
        "records_total": 0,
        "records_augmented": 0,
        "records_with_error": 0,
        "ratio_none_count": 0,
    }
    all0_grand_dist: dict[int, int] = {}

    for path in files:
        output_path = resolve_output_path(path, args.out_suffix)
        logger.info("-" * 60)
        logger.info("Processing: %s", path)
        logger.info("Output:     %s", output_path)

        try:
            stats = augment_file(path, output_path, args.dry_run, logger)
        except Exception as e:
            logger.error("Failed on %s: %s", path, e)
            continue

        logger.info("  records total     : %d", stats["records_total"])
        logger.info("  records augmented : %d", stats["records_augmented"])
        logger.info("  records w/ error  : %d", stats["records_with_error"])
        logger.info("  all0_concept_count distribution:")
        for k in sorted(stats["all0_distribution"].keys()):
            logger.info("      count=%d : %d records",
                        k, stats["all0_distribution"][k])
        logger.info("  ratio=None count  : %d", stats["ratio_none_count"])

        grand_total["files"] += 1
        grand_total["records_total"] += stats["records_total"]
        grand_total["records_augmented"] += stats["records_augmented"]
        grand_total["records_with_error"] += stats["records_with_error"]
        grand_total["ratio_none_count"] += stats["ratio_none_count"]
        for k, v in stats["all0_distribution"].items():
            all0_grand_dist[k] = all0_grand_dist.get(k, 0) + v

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  files processed   : %d", grand_total["files"])
    logger.info("  records total     : %d", grand_total["records_total"])
    logger.info("  records augmented : %d", grand_total["records_augmented"])
    logger.info("  records w/ error  : %d", grand_total["records_with_error"])
    logger.info("  ratio=None total  : %d", grand_total["ratio_none_count"])
    logger.info("  all0_concept_count distribution across all files:")
    for k in sorted(all0_grand_dist.keys()):
        logger.info("      count=%d : %d records", k, all0_grand_dist[k])


if __name__ == "__main__":
    main()
