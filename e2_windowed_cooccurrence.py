#!/usr/bin/env python3
"""E2 Windowed Co-occurrence: Quantify co-occurrence of enabling concepts
from model responses within pretraining corpus using infini-gram CNF queries.

Intuition: Even if a response is not copied verbatim, the corpus may contain
co-occurring 'enabling concepts' (key phrases/entities/verbs) within a context
window, making recombination plausible.

Method:
  - Extract key phrases (n-grams) from the response as "enabling concepts".
  - For each pair of concepts, query infini-gram with a CNF (AND) query:
    count_cnf([[phrase_A], [phrase_B]], max_diff_tokens=w)
  - This counts how many times phrase_A AND phrase_B appear within w tokens
    of each other in the corpus.
  - Repeat for multiple window sizes (e.g., 100, 500, 1000 tokens).

Reference:
  - Project Overview: E2 definitions (Windowed Co-occurrence)
  - infini-gram CNF query API: count_cnf, find_cnf, search_docs_cnf

Usage:
    # Test run (first 5 records)
    python e2_windowed_cooccurrence.py \
        --input results/gpt_j_6b/e1_verbatim_standard.json \
        --output results/gpt_j_6b/e2_cooccurrence_standard.json \
        --limit 5

    # Full run (all records)
    python e2_windowed_cooccurrence.py \
        --input results/gpt_j_6b/e1_verbatim_standard.json \
        --output results/gpt_j_6b/e2_cooccurrence_standard.json

    # Custom windows
    python e2_windowed_cooccurrence.py \
        --input results/gpt_j_6b/e1_verbatim_standard.json \
        --output results/gpt_j_6b/e2_cooccurrence_standard.json \
        --windows 50 100 500 1000 5000

    # Compliant only
    python e2_windowed_cooccurrence.py \
        --input results/gpt_j_6b/e1_verbatim_standard.json \
        --output results/gpt_j_6b/e2_cooccurrence_standard.json \
        --compliant_only
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import combinations

from dotenv import load_dotenv

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = f"logs/e2_windowed_cooccurrence_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filepath, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to file: %s", log_filepath)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2 Windowed Co-occurrence: CNF-based corpus co-occurrence analysis"
    )
    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON (E1 results with e1 field, or raw labeled results)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON with E2 results appended")

    # Index config
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Path to local infini-gram index directory (default: ./index)")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")

    # E2 parameters
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Window sizes (max_diff_tokens) to test "
                             "(default: 100 500 1000)")
    parser.add_argument("--ngram_sizes", type=int, nargs="+",
                        default=[2, 3, 4],
                        help="N-gram sizes to extract as enabling concepts "
                             "(default: 2 3 4)")
    parser.add_argument("--max_concepts", type=int, default=20,
                        help="Maximum number of enabling concepts to extract per record "
                             "(default: 20)")
    parser.add_argument("--max_pairs", type=int, default=50,
                        help="Maximum number of concept pairs to query per record "
                             "(default: 50)")
    parser.add_argument("--max_clause_freq", type=int, default=None,
                        help="Max clause frequency for CNF queries (default: None = no limit)")

    # Processing
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to process (for testing)")
    parser.add_argument("--compliant_only", action="store_true",
                        help="Process only compliant (hb_label=1) records")
    parser.add_argument("--all_records", action="store_true",
                        help="Process all records regardless of hb_label (default)")

    return parser.parse_args()


# ===========================================================================
# Enabling concept extraction
# ===========================================================================
CORPUS_SIZE = 383_299_358_652  # Pile-train, Llama-2 tokenizer


def extract_enabling_concepts(engine, response_ids, tokenizer, ngram_sizes,
                              max_concepts, logger):
    """
    Extract 'enabling concepts' from the response as token n-grams.

    Strategy:
      1. Generate all n-grams of specified sizes from the response token IDs.
      2. Score each n-gram by rarity: score = sum(log(count(t)/N)) for each token.
         Lower score = rarer = more informative.
      3. Deduplicate (keep first occurrence of each unique n-gram).
      4. Return the top max_concepts rarest n-grams.

    Returns:
      List of dicts: [{"ngram_ids": [...], "text": "...", "score": float, "position": int}, ...]
    """
    L = len(response_ids)
    unigram_cache = {}

    # Step 1: Generate all n-grams with their positions
    candidates = []
    for n in ngram_sizes:
        if L < n:
            continue
        for i in range(L - n + 1):
            ngram_ids = tuple(response_ids[i:i + n])
            candidates.append((ngram_ids, i, n))

    logger.info("    Generated %d n-gram candidates (sizes=%s)", len(candidates), ngram_sizes)

    # Step 2: Score each unique n-gram by rarity
    seen_ngrams = set()
    scored = []
    for (ngram_ids, pos, n) in candidates:
        if ngram_ids in seen_ngrams:
            continue
        seen_ngrams.add(ngram_ids)

        # Compute rarity score: sum of log(count/N)
        score = 0.0
        for tid in ngram_ids:
            if tid not in unigram_cache:
                result = engine.count(input_ids=[tid])
                unigram_cache[tid] = result.get("count", 1)
            score += math.log(max(unigram_cache[tid], 1) / CORPUS_SIZE)
        scored.append((ngram_ids, pos, score))

    # Step 3: Sort by score ascending (most rare first)
    scored.sort(key=lambda x: x[2])

    # Step 4: Take top max_concepts
    top_concepts = scored[:max_concepts]
    logger.info("    Selected %d enabling concepts (from %d unique n-grams)",
                len(top_concepts), len(scored))

    # Build result
    concepts = []
    for (ngram_ids, pos, score) in top_concepts:
        text = ""
        try:
            text = tokenizer.decode(list(ngram_ids), skip_special_tokens=False)
        except Exception:
            pass
        concepts.append({
            "ngram_ids": list(ngram_ids),
            "text": text,
            "score": round(score, 4),
            "position": pos,
            "length": len(ngram_ids),
        })

    return concepts


# ===========================================================================
# CNF co-occurrence queries
# ===========================================================================
def compute_pairwise_cooccurrence(engine, concepts, windows, max_pairs,
                                  max_clause_freq, logger):
    """
    For each pair of enabling concepts, query co-occurrence count at each window size.

    CNF query format for infini-gram:
      cnf = [[phrase_A_ids], [phrase_B_ids]]
      count_cnf(cnf, max_diff_tokens=w)

    This counts documents where phrase_A AND phrase_B appear within w tokens.

    Returns:
      {
        "pairs": [
          {
            "concept_a": {"text": ..., "ngram_ids": ...},
            "concept_b": {"text": ..., "ngram_ids": ...},
            "counts_by_window": {100: count, 500: count, 1000: count}
          },
          ...
        ],
        "summary_by_window": {
          100: {"total_pairs": N, "nonzero_pairs": M, "max_count": K, "mean_count": ...},
          ...
        }
      }
    """
    # Generate all pairs, limit to max_pairs
    all_pairs = list(combinations(range(len(concepts)), 2))
    if len(all_pairs) > max_pairs:
        # Prioritize pairs of the rarest concepts (they're already sorted by rarity)
        all_pairs = all_pairs[:max_pairs]

    logger.info("    Querying %d concept pairs across %d windows...",
                len(all_pairs), len(windows))

    pair_results = []
    query_count = 0

    for pair_idx, (i, j) in enumerate(all_pairs):
        ca = concepts[i]
        cb = concepts[j]

        # Build CNF: [[concept_a_ids], [concept_b_ids]]
        cnf = [
            [ca["ngram_ids"]],  # clause 1: concept A (single disjunct)
            [cb["ngram_ids"]],  # clause 2: concept B (single disjunct)
        ]

        counts_by_window = {}
        for w in windows:
            try:
                result = engine.count_cnf(
                    cnf=cnf,
                    max_clause_freq=max_clause_freq,
                    max_diff_tokens=w,
                )
                count_val = result.get("count", 0)
                is_approx = result.get("approx", False)
                counts_by_window[w] = {
                    "count": count_val,
                    "approx": is_approx,
                }
                query_count += 1
            except Exception as exc:
                counts_by_window[w] = {
                    "count": -1,
                    "error": str(exc),
                }
                query_count += 1

        pair_results.append({
            "concept_a": {"text": ca["text"], "ngram_ids": ca["ngram_ids"],
                          "position": ca["position"]},
            "concept_b": {"text": cb["text"], "ngram_ids": cb["ngram_ids"],
                          "position": cb["position"]},
            "counts_by_window": counts_by_window,
        })

        if (pair_idx + 1) % 20 == 0:
            logger.info("      Pair %d / %d done (%d queries so far)",
                        pair_idx + 1, len(all_pairs), query_count)

    # Compute summary statistics per window
    summary_by_window = {}
    for w in windows:
        counts = [p["counts_by_window"][w]["count"]
                  for p in pair_results
                  if p["counts_by_window"][w].get("count", -1) >= 0]
        if counts:
            summary_by_window[w] = {
                "total_pairs": len(counts),
                "nonzero_pairs": sum(1 for c in counts if c > 0),
                "max_count": max(counts),
                "mean_count": round(sum(counts) / len(counts), 2),
                "median_count": sorted(counts)[len(counts) // 2],
            }
        else:
            summary_by_window[w] = {
                "total_pairs": 0,
                "nonzero_pairs": 0,
                "max_count": 0,
                "mean_count": 0,
                "median_count": 0,
            }

    logger.info("    Completed %d CNF queries", query_count)
    return {
        "pairs": pair_results,
        "summary_by_window": summary_by_window,
    }


# ===========================================================================
# E2 metrics computation
# ===========================================================================
def compute_e2_metrics(cooccurrence_result, windows):
    """
    Compute aggregate E2 metrics from pairwise co-occurrence results.

    Metrics per window:
      - E2_cooc(w): max co-occurrence count across all pairs at window w
      - E2_nonzero_frac(w): fraction of pairs with nonzero co-occurrence
      - E2_mean(w): mean co-occurrence count
      - E2_support_score: max_w log(1 + E2_cooc(w)) (from Project Overview §7.2)
    """
    summary = cooccurrence_result["summary_by_window"]

    metrics_by_window = {}
    for w in windows:
        s = summary.get(w, {})
        total = s.get("total_pairs", 0)
        metrics_by_window[w] = {
            "E2_cooc": s.get("max_count", 0),
            "E2_nonzero_frac": round(s.get("nonzero_pairs", 0) / total, 4) if total > 0 else 0.0,
            "E2_mean": s.get("mean_count", 0),
            "E2_median": s.get("median_count", 0),
            "total_pairs": total,
            "nonzero_pairs": s.get("nonzero_pairs", 0),
        }

    # Support score: max_w log(1 + E2_cooc(w))
    support_scores = []
    for w in windows:
        cooc = metrics_by_window[w]["E2_cooc"]
        support_scores.append(math.log(1 + cooc))
    support_score = max(support_scores) if support_scores else 0.0

    return {
        "metrics_by_window": metrics_by_window,
        "E2_support_score": round(support_score, 4),
        "windows_tested": windows,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    logger = setup_logger()
    args = parse_args()

    logger.info("=== E2 Windowed Co-occurrence started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Arguments: %s", vars(args))
    start_time = time.time()

    # Load input data
    logger.info("Loading input from %s ...", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records total", len(records))

    # Filter records
    if args.compliant_only:
        target_records = [r for r in records if r.get("hb_label") == 1]
        logger.info("Filtered to %d compliant (hb_label=1) records", len(target_records))
    elif args.all_records:
        target_records = records
        logger.info("Processing ALL records (--all_records)")
    else:
        target_records = records
        logger.info("Processing all %d records (default)", len(target_records))

    if args.limit is not None and args.limit > 0:
        target_records = target_records[:args.limit]
        logger.info("Limited to %d records (--limit)", len(target_records))

    if not target_records:
        logger.warning("No records to process. Exiting.")
        sys.exit(0)

    # Initialize tokenizer
    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    # Initialize infini-gram engine
    index_dir = args.index_dir or os.path.join(SCRIPT_DIR, "index")
    logger.info("Loading infini-gram engine from: %s", index_dir)

    if not os.path.isdir(index_dir):
        logger.error("Index directory not found: %s", index_dir)
        sys.exit(1)

    from infini_gram.engine import InfiniGramEngine

    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=index_dir,
        eos_token_id=tokenizer.eos_token_id,
        version=4,
        token_dtype="u16",
    )

    logger.info("Windows: %s", args.windows)
    logger.info("N-gram sizes: %s", args.ngram_sizes)
    logger.info("Max concepts per record: %d", args.max_concepts)
    logger.info("Max pairs per record: %d", args.max_pairs)

    # Process each record
    results = []

    for rec_idx, record in enumerate(target_records):
        rec_id = record.get("id", rec_idx)
        response = record.get("response", "")

        logger.info("=" * 70)
        logger.info("[%d/%d] Processing record id=%s",
                    rec_idx + 1, len(target_records), rec_id)
        logger.info("  Response length: %d chars", len(response))

        rec_start = time.time()

        try:
            # Tokenize response
            response_ids = tokenizer.encode(response)
            L = len(response_ids)
            logger.info("  Tokenized response: %d tokens", L)

            # Step 1: Extract enabling concepts
            logger.info("  Step 1: Extracting enabling concepts...")
            concepts = extract_enabling_concepts(
                engine, response_ids, tokenizer,
                ngram_sizes=args.ngram_sizes,
                max_concepts=args.max_concepts,
                logger=logger,
            )

            if len(concepts) < 2:
                logger.warning("  Only %d concepts extracted, skipping co-occurrence",
                               len(concepts))
                e2_metrics = {
                    "metrics_by_window": {w: {} for w in args.windows},
                    "E2_support_score": 0.0,
                    "windows_tested": args.windows,
                    "num_concepts": len(concepts),
                    "num_pairs_queried": 0,
                    "note": "Too few concepts for pairwise queries",
                }
                concepts_output = concepts
                cooc_result = {"pairs": [], "summary_by_window": {}}
            else:
                # Step 2: Pairwise co-occurrence queries
                logger.info("  Step 2: Computing pairwise co-occurrence...")
                cooc_result = compute_pairwise_cooccurrence(
                    engine, concepts, args.windows,
                    max_pairs=args.max_pairs,
                    max_clause_freq=args.max_clause_freq,
                    logger=logger,
                )

                # Step 3: Compute E2 metrics
                e2_metrics = compute_e2_metrics(cooc_result, args.windows)
                e2_metrics["num_concepts"] = len(concepts)
                e2_metrics["num_pairs_queried"] = len(cooc_result["pairs"])
                concepts_output = concepts

            # Build result record
            result = {
                "id": rec_id,
                "prompt": record.get("prompt", ""),
                "response": response,
                "model": record.get("model", ""),
                "metadata": record.get("metadata", {}),
                "hb_label": record.get("hb_label"),
                "e1": record.get("e1", {}),
                "e2": {
                    **e2_metrics,
                    "enabling_concepts": concepts_output,
                    "pairwise_cooccurrence": cooc_result,
                },
            }

            rec_elapsed = time.time() - rec_start

            # Log summary per window
            for w in args.windows:
                wm = e2_metrics.get("metrics_by_window", {}).get(w, {})
                logger.info("  Window w=%d: E2_cooc=%s, nonzero=%s/%s, mean=%s",
                            w,
                            wm.get("E2_cooc", "N/A"),
                            wm.get("nonzero_pairs", "N/A"),
                            wm.get("total_pairs", "N/A"),
                            wm.get("E2_mean", "N/A"))
            logger.info("  E2_support_score=%.4f, elapsed=%.1fs",
                        e2_metrics.get("E2_support_score", 0), rec_elapsed)

        except Exception as exc:
            logger.error("  Error processing record id=%s: %s", rec_id, exc,
                         exc_info=True)
            result = {
                "id": rec_id,
                "prompt": record.get("prompt", ""),
                "response": response,
                "model": record.get("model", ""),
                "metadata": record.get("metadata", {}),
                "hb_label": record.get("hb_label"),
                "e1": record.get("e1", {}),
                "e2": {"error": f"{type(exc).__name__}: {exc}"},
            }

        results.append(result)

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    logger.info("Saving %d results to %s ...", len(results), args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    successful = [r for r in results if "error" not in r.get("e2", {})]
    logger.info("  Total processed: %d", len(results))
    logger.info("  Successful: %d", len(successful))
    logger.info("  Errors: %d", len(results) - len(successful))

    if successful:
        for w in args.windows:
            scores = [r["e2"]["metrics_by_window"].get(w, {}).get("E2_cooc", 0)
                      for r in successful
                      if w in r["e2"].get("metrics_by_window", {})]
            if scores:
                logger.info("  Window w=%d: E2_cooc max=%d, mean=%.1f, "
                            "nonzero=%d/%d records",
                            w, max(scores), sum(scores) / len(scores),
                            sum(1 for s in scores if s > 0), len(scores))

        support_scores = [r["e2"].get("E2_support_score", 0) for r in successful]
        logger.info("  E2_support_score: min=%.4f, max=%.4f, mean=%.4f",
                    min(support_scores), max(support_scores),
                    sum(support_scores) / len(support_scores))

    end_time = time.time()
    elapsed_float = end_time - start_time
    elapsed = int(elapsed_float)
    days = elapsed // 86400
    hours = (elapsed % 86400) // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    logger.info("=== Job finished at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===",
                days, hours, minutes, seconds, elapsed_float)


if __name__ == "__main__":
    main()
