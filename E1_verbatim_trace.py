#!/usr/bin/env python3
"""E1 Verbatim Trace: Compute maximal matching spans between model responses
and pretraining corpus using infini-gram, following OLMoTrace Algorithm 1.

Phase 1: Maximal matching span computation + E1 metrics
Phase 2: Top-K span snippet retrieval (--retrieve_snippets)

Reference:
  - OLMoTrace (Liu et al., 2025): Algorithm 1, Steps 1-3
  - Project Overview: E1 definitions (LongestMatchLen, VerbatimCoverage, ExampleSnippets)

Usage:
    # Phase 1 only (metrics only, no snippet retrieval)
    python e1_verbatim_trace.py \
        --input results/gpt_j_6b/harmbench_standard_labeled.json \
        --output results/gpt_j_6b/e1_verbatim_standard.json

    # Phase 1 + Phase 2 (metrics + snippet retrieval)
    python e1_verbatim_trace.py \
        --input results/gpt_j_6b/harmbench_standard_labeled.json \
        --output results/gpt_j_6b/e1_verbatim_standard.json \
        --retrieve_snippets

    # With custom parameters
    python e1_verbatim_trace.py \
        --input results/gpt_j_6b/harmbench_standard_labeled.json \
        --output results/gpt_j_6b/e1_verbatim_standard.json \
        --retrieve_snippets \
        --l_min 20 \
        --top_k_ratio 0.05 \
        --max_docs_per_span 10 \
        --max_disp_len 80 \
        --limit 5
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# Path setup: load .env and add infini-gram pkg to path BEFORE importing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer


# Logger setup
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = f"logs/E1_verbatim_trace_{timestamp}.log"

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


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 Verbatim Trace: maximal matching spans + metrics"
    )
    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSON (labeled HarmBench results with hb_label field)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON with E1 results appended")

    # Index config
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Path to local infini-gram index directory "
                             "(default: ./index)")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")

    # E1 parameters
    parser.add_argument("--l_min", type=int, default=20,
                        help="Minimum span length (tokens) for VerbatimCoverage "
                             "(Project Overview default: 20)")
    parser.add_argument("--top_k_ratio", type=float, default=0.05,
                        help="Fraction of response tokens to keep as top-K spans "
                             "(OLMoTrace default: 0.05)")

    # Phase 2: snippet retrieval
    parser.add_argument("--retrieve_snippets", action="store_true",
                        help="Enable Phase 2: retrieve corpus snippets for top-K spans")
    parser.add_argument("--max_docs_per_span", type=int, default=10,
                        help="Max documents to retrieve per span (OLMoTrace default: 10)")
    parser.add_argument("--max_disp_len", type=int, default=80,
                        help="Max tokens to display per document snippet "
                             "(OLMoTrace default: 80)")

    # Processing
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of compliant records to process (for testing)")
    parser.add_argument("--compliant_only", action="store_true", default=True,
                        help="Process only compliant (hb_label=1) records (default: True)")
    parser.add_argument("--all_records", action="store_true",
                        help="Process all records regardless of hb_label")

    return parser.parse_args()


# ===========================================================================
# Compute maximal matching spans (OLMoTrace Algorithm 1)
# ===========================================================================
def get_longest_prefix_len(engine, suffix_ids, num_shards):
    """Find the longest prefix of suffix_ids that appears in the corpus.
    This implements GETLONGESTPREFIXLEN from OLMoTrace Algorithm 1."""
    if not suffix_ids:
        return 0

    # First, try the full suffix
    find_result = engine.find(input_ids=suffix_ids)
    total_count = find_result.get("cnt", 0)
    if total_count is None or total_count == 0:
        # Check via segment_by_shard
        total_count = 0
        for seg in find_result.get("segment_by_shard", []):
            total_count += seg[1] - seg[0]

    if total_count > 0:
        # The entire suffix is found verbatim
        return len(suffix_ids)

    # Binary search for the longest prefix that exists
    lo, hi = 0, len(suffix_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        prefix = suffix_ids[:mid]
        count_result = engine.count(input_ids=prefix)
        cnt = count_result.get("count", 0)
        if cnt > 0:
            lo = mid
        else:
            hi = mid - 1

    return lo


def compute_maximal_matching_spans(engine, response_ids, num_shards, logger):
    """Compute all maximal matching spans in the response.
    Implements OLMoTrace Algorithm 1: GETMAXIMALMATCHINGSPANS."""
    L = len(response_ids)
    logger.info("  Computing longest matching prefix for %d positions...", L)

    # Step 1: For each position, get longest matching prefix length
    # (Sequential; OLMoTrace parallelizes this, but 24 records × ~1000 tokens is manageable)
    prefix_lens = []
    for b in range(L):
        suffix = response_ids[b:]
        plen = get_longest_prefix_len(engine, suffix, num_shards)
        prefix_lens.append(plen)

        # Progress logging every 200 positions
        if (b + 1) % 200 == 0:
            logger.info("    Position %d / %d done (current prefix_len=%d)", b + 1, L, plen)

    # Collect raw spans (b, b + len)
    raw_spans = []
    for b, plen in enumerate(prefix_lens):
        if plen > 0:
            raw_spans.append((b, b + plen))

    logger.info("  Raw spans before suppression: %d", len(raw_spans))

    # Step 2: Suppress non-maximal spans (OLMoTrace Algorithm 1: SUPPRESSNONMAXIMALSPANS)
    # Spans are already sorted by beginning position (ascending) since we iterate b=0..L-1.
    # Keep a span only if its end position exceeds the max end seen so far.
    maximal_spans = []
    max_end = 0
    for (b, e) in raw_spans:
        if e > max_end:
            max_end = e
            maximal_spans.append((b, e))

    logger.info("  Maximal spans after suppression: %d", len(maximal_spans))
    return maximal_spans


# ===========================================================================
# Span filtering: keep top-K by span unigram probability (OLMoTrace Step 2)
# ===========================================================================
def compute_span_unigram_prob(engine, span_ids, unigram_cache):
    """Compute span unigram probability = product of token unigram probs.

    For numerical stability, we compute in log space:
      log(span_unigram_prob) = sum of log(unigram_prob(token)) for each token in span """

    # Args:
    #     engine: InfiniGramEngine instance
    #     span_ids: list of token IDs in the span
    #     unigram_cache: dict to cache {token_id: count} to avoid redundant queries

    # Returns:
    #     float: log(span_unigram_prob)  (more negative = more unique/long)
    
    log_prob = 0.0
    for tid in span_ids:
        if tid not in unigram_cache:
            result = engine.count(input_ids=[tid])
            unigram_cache[tid] = result.get("count", 1)
        log_prob += math.log(max(unigram_cache[tid], 1))

    # We don't normalize by total tokens because we only need relative ranking.
    # Lower log_prob (before normalization) actually means less common tokens,
    # but since count is raw count, lower total count = lower probability.
    # For ranking: span_unigram_prob = product(count(t_i)).
    # Smaller product → more unique. We want SMALLEST, so sort ascending.
    return log_prob


def filter_top_k_spans(engine, response_ids, maximal_spans, top_k_ratio, logger):
    """Filter to keep top-K spans with smallest span unigram probability.

    Implements OLMoTrace Step 2.
    K = ceil(top_k_ratio * L) where L = len(response_ids)."""
    L = len(response_ids)
    K = math.ceil(top_k_ratio * L)
    logger.info("  Filtering to top-K=%d spans (from %d maximal spans, L=%d)",
                K, len(maximal_spans), L)

    if len(maximal_spans) <= K:
        logger.info("  All %d spans kept (fewer than K=%d)", len(maximal_spans), K)
        return maximal_spans

    # Compute span unigram probability for each span
    unigram_cache = {}
    scored_spans = []
    for (b, e) in maximal_spans:
        span_ids = response_ids[b:e]
        log_prob = compute_span_unigram_prob(engine, span_ids, unigram_cache)
        scored_spans.append((b, e, log_prob))

    # Sort by log_prob ascending (smallest = most unique/interesting)
    scored_spans.sort(key=lambda x: x[2])

    # Keep top-K
    top_k = [(b, e) for (b, e, _) in scored_spans[:K]]

    # Re-sort by position for readability
    top_k.sort(key=lambda x: x[0])

    logger.info("  Top-K spans selected: %d", len(top_k))
    return top_k


# ===========================================================================
# Phase 2: Retrieve corpus snippets for top-K spans (OLMoTrace Step 3)
# ===========================================================================
def retrieve_snippets_for_span(engine, span_ids, max_docs, max_disp_len, tokenizer):
    """Retrieve up to max_docs document snippets containing the given span.

    Uses engine.find() to locate the span in the suffix array,
    then engine.get_doc_by_rank() to retrieve each document snippet.

    Args:
        engine: InfiniGramEngine instance
        span_ids: list of token IDs for the span
        max_docs: maximum number of documents to retrieve
        max_disp_len: maximum tokens to display per snippet
        tokenizer: tokenizer for decoding token IDs to text

    Returns:
        list of dicts: each dict has keys:
            - doc_ix: document index in the corpus
            - doc_len: total document length in tokens
            - snippet_token_ids: list of token IDs around the match
            - snippet_text: decoded text of the snippet
            - metadata: document metadata (if available)
    """
    find_result = engine.find(input_ids=span_ids)
    segments = find_result.get("segment_by_shard", [])

    snippets = []
    docs_retrieved = 0

    for s, (start_rank, end_rank) in enumerate(segments):
        if docs_retrieved >= max_docs:
            break

        count_in_shard = end_rank - start_rank
        if count_in_shard <= 0:
            continue

        # If more than max_docs occurrences, sample (OLMoTrace Step 3)
        if count_in_shard <= (max_docs - docs_retrieved):
            ranks = range(start_rank, end_rank)
        else:
            import random
            remaining = max_docs - docs_retrieved
            ranks = sorted(random.sample(range(start_rank, end_rank), remaining))

        for rank in ranks:
            if docs_retrieved >= max_docs:
                break
            try:
                doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=max_disp_len)
                if "error" in doc:
                    continue

                snippet_info = {
                    "doc_ix": doc.get("doc_ix"),
                    "doc_len": doc.get("doc_len"),
                    "metadata": doc.get("metadata", ""),
                }

                # Decode snippet text
                if "token_ids" in doc and tokenizer is not None:
                    try:
                        snippet_info["snippet_text"] = tokenizer.decode(
                            doc["token_ids"], skip_special_tokens=False
                        )
                        snippet_info["snippet_token_ids"] = doc["token_ids"]
                    except Exception:
                        snippet_info["snippet_text"] = ""
                        snippet_info["snippet_token_ids"] = doc.get("token_ids", [])
                else:
                    snippet_info["snippet_text"] = ""
                    snippet_info["snippet_token_ids"] = doc.get("token_ids", [])

                snippets.append(snippet_info)
                docs_retrieved += 1

            except Exception:
                continue

    return snippets


# ===========================================================================
# E1 metrics computation
# ===========================================================================
def compute_e1_metrics(response_ids, maximal_spans, top_k_spans, l_min, tokenizer):
    """Compute E1 metrics from spans.

    Metrics (from Project Overview):
      - LongestMatchLen: max span length (tokens) among all maximal spans
      - VerbatimCoverage: fraction of response tokens covered by spans >= l_min tokens
      - num_maximal_spans: total count of maximal matching spans
      - num_top_k_spans: count of top-K filtered spans
      - span_lengths: list of all maximal span lengths
      - top_k_span_details: list of dicts with begin, end, length, text for top-K spans

    Args:
        response_ids: list of token IDs
        maximal_spans: list of (begin, end) from Algorithm 1
        top_k_spans: list of (begin, end) after filtering
        l_min: minimum span length for coverage calculation
        tokenizer: tokenizer for decoding

    Returns:
        dict: E1 metrics
    """
    L = len(response_ids)

    # Span lengths
    all_lengths = [e - b for (b, e) in maximal_spans]

    # LongestMatchLen
    longest = max(all_lengths) if all_lengths else 0

    # VerbatimCoverage: fraction of tokens covered by spans >= l_min
    covered = set()
    for (b, e) in maximal_spans:
        if (e - b) >= l_min:
            for t in range(b, e):
                covered.add(t)
    coverage = len(covered) / L if L > 0 else 0.0

    # VerbatimCoverage with all spans (no l_min filter) for reference
    covered_all = set()
    for (b, e) in maximal_spans:
        for t in range(b, e):
            covered_all.add(t)
    coverage_all = len(covered_all) / L if L > 0 else 0.0

    # Top-K span details
    top_k_details = []
    for (b, e) in top_k_spans:
        span_ids = response_ids[b:e]
        span_text = ""
        if tokenizer is not None:
            try:
                span_text = tokenizer.decode(span_ids, skip_special_tokens=False)
            except Exception:
                pass
        top_k_details.append({
            "begin": b,
            "end": e,
            "length": e - b,
            "text": span_text,
        })

    metrics = {
        "response_token_len": L,
        "LongestMatchLen": longest,
        "VerbatimCoverage": round(coverage, 4),
        "VerbatimCoverage_nomin": round(coverage_all, 4),
        "l_min": l_min,
        "num_maximal_spans": len(maximal_spans),
        "num_top_k_spans": len(top_k_spans),
        "span_length_distribution": {
            "min": min(all_lengths) if all_lengths else 0,
            "max": longest,
            "mean": round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else 0,
            "median": sorted(all_lengths)[len(all_lengths) // 2] if all_lengths else 0,
        },
        "all_maximal_spans": [{"begin": b, "end": e, "length": e - b}
                               for (b, e) in maximal_spans],
        "top_k_spans": top_k_details,
    }

    return metrics


# ===========================================================================
# Main
# ===========================================================================
def main():
    logger = setup_logger()
    args = parse_args()

    logger.info("=== E1 Verbatim Trace started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Arguments: %s", vars(args))
    start_time = time.time()

    # -----------------------------------------------------------------------
    # Load input data
    # -----------------------------------------------------------------------
    logger.info("Loading input from %s ...", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records total", len(records))

    # Filter to compliant only (unless --all_records)
    if args.all_records:
        target_records = records
        logger.info("Processing ALL records (--all_records)")
    else:
        target_records = [r for r in records if r.get("hb_label") == 1]
        logger.info("Filtered to %d compliant (hb_label=1) records", len(target_records))

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
        logger.error("Point --index_dir to the directory containing "
                      "tokenized.*, offset.*, table.* files")
        sys.exit(1)

    from infini_gram.engine import InfiniGramEngine

    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=index_dir,
        eos_token_id=tokenizer.eos_token_id,
        version=4,
        token_dtype="u16",
    )

    # Detect number of shards
    num_shards = len([f for f in os.listdir(index_dir)
                      if f.startswith("tokenized.")])
    logger.info("Detected %d shard(s) in index", num_shards)

    # Process each record
    results = []

    for rec_idx, record in enumerate(target_records):
        rec_id = record.get("id", rec_idx)
        prompt = record.get("prompt", "")
        response = record.get("response", "")

        logger.info("=" * 70)
        logger.info("[%d/%d] Processing record id=%s", rec_idx + 1,
                    len(target_records), rec_id)
        logger.info("  Prompt: %s", prompt[:100])
        logger.info("  Response length: %d chars", len(response))

        rec_start = time.time()

        try:
            # Tokenize response with Llama-2 tokenizer (matching the index)
            response_ids = tokenizer.encode(response)
            L = len(response_ids)
            logger.info("  Tokenized response: %d tokens", L)

            # ---------------------------------------------------------------
            # Phase 1: Compute maximal matching spans (Algorithm 1)
            # ---------------------------------------------------------------
            maximal_spans = compute_maximal_matching_spans(
                engine, response_ids, num_shards, logger
            )

            # Filter to top-K spans
            top_k_spans = filter_top_k_spans(
                engine, response_ids, maximal_spans, args.top_k_ratio, logger
            )

            # Compute E1 metrics
            e1_metrics = compute_e1_metrics(
                response_ids, maximal_spans, top_k_spans, args.l_min, tokenizer
            )

            # ---------------------------------------------------------------
            # Phase 2 (optional): Retrieve snippets for top-K spans
            # ---------------------------------------------------------------
            if args.retrieve_snippets:
                logger.info("  Phase 2: Retrieving snippets for %d top-K spans...",
                            len(top_k_spans))
                snippets_by_span = []
                for span_idx, (b, e) in enumerate(top_k_spans):
                    span_ids = response_ids[b:e]
                    snippets = retrieve_snippets_for_span(
                        engine, span_ids,
                        max_docs=args.max_docs_per_span,
                        max_disp_len=args.max_disp_len,
                        tokenizer=tokenizer,
                    )
                    snippets_by_span.append({
                        "span_begin": b,
                        "span_end": e,
                        "span_length": e - b,
                        "span_text": e1_metrics["top_k_spans"][span_idx]["text"],
                        "num_snippets": len(snippets),
                        "snippets": snippets,
                    })

                    if (span_idx + 1) % 10 == 0:
                        logger.info("    Snippet retrieval: %d / %d spans done",
                                    span_idx + 1, len(top_k_spans))

                e1_metrics["ExampleSnippets"] = snippets_by_span
                logger.info("  Snippets retrieved for %d spans (total %d snippets)",
                            len(snippets_by_span),
                            sum(s["num_snippets"] for s in snippets_by_span))

            # Build result record
            result = {
                "id": rec_id,
                "prompt": prompt,
                "response": response,
                "model": record.get("model", ""),
                "metadata": record.get("metadata", {}),
                "hb_label": record.get("hb_label"),
                "e1": e1_metrics,
            }

            rec_elapsed = time.time() - rec_start
            logger.info("  E1 results: LongestMatchLen=%d, VerbatimCoverage=%.4f "
                        "(l_min=%d), spans=%d/%d, elapsed=%.1fs",
                        e1_metrics["LongestMatchLen"],
                        e1_metrics["VerbatimCoverage"],
                        args.l_min,
                        e1_metrics["num_top_k_spans"],
                        e1_metrics["num_maximal_spans"],
                        rec_elapsed)

        except Exception as exc:
            logger.error("  Error processing record id=%s: %s", rec_id, exc,
                         exc_info=True)
            result = {
                "id": rec_id,
                "prompt": prompt,
                "response": response,
                "model": record.get("model", ""),
                "metadata": record.get("metadata", {}),
                "hb_label": record.get("hb_label"),
                "e1": {"error": f"{type(exc).__name__}: {exc}"},
            }

        results.append(result)

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    logger.info("Saving %d results to %s ...", len(results), args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    successful = [r for r in results if "error" not in r.get("e1", {})]
    logger.info("  Total processed: %d", len(results))
    logger.info("  Successful: %d", len(successful))
    logger.info("  Errors: %d", len(results) - len(successful))

    if successful:
        longest_vals = [r["e1"]["LongestMatchLen"] for r in successful]
        coverage_vals = [r["e1"]["VerbatimCoverage"] for r in successful]
        logger.info("  LongestMatchLen:  min=%d, max=%d, mean=%.1f",
                    min(longest_vals), max(longest_vals),
                    sum(longest_vals) / len(longest_vals))
        logger.info("  VerbatimCoverage: min=%.4f, max=%.4f, mean=%.4f",
                    min(coverage_vals), max(coverage_vals),
                    sum(coverage_vals) / len(coverage_vals))

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