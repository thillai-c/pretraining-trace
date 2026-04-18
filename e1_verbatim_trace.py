#!/usr/bin/env python3
"""E1 Verbatim Trace: Compute maximal matching spans between model responses
and pretraining corpus using infini-gram, following OLMoTrace Algorithm 1.

Phase 1: Maximal matching span computation + E1 metrics
Phase 2: Top-K span snippet retrieval (--retrieve_snippets)

Reference:
  - OLMoTrace (Liu et al., 2025): Algorithm 1, Steps 1-3
  - Project Overview: E1 definitions (LongestMatchLen, VerbatimCoverage, ExampleSnippets)

Usage:
    # GPT-J (local engine, Pile-train index)
    python e1_verbatim_trace.py \
        --model gpt-j \
        --index_dir ./index/the-pile \
        --config standard

    # OLMo 2 with custom API index (single config)
    python e1_verbatim_trace.py \
        --model olmo2-7b \
        --api_index v4_olmo-mix-1124_llama \
        --config contextual
    
    python e1_verbatim_trace.py --model olmo2-7b --api_index v4_olmo-mix-1124_llama --config contextual

    # OLMo 2 with local dolmino index (mid-training; default output: results/{model}/mid_training/...)
    python e1_verbatim_trace.py \
        --model olmo2-7b \
        --index_dir ./index/dolmino-mix-1124 \
        --retrieve_snippets

    # With custom parameters
    python e1_verbatim_trace.py \
        --model olmo2-7b \
        --retrieve_snippets \
        --top_k_ratio 0.05 \
        --max_docs_per_span 10 \
        --max_disp_len 80 \
        --limit 5
"""

import argparse
import json
import logging
from typing import Optional
import math
import os
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer


from utils import MODEL_CONFIGS


# ===========================================================================
# InfiniGramAPIEngine: HTTP API wrapper matching local engine interface
# ===========================================================================
class InfiniGramAPIEngine:
    """Wrapper around the infini-gram HTTP API that mimics the local InfiniGramEngine interface (count, find, get_doc_by_rank).

    The API endpoint accepts JSON POST requests.  We use ``query_ids`` (list of token IDs) rather than ``query`` (string) so that
    tokenization stays under our control — identical to the local engine workflow.
    """

    API_URL = "https://api.infini-gram.io/"

    def __init__(self, index: str, max_retries: int = 8,
                 retry_delay: float = 5.0):
        self.index = index
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    # Internal helper: POST with retry
    def _post(self, payload: dict) -> dict:
        """Send a POST request to the API with exponential-backoff retry."""
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.API_URL,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        raise RuntimeError(
                            f"API returned error: {data['error']}")
                    time.sleep(1)  # 500ms delay between API calls
                    return data

                # Retryable status codes (including 403 for rate limiting)
                if resp.status_code in (403, 429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    resp.raise_for_status()

                # Non-retryable error
                resp.raise_for_status()

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        # Should not reach here, but just in case
        raise RuntimeError("API request failed after all retries")

    def count(self, input_ids: list) -> dict:
        """Count occurrences of the token ID sequence in the corpus.
        If ``input_ids`` is empty, returns the total number of tokens in the corpus (used to obtain CORPUS_SIZE)."""
        if len(input_ids) == 0:
            # Empty string query → total token count
            payload = {
                "index": self.index,
                "query_type": "count",
                "query": "",
            }
        else:
            payload = {
                "index": self.index,
                "query_type": "count",
                "query_ids": input_ids,
            }
        return self._post(payload)

    def find(self, input_ids: list) -> dict:
        """Locate the token ID sequence in the corpus."""
        payload = {
            "index": self.index,
            "query_type": "find",
            "query_ids": input_ids,
        }
        return self._post(payload)

    def get_doc_by_rank(self, s: int, rank: int, max_disp_len: int = 80, query_ids: list = None) -> dict:
        payload = {
            "index": self.index,
            "query_type": "get_doc_by_rank",
            "s": s,
            "rank": rank,
            "max_disp_len": max_disp_len,
        }
        if query_ids is not None:
            payload["query_ids"] = query_ids
        return self._post(payload)


# ===========================================================================
# Logger setup
# ===========================================================================
def setup_logger(model_key: str, config: str = "standard"):
    """Create logger with model-based log directory.
    Log file: ``logs/{model_key}/e1_verbatim_trace.log``
    """
    out_dir = MODEL_CONFIGS[model_key]["out_dir_name"]
    log_dir = os.path.join("logs", out_dir)
    os.makedirs(log_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = os.path.join(
        log_dir, f"e1_verbatim_trace_{config}.log"
    )

    logger_name = f"e1_verbatim_trace.{model_key}.{config}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Logging to file: %s", log_filepath)
    return logger


def e1_results_subdir(args) -> Optional[str]:
    """Subdirectory under results/{model_dir}/ for default --output (from index choice).

    - API ``--api_index`` containing ``olmo-mix-1124_llama`` -> ``pretraining``
    - Local ``--index_dir`` for dolmino-mix-1124 -> ``mid_training``
    - Local path under ``post-training`` (or API name with post-training markers) -> ``post_training``
    - Otherwise -> ``None`` (flat ``results/{model_dir}/``, e.g. Pile local index).

    Explicit ``--output`` bypasses this (see parse_args).
    """
    index_dir = args.index_dir
    api_index = (args.api_index or "")

    if index_dir:
        norm = os.path.normpath(index_dir).replace("\\", "/").lower()
        base = os.path.basename(os.path.normpath(index_dir)).lower()
        if "dolmino-mix-1124" in norm or base == "dolmino-mix-1124":
            return "mid_training"
        if "post-training" in norm or "post_training" in norm:
            return "post_training"
        return None

    al = api_index.lower()
    if "olmo-mix-1124_llama" in al:
        return "pretraining"
    if any(
        marker in al
        for marker in ("post-training", "post_training", "posttrain")
    ):
        return "post_training"
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 Verbatim Trace: maximal matching spans + metrics"
    )

    # Model selection (required)
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (determines auto paths and log dir)")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench config name (default: standard). "
                             "If you want to run multiple configs in one invocation, "
                             "use --configs.")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Run multiple HarmBench configs in one invocation. "
             "Example: --configs standard contextual copyright",
    )

    # I/O (auto-generated if not specified)
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSON (labeled HarmBench results). "
                             "Default: data/{model_dir}/harmbench_{config}_labeled.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON with E1 results. If omitted, path is chosen from "
                             "the index: API olmo-mix-1124_llama -> results/{model_dir}/pretraining/; "
                             "local dolmino-mix-1124 -> .../mid_training/; "
                             "local post-training (or matching API name) -> .../post_training/; "
                             "else results/{model_dir}/e1_verbatim_{config}.json (flat).")

    # Index config — local vs API
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Path to local infini-gram index directory. "
                             "If provided, use local InfiniGramEngine. "
                             "If omitted, use HTTP API.")
    parser.add_argument("--api_index", type=str,
                        default="v4_olmo-mix-1124_llama",
                        help="API index name (only used when --index_dir is "
                             "not provided). Default: v4_olmo-mix-1124_llama")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")

    # E1 parameters
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

    args = parser.parse_args()

    return args


def _resolve_configs(args) -> list:
    if args.configs:
        return list(args.configs)
    return [args.config]


# ===========================================================================
# Compute maximal matching spans (OLMoTrace Algorithm 1)
# ===========================================================================
def get_longest_prefix_len(engine, suffix_ids, num_shards):
    """
    Find the longest prefix of suffix_ids that appears in the corpus. (OLMoTrace Algorithm 1GETLONGESTPREFIXLEN)
      - For API: probe from small lengths to find tight upper bound, then binary search.
      - For local engine: uses find() first, then binary search if needed.
    """
    if not suffix_ids:
        return 0

    if isinstance(engine, InfiniGramAPIEngine):
        # API mode: probe to find tight upper bound, then binary search.
        # Skip find() since it almost always fails on long suffixes.
        if len(suffix_ids) > 500:
            suffix_ids = suffix_ids[:500]

        # Probe progressively larger lengths to find where count drops to 0
        hi = len(suffix_ids)
        for probe in [8, 16, 32, 64, 128, 256]:
            if probe >= hi:
                break
            count_result = engine.count(input_ids=suffix_ids[:probe])
            if count_result.get("count", 0) == 0:
                hi = probe  # matching prefix is shorter than probe
                break

        # Binary search within tighter range [0, hi]
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            count_result = engine.count(input_ids=suffix_ids[:mid])
            if count_result.get("count", 0) > 0:
                lo = mid
            else:
                hi = mid - 1

        return lo

    else:
        # Local engine: original algorithm (find first, then binary search)
        find_result = engine.find(input_ids=suffix_ids)
        total_count = find_result.get("cnt", 0)
        if total_count is None or total_count == 0:
            total_count = 0
            for seg in find_result.get("segment_by_shard", []):
                total_count += seg[1] - seg[0]

        if total_count > 0:
            return len(suffix_ids)

        lo, hi = 0, len(suffix_ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            count_result = engine.count(input_ids=suffix_ids[:mid])
            if count_result.get("count", 0) > 0:
                lo = mid
            else:
                hi = mid - 1

        return lo

def compute_maximal_matching_spans(engine, response_ids, num_shards, logger):
    """
    Compute all maximal matching spans in the response. (OLMoTrace Algorithm 1: GETMAXIMALMATCHINGSPANS.)
      - Step 1: For each position, find longest matching prefix.
      - Step 2: Suppress non-maximal spans.
    """
    L = len(response_ids)
    logger.info("  Computing longest matching prefix for %d positions...", L)

    # Step 1: For each position, get longest matching prefix length
    prefix_lens = []
    for b in range(L):
        plen = get_longest_prefix_len(engine, response_ids[b:], num_shards)
        prefix_lens.append(plen)

        if (b + 1) % 200 == 0:
            logger.info("    Position %d / %d done (current prefix_len=%d)", b + 1, L, plen)

    # Collect raw spans
    raw_spans = [(b, b + plen) for b, plen in enumerate(prefix_lens) if plen > 0]
    logger.info("  Raw spans before suppression: %d", len(raw_spans))

    # Step 2: Suppress non-maximal spans
    maximal_spans = []
    max_end = 0
    for (b, e) in raw_spans:
        if e > max_end:
            max_end = e
            maximal_spans.append((b, e))

    logger.info("  Maximal spans after suppression: %d", len(maximal_spans))
    return maximal_spans


# ===========================================================================
# Span filtering: keep top-K by per-token unigram probability
# ===========================================================================
def compute_span_score(engine, span_ids, unigram_cache, corpus_size):
    """
    Compute span unigram probability (OLMoTrace Step 2).
      - score = sum(log(count(t_i) / N))
      - Lower score → span is longer and/or contains rarer tokens → more unique.
      - NO per-token normalization: longer spans naturally get lower scores
        because each additional token contributes a negative log-probability term.
    """
    log_prob = 0.0
    for tid in span_ids:
        if tid not in unigram_cache:
            result = engine.count(input_ids=[tid])
            unigram_cache[tid] = result.get("count", 1)
        log_prob += math.log(max(unigram_cache[tid], 1) / corpus_size)

    return log_prob if span_ids else 0.0


def filter_top_k_spans(engine, response_ids, maximal_spans, top_k_ratio,
                        corpus_size, logger):
    """
    Filter to keep top-K spans with lowest per-token unigram score.
      - K = ceil(top_k_ratio * L) where L = len(response_ids).
      - Spans are scored by per-token average log unigram count; lower = more unique.
    """
    L = len(response_ids)
    K = math.ceil(top_k_ratio * L)
    logger.info("  Filtering to top-K=%d spans (from %d maximal spans, L=%d)",
                K, len(maximal_spans), L)

    if len(maximal_spans) <= K:
        logger.info("  All %d spans kept (fewer than K=%d)", len(maximal_spans), K)
        return maximal_spans

    # Score each span
    unigram_cache = {}
    scored_spans = []
    for (b, e) in maximal_spans:
        score = compute_span_score(engine, response_ids[b:e], unigram_cache,
                                   corpus_size)
        scored_spans.append((b, e, score))

    # Sort by score ascending (lowest = most unique)
    scored_spans.sort(key=lambda x: x[2])

    # Keep top-K, re-sort by position
    top_k = [(b, e) for (b, e, _) in scored_spans[:K]]
    top_k.sort(key=lambda x: x[0])

    logger.info("  Top-K spans selected: %d", len(top_k))
    return top_k


# ===========================================================================
# Phase 2: Retrieve corpus snippets for top-K spans (OLMoTrace Step 3)
# ===========================================================================
def retrieve_snippets_for_span(engine, span_ids, max_docs, max_disp_len, tokenizer):
    """
    Retrieve up to max_docs document snippets containing the given span.
      - Uses engine.find() → engine.get_doc_by_rank() for each occurrence.
      - If span occurs more than max_docs times, randomly samples max_docs.
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

        # Sample if too many occurrences
        remaining = max_docs - docs_retrieved
        if count_in_shard <= remaining:
            ranks = range(start_rank, end_rank)
        else:
            import random
            ranks = sorted(random.sample(range(start_rank, end_rank), remaining))

        for rank in ranks:
            if docs_retrieved >= max_docs:
                break
            try:
                if isinstance(engine, InfiniGramAPIEngine):
                    doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=max_disp_len, query_ids=span_ids)
                else:
                    doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=max_disp_len)
                    
                if "error" in doc:
                    continue

                snippet_info = {
                    "doc_ix": doc.get("doc_ix"),
                    "doc_len": doc.get("doc_len"),
                    "metadata": doc.get("metadata", ""),
                }

                if isinstance(engine, InfiniGramAPIEngine) and "spans" in doc:
                    # API mode: snippet text is in spans[0][0]
                    spans_data = doc.get("spans", [])
                    if spans_data and spans_data[0][0]:
                        snippet_info["snippet_text"] = "".join(spans_data[0])
                    else:
                        snippet_info["snippet_text"] = ""
                    snippet_info["snippet_token_ids"] = doc.get("token_ids", [])
                elif "token_ids" in doc and tokenizer is not None:
                    # Local mode: decode token_ids
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
def compute_e1_metrics(response_ids, maximal_spans, top_k_spans, tokenizer):
    """
    Compute E1 metrics from spans.

    Metrics:
      - LongestMatchLen: max span length (tokens) among all maximal spans
      - VerbatimCoverage: fraction of response tokens covered by any maximal span
      - span_length_distribution: min, max, mean, median of all maximal span lengths
      - all_maximal_spans: full list of maximal spans
      - top_k_spans: top-K span details with decoded text
    """
    L = len(response_ids)
    all_lengths = [e - b for (b, e) in maximal_spans]

    # LongestMatchLen
    longest = max(all_lengths) if all_lengths else 0

    # VerbatimCoverage: fraction of tokens covered by any maximal span
    covered = set()
    for (b, e) in maximal_spans:
        for t in range(b, e):
            covered.add(t)
    coverage = len(covered) / L if L > 0 else 0.0

    # Top-K span details with decoded text
    top_k_details = []
    for (b, e) in top_k_spans:
        span_text = ""
        if tokenizer is not None:
            try:
                span_text = tokenizer.decode(response_ids[b:e],
                                             skip_special_tokens=False)
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
# Engine initialization helper
# ===========================================================================
def init_engine(args, tokenizer, logger):
    """Initialize either a local InfiniGramEngine or an InfiniGramAPIEngine."""
    if args.index_dir is not None:
        # ---- Local engine (GPT-J / Pile-train) ----
        index_dir = args.index_dir
        logger.info("Using LOCAL engine from: %s", index_dir)

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

        num_shards = len([f for f in os.listdir(index_dir)
                          if f.startswith("tokenized.")])
        logger.info("Detected %d shard(s) in index", num_shards)

        # Get corpus size from engine
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (local): %d tokens", corpus_size)

        if corpus_size == 0:
            logger.warning("Corpus size is 0! Falling back to hardcoded. Pile-train size: 383,299,358,652")
            corpus_size = 383_299_358_652

    else:
        # ---- API engine (OLMo 2 / olmo-mix-1124) ----
        logger.info("Using API engine: index=%s", args.api_index)
        engine = InfiniGramAPIEngine(index=args.api_index)

        # num_shards is not used by the algorithm itself, but we keep the variable for interface consistency.
        # The API handles sharding internally.
        num_shards = 0

        # Get corpus size via empty-string count query
        logger.info("Querying corpus size from API (empty-string count)...")
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (API, %s): %d tokens",
                    args.api_index, corpus_size)

        if corpus_size == 0:
            logger.error("Failed to retrieve corpus size from API. "
                         "Cannot compute span scores.")
            sys.exit(1)

    return engine, num_shards, corpus_size


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()
    configs = _resolve_configs(args)
    if not configs:
        configs = [args.config]

    multi = len(configs) > 1
    model_dir = MODEL_CONFIGS[args.model]["out_dir_name"]
    logger0 = setup_logger(args.model, configs[0])
    if multi:
        logger0.info("Running configs: %s", ", ".join(configs))

    # Initialize tokenizer (shared across configs)
    logger0.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    # Initialize engine (shared across configs)
    engine, num_shards, corpus_size = init_engine(args, tokenizer, logger0)

    def _format_path(path_template: str, config: str) -> str:
        return path_template.format(model_dir=model_dir, config=config)

    for config in configs:
        logger = setup_logger(args.model, config)

        if args.input is None:
            input_path = os.path.join("data", model_dir, f"harmbench_{config}_labeled.json")
        else:
            if ("{config}" in args.input) or ("{model_dir}" in args.input):
                input_path = _format_path(args.input, config)
            elif multi:
                base, ext = os.path.splitext(args.input)
                input_path = f"{base}_{config}{ext}"
            else:
                input_path = args.input

        if args.output is None:
            sub = e1_results_subdir(args)
            if sub:
                output_path = os.path.join("results", model_dir, sub, f"e1_verbatim_{config}.json")
            else:
                output_path = os.path.join("results", model_dir, f"e1_verbatim_{config}.json")
        else:
            if ("{config}" in args.output) or ("{model_dir}" in args.output):
                output_path = _format_path(args.output, config)
            elif multi:
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_{config}{ext}"
            else:
                output_path = args.output

        logger.info("=== E1 Verbatim Trace started at %s ===",
                    datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
        logger.info("Config: %s", config)
        logger.info(
            "Arguments: %s",
            {**vars(args), "config": config, "input": input_path, "output": output_path},
        )
        start_time = time.time()

        # Determine backend mode for logging
        if args.index_dir is not None:
            logger.info("Backend: LOCAL engine (--index_dir=%s)", args.index_dir)
        else:
            logger.info("Backend: HTTP API (--api_index=%s)", args.api_index)

        # Load input data
        logger.info("Loading input from %s ...", input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info("Loaded %d records total", len(records))

        # Filter records
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
            logger.warning("No records to process for config=%s. Skipping.", config)
            continue

        logger.info("CORPUS_SIZE = %d", corpus_size)

        # Process each record (with incremental save + resume)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Resume: load existing results if output file exists
        results = []
        completed_ids = set()
        if os.path.isfile(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                for r in results:
                    if "error" not in r.get("e1", {}):
                        completed_ids.add(r.get("id"))
                logger.info("Resumed from %s: %d existing results (%d successful)",
                            output_path, len(results), len(completed_ids))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Could not load existing output (%s), starting fresh.", e)
                results = []
                completed_ids = set()

        for rec_idx, record in enumerate(target_records):
            rec_id = record.get("id", rec_idx)
            prompt = record.get("prompt", "")
            response = record.get("response", "")

            # Skip if already successfully completed
            if rec_id in completed_ids:
                logger.info("[%d/%d] Skipping record id=%s (already completed)",
                            rec_idx + 1, len(target_records), rec_id)
                continue

            logger.info("=" * 70)
            logger.info("[%d/%d] Processing record id=%s",
                        rec_idx + 1, len(target_records), rec_id)
            logger.info("  Prompt: %s", prompt[:100])
            logger.info("  Response length: %d chars", len(response))

            rec_start = time.time()

            try:
                response_ids = tokenizer.encode(response)
                L = len(response_ids)
                logger.info("  Tokenized response: %d tokens", L)

                # Phase 1: Compute maximal matching spans
                maximal_spans = compute_maximal_matching_spans(
                    engine, response_ids, num_shards, logger
                )

                # Filter to top-K spans
                top_k_spans = filter_top_k_spans(
                    engine, response_ids, maximal_spans, args.top_k_ratio,
                    corpus_size, logger
                )

                # Compute E1 metrics
                e1_metrics = compute_e1_metrics(
                    response_ids, maximal_spans, top_k_spans, tokenizer
                )

                # Phase 2 (optional): Retrieve snippets for top-K spans
                if args.retrieve_snippets:
                    logger.info("  Phase 2: Retrieving snippets for %d top-K spans...",
                                len(top_k_spans))
                    snippets_by_span = []
                    for span_idx, (b, e) in enumerate(top_k_spans):
                        snippets = retrieve_snippets_for_span(
                            engine, response_ids[b:e],
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
                logger.info("  E1 results: LongestMatchLen=%d, VerbatimCoverage=%.4f, "
                            "spans=%d/%d, elapsed=%.1fs",
                            e1_metrics["LongestMatchLen"],
                            e1_metrics["VerbatimCoverage"],
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

            # Incremental save after each record
            logger.info("  Saving %d results to %s ...", len(results), output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        # Summary
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