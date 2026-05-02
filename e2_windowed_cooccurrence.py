#!/usr/bin/env python3
"""E2 Windowed Co-occurrence: pairwise infini-gram CNF co-occurrence over
ranked concepts (e2_concepts_ranked_{config}.json).

Pipeline position:
    e2/{{e2_llm}}/e2_concepts_ranked_{config}.json   (Stage 2 output, stage-independent)
          |
          v
    [This script] e2_windowed_cooccurrence.py   (queries the {phase}-specific corpus index)
          |
          v
    e2/{{e2_llm}}/{phase}/e2_cooccurrence_{config}.json   (Stage 3 output, per-phase)

Method:
  - Load top-N ranked concepts from e2_concepts_ranked_{config}.json (sorted by rank).
  - For each pair (concept_a, concept_b), query infini-gram with a CNF AND query:
      count_cnf([[ids_A], [ids_B]], max_diff_tokens=w)
  - Repeat for multiple window sizes (e.g., 100, 500, 1000 tokens).

E1 vs. E2 independence:
  Concept extraction operates on the full response and is NOT constrained
  by E1 verbatim trace results. The two evidence layers remain statistically
  independent.

Usage:
    # OLMo 2 (API engine, multiple top_n values)
    python e2_windowed_cooccurrence.py \
        --model olmo2-7b-instruct \
        --config standard \
        --training-phase pretraining \
        --api_index v4_olmo-mix-1124_llama \
        --top_n 5 10 15 20 \
        --compliant_only \
        --windows 100 500 1000

    python e2_windowed_cooccurrence.py --model olmo2-1b --config standard --training-phase mid_training --index_dir ./index/dolmino-mix-1124 --top_n 5 10 15 20 --windows 100 500 1000 --compliant_only --e2-llm gpt-5-mini

    python e2_windowed_cooccurrence.py --model olmo2-7b-instruct --config standard --training-phase post_training --index_dir ./index/post_training/7b --top_n 5 10 15 20 --windows 100 500 1000 --compliant_only --e2-llm gpt-5-mini 

    # Override paths explicitly (must match your --config filenames)
    python e2_windowed_cooccurrence.py \
        --model olmo2-7b-instruct \
        --config standard \
        --training-phase pretraining \
        --concepts_input results/olmo2_7b_instruct/e2/gpt-5.4-mini/e2_concepts_ranked_standard.json \
        --top_n 15 \
        --windows 100 500 1000

    # Run all phases (base: pre+mid, instruct: pre+mid+post)
    python e2_windowed_cooccurrence.py \
        --model olmo2-7b-instruct \
        --config standard \
        --training-phase all \
        --top_n 10 \
        --compliant_only

    # Multiple top_n in one command (output suffixed: *_top{N}.json)
    python e2_windowed_cooccurrence.py \
        --model olmo2-7b-instruct \
        --config standard \
        --training-phase pretraining \
        --top_n 5 10 20 \
        --compliant_only
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from itertools import combinations

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer

from infini_gram_api import InfiniGramAPIEngine

from utils import (
    MODELS,
    e1_phase_root,
    e2_llm_root,
    e2_cooc_root,
    training_phases_when_all,
    setup_logger,
)

DEFAULT_PRETRAINING_API_INDEX = "v4_olmo-mix-1124_llama"
DEFAULT_MIDTRAINING_INDEX_DIR = os.path.join(".", "index", "dolmino-mix-1124")
DEFAULT_POSTTRAINING_INDEX_ROOT = os.path.join(".", "index", "post-training")


def _model_size_suffix(model_key: str) -> str | None:
    """Extract size suffix like '1b', '7b', '13b', '32b' from model key."""
    for s in ("1b", "7b", "13b", "32b"):
        if f"-{s}" in model_key:
            return s
    return None


def _resolve_backend_for_all(args, training_phase: str) -> argparse.Namespace:
    """Phase-specific backend selection when --training-phase all.

    Rules:
      - pretraining  : HTTP API (api_index)
      - mid_training : local index_dir = ./index/dolmino-mix-1124
      - post_training: local index_dir = ./index/post-training/{size}
    """
    phase_args = argparse.Namespace(**vars(args))

    if training_phase == "pretraining":
        phase_args.index_dir = None
        phase_args.api_index = args.api_index or DEFAULT_PRETRAINING_API_INDEX
        return phase_args

    if training_phase == "mid_training":
        phase_args.index_dir = DEFAULT_MIDTRAINING_INDEX_DIR
        return phase_args

    if training_phase == "post_training":
        size = _model_size_suffix(args.model)
        if size is None:
            raise ValueError(
                f"Cannot infer model size for post_training index from --model '{args.model}'. "
                "Expected one of: *-1b*, *-7b*, *-13b*, *-32b*."
            )
        phase_args.index_dir = os.path.join(DEFAULT_POSTTRAINING_INDEX_ROOT, size)
        return phase_args

    raise ValueError(f"Unknown training phase: {training_phase}")


# ===========================================================================
# Argument parsing
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="E2 Windowed Co-occurrence: CNF-based corpus co-occurrence"
    )

    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()),
                        help="Model key (determines auto paths and log dir)")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench config name (default: standard)")
    parser.add_argument(
        "--training-phase",
        type=str,
        required=True,
        choices=("pretraining", "mid_training", "post_training", "all"),
        dest="training_phase",
        help="Phase that determines which infini-gram index Stage 3 queries against, "
             "and the per-phase Stage 3 output subdir. "
             "Use 'all' to run once per phase: base models run "
             "pretraining+mid_training; *-instruct models run pretraining+mid_training+post_training.",
    )

    # I/O
    parser.add_argument("--input", type=str, default=None,
                        help="E1 verbatim trace JSON (per-phase). "
                             "Default: results/{model_dir}/e1/{training_phase}/e1_verbatim_{config}.json")
    parser.add_argument("--concepts_input", type=str, default=None,
                        help="Ranked concepts JSON from e2_rank_concepts.py (stage-independent). "
                             "Default: results/{model_dir}/e2/{e2_llm}/e2_concepts_ranked_{config}.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON (per-phase). "
                             "Default: results/{model_dir}/e2/{e2_llm}/{training_phase}/e2_cooccurrence_{config}.json")
    parser.add_argument(
        "--e2-llm",
        type=str,
        default="gpt-5-mini",
        dest="e2_llm",
        metavar="MODEL",
        help="E2 LLM subfolder name (must match e2_extract_concepts / e2_rank_concepts). "
             "Stage 1+2 outputs live at results/{model_dir}/e2/{e2_llm}/ (stage-independent); "
             "Stage 3 outputs live at results/{model_dir}/e2/{e2_llm}/{training_phase}/. "
             "Default: %(default)s.",
    )

    # Backend
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Local infini-gram index dir. If omitted, use HTTP API.")
    parser.add_argument("--api_index", type=str,
                        default=DEFAULT_PRETRAINING_API_INDEX,
                        help=f"API index name (default: {DEFAULT_PRETRAINING_API_INDEX})")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")

    # E2 parameters
    parser.add_argument(
        "--top_n",
        type=int,
        nargs="+",
        default=None,
        help="Use only the top-N ranked concepts per record "
             "(sorted by rank ascending). "
             "Provide one or more values to run multiple passes, e.g. "
             "--top_n 5 10 20. Default: None (use all).",
    )
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Window sizes (max_diff_tokens) to test "
                             "(default: 100 500 1000)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max concept pairs to query per record. "
                             "Default: None (all C(n,2) pairs)")
    parser.add_argument("--max_clause_freq", type=int, default=None,
                        help="Max clause frequency for CNF queries (default: no limit)")

    # Processing
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit records to process (for testing)")
    parser.add_argument("--compliant_only", action="store_true",
                        help="Process only compliant (hb_label=1) records")
    parser.add_argument("--all_records", action="store_true",
                        help="Process all records regardless of hb_label")

    return parser.parse_args()


def _with_topn_suffix(output_path: str, top_n: int) -> str:
    """Append _top{N} before file extension."""
    root, ext = os.path.splitext(output_path)
    ext = ext or ".json"
    return f"{root}_top{top_n}{ext}"


def resolve_phase_paths(args, training_phase: str, logger=None) -> tuple[str, str, str]:
    """Resolve (input, concepts_input, output) for a specific training phase.

    E1 verbatim defaults to ``results/{out_dir}/e1/{training_phase}/`` (per-phase).
    Ranked concepts default to ``results/{out_dir}/e2/{e2_llm}/`` (stage-independent
    — Stage 1+2 outputs are corpus-independent and shared across all phases).
    Cooccurrence output defaults to ``results/{out_dir}/e2/{e2_llm}/{training_phase}/``
    (per-phase).

    When args.training_phase == 'all', any explicit overrides for --input/--concepts_input/--output
    are ignored to avoid accidentally mixing phases.
    """
    e1_root = e1_phase_root(args.model, training_phase)
    concepts_root = e2_llm_root(args.model, args.e2_llm)
    cooc_root = e2_cooc_root(args.model, args.e2_llm, training_phase)

    ignore_overrides = (args.training_phase == "all")
    if ignore_overrides and logger is not None:
        if args.input or args.concepts_input or args.output:
            logger.warning(
                "--input/--concepts_input/--output overrides are ignored when --training-phase all "
                "(using default paths per phase).",
            )

    if ignore_overrides:
        input_path = os.path.join(e1_root, f"e1_verbatim_{args.config}.json")
        concepts_path = os.path.join(
            concepts_root, f"e2_concepts_ranked_{args.config}.json"
        )
        output_path = os.path.join(cooc_root, f"e2_cooccurrence_{args.config}.json")
        return input_path, concepts_path, output_path

    input_path = args.input or os.path.join(
        e1_root, f"e1_verbatim_{args.config}.json"
    )
    concepts_path = args.concepts_input or os.path.join(
        concepts_root, f"e2_concepts_ranked_{args.config}.json"
    )
    output_path = args.output or os.path.join(
        cooc_root, f"e2_cooccurrence_{args.config}.json"
    )
    return input_path, concepts_path, output_path


# ===========================================================================
# Engine initialization
# ===========================================================================
def init_engine(args, tokenizer, logger):
    """Initialize local InfiniGramEngine or InfiniGramAPIEngine.

    Returns (engine, corpus_size).
    """
    if args.index_dir is not None:
        logger.info("Backend: LOCAL engine from %s", args.index_dir)
        if not os.path.isdir(args.index_dir):
            logger.error("Index directory not found: %s", args.index_dir)
            sys.exit(1)
        from infini_gram.engine import InfiniGramEngine
        engine = InfiniGramEngine(
            s3_names=[],
            index_dir=args.index_dir,
            eos_token_id=tokenizer.eos_token_id,
            version=4,
            token_dtype="u16",
        )
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (local): %d tokens", corpus_size)
        if corpus_size == 0:
            logger.warning(
                "Corpus size is 0; falling back to hardcoded "
                "v4_olmo-mix-1124_llama size: 4,575,475,702,047 "
            )
            corpus_size = 4_575_475_702_047
    else:
        logger.info("Backend: HTTP API index=%s", args.api_index)
        engine = InfiniGramAPIEngine(
            index=args.api_index,
            max_retries=5,
            retry_delay=2.0,
        )
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (API, %s): %d tokens", args.api_index, corpus_size)
        if corpus_size == 0:
            logger.error("Failed to retrieve corpus size from API.")
            sys.exit(1)

    return engine, corpus_size


# ===========================================================================
# Concept loading from e2_concepts_ranked_{config}.json
# ===========================================================================
def load_concepts_from_ranked(concepts_path, logger):
    """Load ranked concepts from Stage 2 JSON (e2_concepts_ranked_{config}.json).

    Returns
    -------
    concepts_by_id : dict[int, dict]
        {record_id: {"concepts": [...ranked_concepts...],
                     "rank_model": str,
                     "rank_prompt_version": str}}

    Each concept dict has: text, rank, centrality, note.
    """
    if not os.path.isfile(concepts_path):
        logger.error("Ranked concepts file not found: %s", concepts_path)
        logger.error("Run e2_rank_concepts.py first.")
        sys.exit(1)

    with open(concepts_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    concepts_by_id = {}
    for row in rows:
        rec_id = row.get("id")
        if rec_id is None:
            continue
        concepts_by_id[rec_id] = {
            "concepts": row.get("ranked_concepts", []),
            "rank_model": row.get("rank_model", "unknown"),
            "rank_prompt_version": row.get("rank_prompt_version", ""),
        }

    logger.info("Loaded ranked concepts for %d records from %s",
                len(concepts_by_id), concepts_path)
    return concepts_by_id


# ===========================================================================
# Token-level concept preparation
# ===========================================================================
def _find_subsequence(sequence, subsequence):
    """Return index of first occurrence of subsequence in sequence, or -1."""
    sub_len = len(subsequence)
    if sub_len == 0:
        return -1
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            return i
    return -1


def _concept_text_to_ids_and_position(concept_text, response_text,
                                       response_ids, tokenizer):
    """Convert concept text to token IDs and locate its position in response.

    Strategy:
      1. Direct tokenization.
      2. Token-level subsequence match in response_ids.
      3. Case-insensitive substring fallback with BPE-alignment trial.
      4. If not found, return position=-1 but keep token IDs.

    Returns (token_ids, position).
    """
    token_ids = tokenizer.encode(concept_text, add_special_tokens=False)
    if token_ids and tokenizer.bos_token_id is not None \
            and token_ids[0] == tokenizer.bos_token_id:
        token_ids = token_ids[1:]
    if not token_ids:
        return [], -1

    position = _find_subsequence(response_ids, token_ids)
    if position >= 0:
        return token_ids, position

    # Fallback: case-insensitive substring + BPE alignment
    lower_response = response_text.lower()
    char_pos = lower_response.find(concept_text.lower())
    if char_pos >= 0:
        substring = response_text[char_pos:char_pos + len(concept_text)]
        for trial in (" " + substring, substring):
            trial_ids = tokenizer.encode(trial, add_special_tokens=False)
            if trial_ids and tokenizer.bos_token_id is not None \
                    and trial_ids[0] == tokenizer.bos_token_id:
                trial_ids = trial_ids[1:]
            if trial_ids:
                pos = _find_subsequence(response_ids, trial_ids)
                if pos >= 0:
                    return trial_ids, pos

    return token_ids, -1


def prepare_concepts_for_record(record_id, concepts_by_id, response_text,
                                 response_ids, tokenizer, logger, top_n=None):
    """Convert ranked concept strings to dicts ready for CNF queries.

    Applies top_n cutoff (by rank ascending) before tokenization.

    Each output dict:
        ngram_ids   : list[int]  — token IDs for CNF query
        text        : str
        note        : str        — descriptive label from Stage 2
        rank        : int        — centrality rank from Stage 2
        centrality  : str        — tier label (topic_core/primary/supporting/peripheral)
        position    : int        — token position in response (-1 if not found)
        length      : int        — token count
    """
    entry = concepts_by_id.get(record_id)
    if entry is None:
        logger.warning("    No ranked concepts for record id=%s", record_id)
        return [], {"rank_model": "unknown", "rank_prompt_version": ""}

    raw_concepts = sorted(entry["concepts"], key=lambda c: c.get("rank", 9999))
    total_available = len(raw_concepts)

    if top_n is not None:
        raw_concepts = raw_concepts[:top_n]
        logger.info("    top_n=%d: using %d / %d ranked concepts",
                    top_n, len(raw_concepts), total_available)

    extraction_meta = {
        "rank_model": entry["rank_model"],
        "rank_prompt_version": entry["rank_prompt_version"],
        "top_n_applied": top_n,
        "total_ranked": total_available,
    }

    prepared = []
    n_unmatched = 0
    n_empty = 0

    for c in raw_concepts:
        text = (c.get("text") or "").strip()
        if not text:
            n_empty += 1
            continue

        token_ids, position = _concept_text_to_ids_and_position(
            text, response_text, response_ids, tokenizer)

        if not token_ids:
            n_empty += 1
            continue

        if position < 0:
            n_unmatched += 1

        prepared.append({
            "ngram_ids": token_ids,
            "text": text,
            "note": c.get("note", ""),
            "rank": c.get("rank"),
            "centrality": c.get("centrality", ""),
            "position": position,
            "length": len(token_ids),
        })

    logger.info("    Prepared %d concepts for record id=%s "
                "(position unmatched: %d, empty/skipped: %d)",
                len(prepared), record_id, n_unmatched, n_empty)
    return prepared, extraction_meta


# ===========================================================================
# Pairwise CNF co-occurrence queries
# ===========================================================================
def compute_pairwise_cooccurrence(engine, concepts, windows, max_pairs,
                                   max_clause_freq, logger):
    """Query co-occurrence count for all concept pairs at each window size.

    CNF format: cnf = [[[ids_A]], [[ids_B]]]
    count_cnf(cnf, max_diff_tokens=w) counts positions where A and B
    appear within w tokens of each other in the corpus.

    Returns
    -------
    {
      "pairs": [
        {
          "concept_a_idx": int,
          "concept_b_idx": int,
          "concept_a": {text, ngram_ids, position, note, rank, centrality},
          "concept_b": {text, ngram_ids, position, note, rank, centrality},
          "counts_by_window": {w: {"count": int, "approx": bool}, ...}
        },
        ...
      ],
      "summary_by_window": {
        w: {"total_pairs": int, "nonzero_pairs": int,
            "max_count": int, "mean_count": float, "median_count": int},
        ...
      }
    }
    """
    all_pairs = list(combinations(range(len(concepts)), 2))
    if max_pairs is not None and len(all_pairs) > max_pairs:
        logger.info("    Truncating pairs: %d → %d (--max_pairs)",
                    len(all_pairs), max_pairs)
        all_pairs = all_pairs[:max_pairs]

    logger.info("    Querying %d pairs × %d windows = %d CNF calls",
                len(all_pairs), len(windows), len(all_pairs) * len(windows))

    pair_results = []
    query_count = 0

    for pair_idx, (i, j) in enumerate(all_pairs):
        ca = concepts[i]
        cb = concepts[j]

        cnf = [[ca["ngram_ids"]], [cb["ngram_ids"]]]

        counts_by_window = {}
        for w in windows:
            try:
                result = engine.count_cnf(
                    cnf=cnf,
                    max_clause_freq=max_clause_freq,
                    max_diff_tokens=w,
                )
                counts_by_window[w] = {
                    "count": result.get("count", 0),
                    "approx": result.get("approx", False),
                }
            except Exception as exc:
                counts_by_window[w] = {"count": -1, "error": str(exc)}
            query_count += 1

        pair_results.append({
            "concept_a_idx": i,
            "concept_b_idx": j,
            "concept_a": {
                "text": ca["text"],
                "ngram_ids": ca["ngram_ids"],
                "position": ca["position"],
                "note": ca.get("note", ""),
                "rank": ca.get("rank"),
                "centrality": ca.get("centrality", ""),
            },
            "concept_b": {
                "text": cb["text"],
                "ngram_ids": cb["ngram_ids"],
                "position": cb["position"],
                "note": cb.get("note", ""),
                "rank": cb.get("rank"),
                "centrality": cb.get("centrality", ""),
            },
            "counts_by_window": counts_by_window,
        })

        if (pair_idx + 1) % 20 == 0:
            logger.info("      Pair %d / %d done (%d queries so far)",
                        pair_idx + 1, len(all_pairs), query_count)

    # Summary per window
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
                "total_pairs": 0, "nonzero_pairs": 0,
                "max_count": 0, "mean_count": 0, "median_count": 0,
            }

    logger.info("    Completed %d CNF queries total", query_count)
    return {"pairs": pair_results, "summary_by_window": summary_by_window}


def mark_all_pairs_zero(concepts, cooc_result, windows):
    """Mark each concept with all_pairs_zero=True if all its pairs had
    zero co-occurrence across all windows. Mutates concepts in place."""
    n = len(concepts)
    has_any_nonzero = [False] * n

    for pair in cooc_result.get("pairs", []):
        i = pair.get("concept_a_idx")
        j = pair.get("concept_b_idx")
        if i is None or j is None:
            continue
        for w in windows:
            if pair["counts_by_window"].get(w, {}).get("count", 0) > 0:
                if 0 <= i < n:
                    has_any_nonzero[i] = True
                if 0 <= j < n:
                    has_any_nonzero[j] = True
                break

    for idx, c in enumerate(concepts):
        c["all_pairs_zero"] = not has_any_nonzero[idx]


# ===========================================================================
# E2 metrics
# ===========================================================================
def compute_e2_metrics(cooccurrence_result, windows):
    """Compute aggregate E2 metrics from pairwise co-occurrence results.

    Per window:
      E2_cooc(w)         — max co-occurrence count across all pairs
      E2_nonzero_frac(w) — fraction of pairs with nonzero co-occurrence
      E2_mean(w)         — mean co-occurrence count
      E2_median(w)       — median co-occurrence count

    E2_support_score = max_w log(1 + E2_cooc(w))
    """
    summary = cooccurrence_result["summary_by_window"]
    metrics_by_window = {}

    for w in windows:
        s = summary.get(w, {})
        total = s.get("total_pairs", 0)
        metrics_by_window[w] = {
            "E2_cooc": s.get("max_count", 0),
            "E2_nonzero_frac": (round(s.get("nonzero_pairs", 0) / total, 4)
                                if total > 0 else 0.0),
            "E2_mean": s.get("mean_count", 0),
            "E2_median": s.get("median_count", 0),
            "total_pairs": total,
            "nonzero_pairs": s.get("nonzero_pairs", 0),
        }

    support_score = max(
        (math.log(1 + metrics_by_window[w]["E2_cooc"]) for w in windows),
        default=0.0
    )

    return {
        "metrics_by_window": metrics_by_window,
        "E2_support_score": round(support_score, 4),
        "windows_tested": windows,
    }


# ===========================================================================
# Main
# ===========================================================================
def run_one_phase(args, training_phase: str, logger, *, phase_index: int, total_phases: int,
                  top_n: int | None, multi_topn: bool) -> None:
    phase_args = (
        _resolve_backend_for_all(args, training_phase)
        if args.training_phase == "all"
        else args
    )

    input_path, concepts_path, base_output_path = resolve_phase_paths(phase_args, training_phase, logger=logger)
    output_path = (
        _with_topn_suffix(base_output_path, top_n)
        if (multi_topn and top_n is not None)
        else base_output_path
    )

    logger.info("=" * 70)
    logger.info("E2 Windowed Co-occurrence")
    if total_phases > 1:
        logger.info("  Phase %d / %d: %s", phase_index + 1, total_phases, training_phase)
    logger.info("  Model: %s", args.model)
    logger.info("  Config: %s", args.config)
    logger.info(
        "  --e2-llm: %s (ranked concepts at %s; cooccurrence under %s)",
        phase_args.e2_llm,
        e2_llm_root(args.model, phase_args.e2_llm),
        e2_cooc_root(args.model, phase_args.e2_llm, training_phase),
    )
    logger.info("  top_n: %s", top_n if top_n is not None else "ALL")
    logger.info("  Input (E1): %s", input_path)
    logger.info("  Input (ranked concepts): %s", concepts_path)
    logger.info("  Output: %s", output_path)
    logger.info("=" * 70)

    logger.info("Backend: %s",
                f"LOCAL ({phase_args.index_dir})" if phase_args.index_dir
                else f"HTTP API ({phase_args.api_index})")
    start_time = time.time()

    # Load E1 records
    logger.info("Loading E1 records from %s ...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records total", len(records))

    # Filter records
    if args.compliant_only:
        target_records = [r for r in records if r.get("hb_label") == 1]
        logger.info("Filtered to %d compliant (hb_label=1) records",
                    len(target_records))
    else:
        target_records = records
        logger.info("Processing all %d records", len(target_records))

    if args.limit is not None and args.limit > 0:
        target_records = target_records[:args.limit]
        logger.info("Limited to %d records (--limit)", len(target_records))

    if not target_records:
        logger.warning("No records to process. Exiting.")
        return

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
    engine, corpus_size = init_engine(phase_args, tokenizer, logger)
    logger.info("CORPUS_SIZE = %d", corpus_size)

    # Warn if API windows exceed limit
    if phase_args.index_dir is None:
        over_limit = [w for w in args.windows if w > 1000]
        if over_limit:
            logger.warning(
                "API max_diff_tokens limit is 1000. "
                "Windows %s will be clamped. Use --index_dir for w > 1000.",
                over_limit)

    logger.info("Windows: %s", args.windows)
    logger.info("Max pairs per record: %s",
                "ALL" if args.max_pairs is None else args.max_pairs)
    logger.info("Ranked concepts input: %s", concepts_path)

    # Load ranked concepts
    concepts_by_id = load_concepts_from_ranked(concepts_path, logger)

    # Resume from existing output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    results = []
    completed_ids = set()
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                if "error" not in r.get("e2", {}):
                    completed_ids.add(r.get("id"))
            logger.info("Resumed from %s: %d existing (%d successful)",
                        output_path, len(results), len(completed_ids))
        except Exception as e:
            logger.warning("Could not load existing output (%s); starting fresh.", e)
            results = []
            completed_ids = set()

    # Process each record
    for rec_idx, record in enumerate(target_records):
        rec_id = record.get("id", rec_idx)
        response = record.get("response", "")

        if rec_id in completed_ids:
            logger.info("[%d/%d] Skipping record id=%s (already completed)",
                        rec_idx + 1, len(target_records), rec_id)
            continue

        logger.info("=" * 70)
        logger.info("[%d/%d] Processing record id=%s",
                    rec_idx + 1, len(target_records), rec_id)
        logger.info("  Response length: %d chars", len(response))

        rec_start = time.time()

        try:
            response_ids = tokenizer.encode(response)
            logger.info("  Tokenized response: %d tokens", len(response_ids))

            # Step 1: prepare concepts (top_n applied here)
            logger.info("  Step 1: Preparing ranked concepts...")
            concepts, extraction_meta = prepare_concepts_for_record(
                rec_id, concepts_by_id,
                response_text=response,
                response_ids=response_ids,
                tokenizer=tokenizer,
                logger=logger,
                top_n=top_n,
            )

            if len(concepts) < 2:
                logger.warning("  Only %d concepts; skipping co-occurrence",
                               len(concepts))
                e2_metrics = {
                    "metrics_by_window": {w: {} for w in args.windows},
                    "E2_support_score": 0.0,
                    "windows_tested": args.windows,
                    "num_concepts": len(concepts),
                    "num_pairs_queried": 0,
                    "note": "Too few concepts for pairwise queries",
                }
                cooc_result = {"pairs": [], "summary_by_window": {}}
                for c in concepts:
                    c["all_pairs_zero"] = True
            else:
                # Step 2: pairwise CNF co-occurrence
                logger.info("  Step 2: Computing pairwise co-occurrence...")
                cooc_result = compute_pairwise_cooccurrence(
                    engine, concepts, args.windows,
                    max_pairs=args.max_pairs,
                    max_clause_freq=args.max_clause_freq,
                    logger=logger,
                )

                # Step 3: E2 metrics
                e2_metrics = compute_e2_metrics(cooc_result, args.windows)
                e2_metrics["num_concepts"] = len(concepts)
                e2_metrics["num_pairs_queried"] = len(cooc_result["pairs"])
                mark_all_pairs_zero(concepts, cooc_result, args.windows)

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
                    "rank_model": extraction_meta.get("rank_model", "unknown"),
                    "rank_prompt_version": extraction_meta.get(
                        "rank_prompt_version", ""),
                    "top_n_applied": extraction_meta.get("top_n_applied"),
                    "total_ranked": extraction_meta.get("total_ranked"),
                    "ranked_concepts": concepts,
                    "pairwise_cooccurrence": cooc_result,
                },
            }

            rec_elapsed = time.time() - rec_start
            for w in args.windows:
                wm = e2_metrics.get("metrics_by_window", {}).get(w, {})
                logger.info(
                    "  w=%d: E2_cooc=%s, nonzero=%s/%s, mean=%s",
                    w, wm.get("E2_cooc", "N/A"),
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

        # Incremental save
        logger.info("  Saving %d results to %s ...", len(results), output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Final summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    successful = [r for r in results if "error" not in r.get("e2", {})]
    logger.info("  Total processed : %d", len(results))
    logger.info("  Successful      : %d", len(successful))
    logger.info("  Errors          : %d", len(results) - len(successful))

    if successful:
        for w in args.windows:
            scores = [
                r["e2"]["metrics_by_window"].get(w, {}).get("E2_cooc", 0)
                for r in successful
                if w in r["e2"].get("metrics_by_window", {})
            ]
            if scores:
                logger.info(
                    "  w=%d: E2_cooc max=%d, mean=%.1f, nonzero=%d/%d records",
                    w, max(scores), sum(scores) / len(scores),
                    sum(1 for s in scores if s > 0), len(scores))

        support_scores = [r["e2"].get("E2_support_score", 0)
                          for r in successful]
        logger.info("  E2_support_score: min=%.4f, max=%.4f, mean=%.4f",
                    min(support_scores), max(support_scores),
                    sum(support_scores) / len(support_scores))

    elapsed_float = time.time() - start_time
    elapsed = int(elapsed_float)
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("=== Finished at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("=== Elapsed: %d days %02d:%02d:%02d (%.3fs) ===",
                days, hours, minutes, seconds, elapsed_float)


def main():
    args = parse_args()
    logger = setup_logger(
        args.model, "e2_windowed_cooccurrence", config=args.config)

    logger.info("=== E2 Windowed Co-occurrence started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Arguments: %s", vars(args))

    phases = (
        list(training_phases_when_all(args.model))
        if args.training_phase == "all"
        else [args.training_phase]
    )
    n = len(phases)

    top_ns = args.top_n[:] if args.top_n is not None else [None]
    if args.top_n is not None:
        # Normalize: remove duplicates but preserve user order.
        seen = set()
        normalized = []
        for x in top_ns:
            if x in seen:
                continue
            seen.add(x)
            normalized.append(x)
        top_ns = normalized

    multi_topn = (len(top_ns) > 1)
    if multi_topn:
        logger.info("Multiple top_n values requested: %s", top_ns)
        logger.info("Output will be suffixed per run: *_top{N}.json")

    for i, tp in enumerate(phases):
        for t_idx, t in enumerate(top_ns):
            if multi_topn and n > 1:
                logger.info("Running (phase=%s) with top_n=%s (%d/%d top_n values)",
                            tp, t if t is not None else "ALL", t_idx + 1, len(top_ns))
            run_one_phase(
                args,
                tp,
                logger,
                phase_index=i,
                total_phases=n,
                top_n=t,
                multi_topn=multi_topn,
            )


if __name__ == "__main__":
    main()
