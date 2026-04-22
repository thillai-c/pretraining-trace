#!/usr/bin/env python3
"""E2 Windowed Co-occurrence: Quantify co-occurrence of enabling concepts
from model responses within pretraining corpus using infini-gram CNF queries.

Supports both local infini-gram engine (GPT-J / The Pile) and HTTP API
(OLMo 2 / olmo-mix-1124), mirroring the dual-backend architecture of
e1_verbatim_trace.py.

Intuition: Even if a response is not copied verbatim, the corpus may contain
co-occurring 'enabling concepts' (key phrases/entities/verbs) within a context
window, making recombination plausible.

Method:
  - Load LLM-extracted enabling concepts from e2_concepts.json
    (produced by e2_extract_concepts.py).
  - For each pair of concepts, query infini-gram with a CNF (AND) query:
    count_cnf([[phrase_A], [phrase_B]], max_diff_tokens=w)
  - This counts how many times phrase_A AND phrase_B appear within w tokens
    of each other in the corpus.
  - Repeat for multiple window sizes (e.g., 100, 500, 1000, 2048 tokens).

Pipeline:
  Step 1: Run e2_extract_concepts.py to produce e2_concepts.json (LLM-based).
  Step 2: Run this script to compute pairwise co-occurrence over the
          extracted concepts.

E1 vs. E2 independence:
  Concept extraction (e2_extract_concepts.py) operates on the full response
  and is NOT constrained by E1 verbatim trace results. This script consumes
  those concepts and queries infini-gram. The two evidence layers (E1 and E2)
  remain statistically independent.

Reference:
  - Project Overview: E2 definitions (Windowed Co-occurrence)
  - infini-gram CNF query API: count_cnf, find_cnf, search_docs_cnf
  - e2_extract_concepts.py: upstream concept extraction (LLM-based)

Usage:
    # GPT-J (local engine, Pile-train index, custom windows)
    python e2_windowed_cooccurrence.py \
        --model gpt-j \
        --index_dir ./index \
        --windows 100 500 1000 1024 2048

    # OLMo 2 (API engine)
    python e2_windowed_cooccurrence.py \
        --model olmo2-1b \
        --api_index v4_olmo-mix-1124_llama \
        --compliant_only \
        --windows 100 500 1000

    python e2_windowed_cooccurrence.py --model olmo2-13b-instruct --api_index v4_olmo-mix-1124_llama --compliant_only --windows 100 500 1000
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from itertools import combinations

import requests
from dotenv import load_dotenv

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer


# Model configuration
MODEL_CONFIGS = {
    "gpt-j":              {"out_dir": "gpt_j_6b"},
    "olmo2-1b":           {"out_dir": "olmo2_1b"},
    "olmo2-7b":           {"out_dir": "olmo2_7b"},
    "olmo2-13b":          {"out_dir": "olmo2_13b"},
    "olmo2-32b":          {"out_dir": "olmo2_32b"},
    "olmo2-1b-instruct":  {"out_dir": "olmo2_1b_instruct"},
    "olmo2-7b-instruct":  {"out_dir": "olmo2_7b_instruct"},
    "olmo2-13b-instruct": {"out_dir": "olmo2_13b_instruct"},
    "olmo2-32b-instruct": {"out_dir": "olmo2_32b_instruct"},
}


# ===========================================================================
# InfiniGramAPIEngine: HTTP API wrapper matching local engine interface
# ===========================================================================
class InfiniGramAPIEngine:
    """Wrapper around the infini-gram HTTP API that mimics the local
    InfiniGramEngine interface for the methods used in E2:
      - count(input_ids)    : count occurrences of a token sequence
      - count_cnf(cnf, ...) : count co-occurrence of multiple clauses within
                              a window (AND query)

    The API endpoint accepts JSON POST requests.  We use ``query_ids``
(list of token IDs) rather than ``query`` (string) so that tokenization
    stays under our control.
    """

    API_URL = "https://api.infini-gram.io/"

    def __init__(self, index: str, max_retries: int = 5,
                 retry_delay: float = 2.0):
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
                    time.sleep(1)  # rate-limit courtesy delay
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

        raise RuntimeError("API request failed after all retries")

    def count(self, input_ids: list) -> dict:
        """Count occurrences of the token ID sequence in the corpus.
        If ``input_ids`` is empty, returns the total number of tokens
        (used to obtain CORPUS_SIZE)."""
        if len(input_ids) == 0:
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

    def count_cnf(self, cnf: list, max_clause_freq: int = None,
                  max_diff_tokens: int = 1000) -> dict:
        """Count co-occurrences using a CNF (AND) query via the HTTP API.

        IMPORTANT — API vs. local engine difference:
          - Local engine: uses a separate method ``engine.count_cnf(cnf=...)``
          - HTTP API: uses ``query_type: "count"`` with ``query_ids`` set to the CNF (triply-nested list).  
            The API auto-detects CNF format from the nesting depth of ``query_ids``.

        API constraints:
          - max_diff_tokens must be in [1, 1000] (API hard limit).
            Values > 1000 are clamped to 1000 with a warning.

        Parameters
        ----------
        cnf : list of list of list[int] CNF clauses. Each clause is a list of disjuncts, each disjunct is a list of token IDs.
            Example: [[[101, 102]], [[201, 202]]] = (token seq [101,102]) AND (token seq [201,202])
        max_clause_freq : int or None
            Maximum allowed clause frequency.  If a clause exceeds this, the API may return an approximation.
            Range: [1, 500000].
        max_diff_tokens : int
            Maximum distance (in tokens) between the matched clauses.
            API limit: [1, 1000].  Values > 1000 are clamped.

        Returns
        -------
        dict with at least {"count": int} and optionally {"approx": bool}.
        """
        # Clamp max_diff_tokens to API limit
        api_max_diff = min(max_diff_tokens, 1000)

        payload = {
            "index": self.index,
            "query_type": "count",
            "query_ids": cnf,
            "max_diff_tokens": api_max_diff,
        }
        if max_clause_freq is not None:
            payload["max_clause_freq"] = max_clause_freq
        return self._post(payload)


# ===========================================================================
# Logger setup (model-based directory, matching e1_verbatim_trace.py)
# ===========================================================================
def setup_logger(model_key: str, config: str = "standard"):
    """Create logger with model-based log directory.
    Log file: ``logs/{model_key}/e2_windowed_cooccurrence.log``
    """
    out_dir = MODEL_CONFIGS[model_key]["out_dir"]
    log_dir = os.path.join("logs", out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(
        log_dir, "e2_windowed_cooccurrence.log"
    )

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
        description="E2 Windowed Co-occurrence: CNF-based corpus "
                    "co-occurrence analysis"
    )

    # Model selection (required) — same as e1_verbatim_trace.py
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (determines auto paths and log dir)")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench config name (default: standard)")

    # I/O (auto-generated if not specified)
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSON (E1 results with e1 field). "
                             "Default: results/{model_dir}/"
                             "e1_verbatim_{config}.json")
    parser.add_argument("--concepts_input", type=str, default=None,
                        help="LLM-extracted concepts JSON (from "
                             "e2_extract_concepts.py). "
                             "Default: results/{model_dir}/e2_concepts.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON with E2 results appended. "
                             "Default: results/{model_dir}/"
                             "e2_cooccurrence_{config}.json")

    # Index config — local vs API (same pattern as e1_verbatim_trace.py)
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

    # E2 parameters
    parser.add_argument("--windows", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Window sizes (max_diff_tokens) to test "
                             "(default: 100 500 1000)")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Maximum number of concept pairs to query "
                             "per record. Default: None (no limit, query all "
                             "C(n,2) pairs)")
    parser.add_argument("--max_clause_freq", type=int, default=None,
                        help="Max clause frequency for CNF queries "
                             "(default: None = no limit)")

    # Processing
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of records to process (for testing)")
    parser.add_argument("--compliant_only", action="store_true",
                        help="Process only compliant (hb_label=1) records")
    parser.add_argument("--all_records", action="store_true",
                        help="Process all records regardless of hb_label "
                             "(default behavior)")

    args = parser.parse_args()

    # Auto-generate input/output paths if not specified
    model_dir = MODEL_CONFIGS[args.model]["out_dir"]
    if args.input is None:
        args.input = os.path.join(
            "results", model_dir,
            f"e1_verbatim_{args.config}.json"
        )
    if args.concepts_input is None:
        args.concepts_input = os.path.join(
            "results", model_dir,
            "e2_concepts.json"
        )
    if args.output is None:
        args.output = os.path.join(
            "results", model_dir,
            f"e2_cooccurrence_{args.config}.json"
        )

    return args


# ===========================================================================
# Engine initialization helper (matching e1_verbatim_trace.py pattern)
# ===========================================================================
def init_engine(args, tokenizer, logger):
    """Initialize either a local InfiniGramEngine or an InfiniGramAPIEngine.

    Returns
    -------
    (engine, corpus_size) : tuple
        engine     — local InfiniGramEngine or InfiniGramAPIEngine
        corpus_size — total token count of the indexed corpus
    """
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

        # Get corpus size from engine
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (local): %d tokens", corpus_size)

        if corpus_size == 0:
            logger.warning(
                "Corpus size is 0! Falling back to hardcoded "
                "Pile-train size: 383,299,358,652"
            )
            corpus_size = 383_299_358_652

    else:
        # ---- API engine (OLMo 2 / olmo-mix-1124) ----
        logger.info("Using API engine: index=%s", args.api_index)
        engine = InfiniGramAPIEngine(index=args.api_index)

        # Get corpus size via empty-string count query
        logger.info("Querying corpus size from API (empty-string count)...")
        corpus_result = engine.count(input_ids=[])
        corpus_size = corpus_result.get("count", 0)
        logger.info("Corpus size (API, %s): %d tokens",
                     args.api_index, corpus_size)

        if corpus_size == 0:
            logger.error(
                "Failed to retrieve corpus size from API. "
                "Cannot compute concept rarity scores."
            )
            sys.exit(1)

    return engine, corpus_size


# ===========================================================================
# Concept loading & token-id conversion
# ===========================================================================
# Concepts are pre-extracted by e2_extract_concepts.py (LLM-based) and
# loaded here from a JSON file. We then convert each concept text into
# token IDs and locate its position within the response token sequence.

def load_concepts_from_extraction(concepts_path, logger):
    """Load LLM-extracted concepts from e2_concepts.json.

    Returns
    -------
    concepts_by_id : dict[int, dict]
        Map from record_id to a dict with the form
        {"concepts": [{"text": str, "rationale": str}, ...],
         "extraction_model": str,
         "extraction_notes": str}
    """
    if not os.path.isfile(concepts_path):
        logger.error("Concepts file not found: %s", concepts_path)
        logger.error("Run e2_extract_concepts.py first to produce this file.")
        sys.exit(1)

    with open(concepts_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    concepts_by_id = {}
    for row in rows:
        rec_id = row.get("id")
        if rec_id is None:
            continue
        concepts_by_id[rec_id] = {
            "concepts": row.get("concepts", []),
            "extraction_model": row.get("extraction_model", "unknown"),
            "extraction_notes": row.get("extraction_notes", ""),
        }

    logger.info("Loaded LLM-extracted concepts for %d records from %s",
                len(concepts_by_id), concepts_path)
    return concepts_by_id


def _find_subsequence(sequence, subsequence):
    """Find the first occurrence of subsequence in sequence.
    Returns index or -1."""
    sub_len = len(subsequence)
    if sub_len == 0:
        return -1
    for i in range(len(sequence) - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            return i
    return -1


def _concept_text_to_ids_and_position(concept_text, response_text, response_ids,
                                      tokenizer):
    """Convert a concept string into token IDs and locate its position.

    Strategy
    --------
    1. Tokenize ``concept_text`` directly.
    2. Search for the resulting token sequence in ``response_ids``
       (token-level subsequence match).
    3. If not found, fall back to a case-insensitive substring search in
       the decoded response text, then tokenize the matched substring
       using its leading whitespace context (so that BPE merges align).
    4. If still not found, return position = -1 but still return the
       token IDs (they may still produce non-zero counts in infini-gram).

    Returns
    -------
    (token_ids, position) : (list[int], int)
        token_ids may be empty if tokenization failed entirely.
        position is -1 if the concept was not located in response_ids.
    """
    # Step 1: direct tokenization
    token_ids = tokenizer.encode(concept_text, add_special_tokens=False)

    # Some tokenizers may still inject a BOS — strip if present
    if token_ids and tokenizer.bos_token_id is not None \
            and token_ids[0] == tokenizer.bos_token_id:
        token_ids = token_ids[1:]

    if not token_ids:
        return [], -1

    # Step 2: token-level subsequence match
    position = _find_subsequence(response_ids, token_ids)
    if position >= 0:
        return token_ids, position

    # Step 3: case-insensitive substring fallback
    lower_response = response_text.lower()
    lower_concept = concept_text.lower()
    char_pos = lower_response.find(lower_concept)
    if char_pos >= 0:
        # Re-tokenize using the original-case substring from response
        substring = response_text[char_pos:char_pos + len(concept_text)]
        # Try with a leading space to encourage BPE alignment
        for trial in (" " + substring, substring):
            trial_ids = tokenizer.encode(trial, add_special_tokens=False)
            if trial_ids and tokenizer.bos_token_id is not None \
                    and trial_ids[0] == tokenizer.bos_token_id:
                trial_ids = trial_ids[1:]
            if trial_ids:
                pos = _find_subsequence(response_ids, trial_ids)
                if pos >= 0:
                    return trial_ids, pos

    # Step 4: not found — return original token_ids with position = -1
    return token_ids, -1


def prepare_concepts_for_record(record_id, concepts_by_id, response_text,
                                 response_ids, tokenizer, logger):
    """Convert LLM-extracted concept strings into the dict format that
    compute_pairwise_cooccurrence expects.

    Each output dict has keys:
        ngram_ids   : list[int]   — token IDs for CNF query
        text        : str         — original concept text
        rationale   : str         — LLM rationale for extracting this concept
        position    : int         — token position in response (-1 if not found)
        length      : int         — number of tokens in the concept
    """
    entry = concepts_by_id.get(record_id)
    if entry is None:
        logger.warning("    No extracted concepts for record id=%s", record_id)
        return [], {"extraction_model": "unknown", "extraction_notes": ""}

    raw_concepts = entry["concepts"]
    extraction_meta = {
        "extraction_model": entry["extraction_model"],
        "extraction_notes": entry["extraction_notes"],
    }

    prepared = []
    n_unmatched = 0
    n_empty = 0
    for c in raw_concepts:
        text = (c.get("text") or "").strip()
        rationale = c.get("rationale", "")
        if not text:
            n_empty += 1
            continue

        token_ids, position = _concept_text_to_ids_and_position(
            text, response_text, response_ids, tokenizer
        )

        if not token_ids:
            n_empty += 1
            continue

        if position < 0:
            n_unmatched += 1

        prepared.append({
            "ngram_ids": token_ids,
            "text": text,
            "rationale": rationale,
            "position": position,
            "length": len(token_ids),
        })

    logger.info("    Loaded %d LLM concepts for record id=%s "
                "(unmatched position: %d, empty: %d)",
                len(prepared), record_id, n_unmatched, n_empty)
    return prepared, extraction_meta


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

    This counts positions where phrase_A AND phrase_B appear within w tokens.
    Works identically for local engine and API engine — both expose count_cnf() with the same interface.

    Parameters
    ----------
    max_pairs : int or None
        Maximum number of concept pairs to query. If None, query ALL
        C(n, 2) pairs (no limit). If an integer, take the first N pairs
        (note: with LLM-based extraction, concepts are NOT pre-sorted by
        rarity, so the first N pairs are simply the lexicographically
        first pairs).

    Returns:
      {
        "pairs": [
          {
            "concept_a_idx": int,                  # index into concepts list
            "concept_b_idx": int,                  # index into concepts list
            "concept_a": {"text": ..., "ngram_ids": ..., "position": ..., "rationale": ...},
            "concept_b": {"text": ..., "ngram_ids": ..., "position": ..., "rationale": ...},
            "counts_by_window": {100: {"count": N, "approx": bool}, ...}
          },
          ...
        ],
        "summary_by_window": {
          100: {"total_pairs": N, "nonzero_pairs": M, "max_count": K, ...},
          ...
        }
      }
    """
    # Generate all pairs; only truncate if max_pairs is set
    all_pairs = list(combinations(range(len(concepts)), 2))
    if max_pairs is not None and len(all_pairs) > max_pairs:
        logger.info("    Truncating pairs from %d to %d (--max_pairs)",
                    len(all_pairs), max_pairs)
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
            "concept_a_idx": i,
            "concept_b_idx": j,
            "concept_a": {
                "text": ca["text"],
                "ngram_ids": ca["ngram_ids"],
                "position": ca["position"],
                "rationale": ca.get("rationale", ""),
            },
            "concept_b": {
                "text": cb["text"],
                "ngram_ids": cb["ngram_ids"],
                "position": cb["position"],
                "rationale": cb.get("rationale", ""),
            },
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


def mark_all_pairs_zero(concepts, cooc_result, windows):
    """For each concept, mark whether ALL pairs containing it had zero
    co-occurrence in ALL windows. This flags suspicious concepts (likely
    not present in the corpus) without making any extra API calls.

    Mutates ``concepts`` in place by adding ``all_pairs_zero: bool``.
    """
    n = len(concepts)
    # Track for each concept index whether any of its pairs had a positive count
    has_any_nonzero = [False] * n

    for pair in cooc_result.get("pairs", []):
        i = pair.get("concept_a_idx")
        j = pair.get("concept_b_idx")
        if i is None or j is None:
            continue
        any_pos = False
        for w in windows:
            cnt = pair["counts_by_window"].get(w, {}).get("count", 0)
            if cnt > 0:
                any_pos = True
                break
        if any_pos:
            if 0 <= i < n:
                has_any_nonzero[i] = True
            if 0 <= j < n:
                has_any_nonzero[j] = True

    for idx, c in enumerate(concepts):
        c["all_pairs_zero"] = (not has_any_nonzero[idx])


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
      - E2_support_score: max_w log(1 + E2_cooc(w))
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
    args = parse_args()
    logger = setup_logger(args.model, args.config)

    logger.info("=== E2 Windowed Co-occurrence started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Arguments: %s", vars(args))
    start_time = time.time()

    # Determine backend mode for logging
    if args.index_dir is not None:
        logger.info("Backend: LOCAL engine (--index_dir=%s)", args.index_dir)
    else:
        logger.info("Backend: HTTP API (--api_index=%s)", args.api_index)

    # Load input data
    logger.info("Loading input from %s ...", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records total", len(records))

    # Filter records
    if args.compliant_only:
        target_records = [r for r in records if r.get("hb_label") == 1]
        logger.info("Filtered to %d compliant (hb_label=1) records",
                     len(target_records))
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

    # Initialize engine (local or API)
    engine, corpus_size = init_engine(args, tokenizer, logger)
    logger.info("CORPUS_SIZE = %d", corpus_size)

    # Warn about API max_diff_tokens limit
    is_api_mode = args.index_dir is None
    if is_api_mode:
        over_limit = [w for w in args.windows if w > 1000]
        if over_limit:
            logger.warning(
                "API max_diff_tokens limit is 1000. "
                "Windows %s will be clamped to 1000. "
                "Use local engine (--index_dir) for windows > 1000.",
                over_limit)

    logger.info("Windows: %s", args.windows)
    logger.info("Max pairs per record: %s",
                "ALL (no limit)" if args.max_pairs is None else args.max_pairs)
    logger.info("Concepts input: %s", args.concepts_input)

    # Load LLM-extracted concepts (produced by e2_extract_concepts.py)
    concepts_by_id = load_concepts_from_extraction(args.concepts_input, logger)

    # Resume: load existing results if output file exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    results = []
    completed_ids = set()
    if os.path.isfile(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                if "error" not in r.get("e2", {}):
                    completed_ids.add(r.get("id"))
            logger.info(
                "Resumed from %s: %d existing results (%d successful)",
                args.output, len(results), len(completed_ids))
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Could not load existing output (%s), starting fresh.", e)
            results = []
            completed_ids = set()

    # Process each record
    for rec_idx, record in enumerate(target_records):
        rec_id = record.get("id", rec_idx)
        response = record.get("response", "")

        # Skip if already successfully completed
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
            # Tokenize response
            response_ids = tokenizer.encode(response)
            L = len(response_ids)
            logger.info("  Tokenized response: %d tokens", L)

            # Step 1: Load LLM-extracted concepts and convert to token IDs
            logger.info("  Step 1: Loading LLM-extracted concepts...")
            concepts, extraction_meta = prepare_concepts_for_record(
                rec_id, concepts_by_id,
                response_text=response,
                response_ids=response_ids,
                tokenizer=tokenizer,
                logger=logger,
            )

            if len(concepts) < 2:
                logger.warning(
                    "  Only %d concepts available, skipping co-occurrence",
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
                # Still mark all_pairs_zero (vacuously true for any concept)
                for c in concepts:
                    c["all_pairs_zero"] = True
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

                # Mark concepts whose pairs are all zero (no extra API calls)
                mark_all_pairs_zero(concepts, cooc_result, args.windows)

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
                    "extraction_method": "llm",
                    "extraction_model": extraction_meta.get(
                        "extraction_model", "unknown"),
                    "extraction_notes": extraction_meta.get(
                        "extraction_notes", ""),
                    "enabling_concepts": concepts,
                    "pairwise_cooccurrence": cooc_result,
                },
            }

            rec_elapsed = time.time() - rec_start

            # Log summary per window
            for w in args.windows:
                wm = e2_metrics.get("metrics_by_window", {}).get(w, {})
                logger.info(
                    "  Window w=%d: E2_cooc=%s, nonzero=%s/%s, mean=%s",
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

        # Incremental save after each record
        logger.info("  Saving %d results to %s ...",
                     len(results), args.output)
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
            scores = [
                r["e2"]["metrics_by_window"].get(w, {}).get("E2_cooc", 0)
                for r in successful
                if w in r["e2"].get("metrics_by_window", {})
            ]
            if scores:
                logger.info(
                    "  Window w=%d: E2_cooc max=%d, mean=%.1f, "
                    "nonzero=%d/%d records",
                    w, max(scores), sum(scores) / len(scores),
                    sum(1 for s in scores if s > 0), len(scores))

        support_scores = [r["e2"].get("E2_support_score", 0)
                          for r in successful]
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
    logger.info(
        "=== Elapsed time: %d days %02d:%02d:%02d "
        "(total %.3f seconds) ===",
        days, hours, minutes, seconds, elapsed_float)


if __name__ == "__main__":
    main()