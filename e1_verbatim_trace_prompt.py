#!/usr/bin/env python3
"""E1 Verbatim Trace (Prompt-Level): maximal matching spans between each
HarmBench prompt x and a reference corpus, parallel to ``e1_verbatim_trace.py``
which operates on model responses y.

HarmBench prompts are model-independent (identical across all OLMo 2 variants),
so this script reads from a single reference model directory. For pretraining
and mid-training the corpus is shared across the OLMo 2 family, so one output
is produced. For post-training the corpus is per-size (1b/7b/13b/32b); the
size is auto-detected from ``--index_dir`` and encoded in the output path
(e.g., ``./index/post-training/7b`` -> ``results/prompt/post_training/7b/``).
ALL prompts are processed (no compliance/refusal filter, no degeneracy filter
— degeneracy is a response-level concept).

Backends mirror the response-level script:
  - ``--api_index v4_olmo-mix-1124_llama``  -> corpus label ``pretraining``
  - ``--index_dir ./index/dolmino-mix-1124`` -> corpus label ``mid_training``
  - ``--index_dir ./index/post-training/{size}`` -> corpus label ``post_training/{size}``

Output: ``results/prompt/{corpus}/e1_verbatim_{config}.json``
Log:    ``logs/prompt/{corpus}/e1_verbatim_trace_prompt_{config}.log``

Snippet schema (Phase 2, prompt-level):
    Unlike ``e1_verbatim_trace.retrieve_snippets_for_span`` (response-level,
    pre-only by design choice), this script splits each retrieved doc into
    pre/match/post segments using the API's ``spans = [(text, label_or_None)]``
    structure (``label='0'`` for matched needle, ``None`` for context). Per
    snippet: ``pre_context_text``, ``match_text``, ``post_context_text``,
    ``snippet_text`` (= pre + match + post), plus ``blocked`` flag.

Usage:
    # Pretraining corpus, standard config, single-record sanity check
    python e1_verbatim_trace_prompt.py \
        --config standard \
        --api_index v4_olmo-mix-1124_llama \
        --test --record_id 30 --retrieve_snippets

    # Pretraining corpus, full run with snippets
    python e1_verbatim_trace_prompt.py \
        --config standard \
        --api_index v4_olmo-mix-1124_llama \
        --retrieve_snippets

    # Mid-training corpus (local dolmino index)
    python e1_verbatim_trace_prompt.py \
        --config standard \
        --index_dir ./index/dolmino-mix-1124 \
        --retrieve_snippets

    # Post-training corpus (per-model-size local index; size auto-detected
    # from the path, output goes to results/prompt/post_training/7b/)
    python e1_verbatim_trace_prompt.py \
        --config standard \
        --index_dir ./index/post-training/7b \
        --retrieve_snippets
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

# Path setup (mirrors e1_verbatim_trace.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer

from infini_gram_api import InfiniGramAPIEngine

from e1_verbatim_trace import (
    compute_e1_metrics,
    compute_maximal_matching_spans,
    filter_top_k_spans,
    init_engine,
)


# Reference model directory for prompt input (prompts are identical across
# OLMo 2 variants; pick one canonical source).
REFERENCE_MODEL_DIR = "olmo2_7b"


# ===========================================================================
# Snippet retrieval — prompt-level variant (full pre/match/post context).
#
# The infini-gram HTTP API returns ``spans`` as a list of ``(text, label)``
# tuples where ``label`` is ``None`` for context segments and ``"0"`` for the
# matched needle. Typical shape (when needle is mid-doc):
#
#     spans = [(pre_text, None), (match_text, "0"), (post_text, None)]
#
# Edge cases: needle at doc start -> ``[(match, "0"), (post, None)]``;
# needle at doc end -> ``[(pre, None), (match, "0")]``. Multi-occurrences in
# one doc produce additional ``(text, None)`` segments interleaved with
# ``(text, "0")``; we keep the FIRST match and treat all label=None segments
# before/after it as pre/post respectively.
#
# Differs from ``e1_verbatim_trace.retrieve_snippets_for_span`` (response-level,
# pre-only by design choice) — prompt-level keeps everything since this is
# new code with no downstream backward-compat constraint.
# ===========================================================================
def _join_with_boundary_space(pre: str, match: str, post: str) -> str:
    """Join ``pre + match + post`` with a single space at any non-empty
    boundary, defensively avoiding double spaces.

    The infini-gram API tokenizer can split mid-word (e.g. "the199Os"),
    so naive concatenation produces unreadable boundaries. This helper:
      - skips empty parts entirely (no leading/trailing space artifact)
      - uses ``rstrip()`` on ``pre`` and ``lstrip()`` on ``post`` so that
        already-whitespace-padded boundaries don't double up
      - leaves ``match`` untouched (it's the decoded query span, kept raw)

    Raw segments remain available in ``pre_context_text`` / ``match_text``
    / ``post_context_text`` for analyses that need exact boundaries.
    """
    parts = []
    if pre:
        parts.append(pre.rstrip())
    if match:
        parts.append(match)
    if post:
        parts.append(post.lstrip())
    return " ".join(parts)


def retrieve_snippets_for_span_full_context(engine, span_ids, max_docs,
                                             max_disp_len, tokenizer):
    """Retrieve up to ``max_docs`` snippets, splitting each into
    pre/match/post segments based on the API's spans labels.

    Per-snippet schema:
        - doc_ix, doc_len, metadata          (unchanged from response-level)
        - blocked: bool                      (defensive marker; default False)
        - pre_context_text:  str             (label=None segment(s) before match)
        - match_text:        str             (label='0' segment, == decoded span_ids)
        - post_context_text: str             (label=None segment(s) after match,
                                              "" if needle is at doc end)
        - snippet_text:      str             (pre/match/post joined with a
                                              defensive single space at each
                                              non-empty boundary; see
                                              ``_join_with_boundary_space``)
        - snippet_token_ids: list[int]       (raw token_ids returned by API)

    None-safety:
        - Per-segment ``text`` of ``None`` (rare; should not occur with the
          server spec) is replaced with ``""`` and a warning is logged.
        - ``blocked=True`` docs become marker entries (all *_text fields ""),
          never silently dropped.
    """
    _log = logging.getLogger(__name__)
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
                    doc = engine.get_doc_by_rank(
                        s=s, rank=rank, max_disp_len=max_disp_len,
                        query_ids=span_ids,
                    )
                else:
                    doc = engine.get_doc_by_rank(
                        s=s, rank=rank, max_disp_len=max_disp_len,
                    )

                if "error" in doc:
                    continue

                is_blocked = bool(doc.get("blocked", False))

                snippet_info = {
                    "doc_ix": doc.get("doc_ix"),
                    "doc_len": doc.get("doc_len"),
                    "metadata": doc.get("metadata", ""),
                    "blocked": is_blocked,
                }

                if is_blocked:
                    snippet_info.update({
                        "pre_context_text": "",
                        "match_text": "",
                        "post_context_text": "",
                        "snippet_text": "",
                        "snippet_token_ids": [],
                    })
                    snippets.append(snippet_info)
                    docs_retrieved += 1
                    continue

                if isinstance(engine, InfiniGramAPIEngine) and "spans" in doc:
                    # API mode: parse the (text, label) segment list.
                    spans_data = doc.get("spans") or []
                    pre_text = ""
                    match_text = ""
                    post_text = ""
                    none_text_count = 0
                    seen_match = False
                    for seg in spans_data:
                        if not seg or len(seg) < 2:
                            continue
                        text, label = seg[0], seg[1]
                        if text is None:
                            none_text_count += 1
                            text = ""
                        if label == "0" and not seen_match:
                            match_text = text
                            seen_match = True
                        elif seen_match:
                            post_text += text
                        else:
                            pre_text += text
                    if none_text_count > 0:
                        _log.warning(
                            "doc_ix=%s has %d segments with None text "
                            "(non-blocked); reconstruction is partial",
                            doc.get("doc_ix"), none_text_count,
                        )
                    snippet_info["pre_context_text"] = pre_text
                    snippet_info["match_text"] = match_text
                    snippet_info["post_context_text"] = post_text
                    snippet_info["snippet_text"] = _join_with_boundary_space(
                        pre_text, match_text, post_text)
                    snippet_info["snippet_token_ids"] = doc.get("token_ids", [])

                elif "token_ids" in doc and tokenizer is not None:
                    # Local engine: no spans field; split via needle_offset.
                    tids = doc.get("token_ids", []) or []
                    needle_offset = doc.get("needle_offset")
                    needle_len = len(span_ids)
                    pre_text = match_text = post_text = ""
                    if (needle_offset is not None
                            and isinstance(needle_offset, int)
                            and needle_offset >= 0
                            and needle_offset + needle_len <= len(tids)):
                        try:
                            pre_ids = tids[:needle_offset]
                            match_ids = tids[needle_offset:needle_offset + needle_len]
                            post_ids = tids[needle_offset + needle_len:]
                            if pre_ids:
                                pre_text = tokenizer.decode(
                                    pre_ids, skip_special_tokens=False)
                            if match_ids:
                                match_text = tokenizer.decode(
                                    match_ids, skip_special_tokens=False)
                            if post_ids:
                                post_text = tokenizer.decode(
                                    post_ids, skip_special_tokens=False)
                        except Exception as exc:
                            _log.warning(
                                "Local-mode decode failed for doc_ix=%s: %s",
                                doc.get("doc_ix"), exc,
                            )
                            try:
                                match_text = tokenizer.decode(
                                    tids, skip_special_tokens=False)
                            except Exception:
                                match_text = ""
                    else:
                        # No needle offset info — best effort: decode full window.
                        try:
                            match_text = tokenizer.decode(
                                tids, skip_special_tokens=False)
                        except Exception:
                            match_text = ""
                    snippet_info["pre_context_text"] = pre_text
                    snippet_info["match_text"] = match_text
                    snippet_info["post_context_text"] = post_text
                    snippet_info["snippet_text"] = _join_with_boundary_space(
                        pre_text, match_text, post_text)
                    snippet_info["snippet_token_ids"] = tids

                else:
                    snippet_info.update({
                        "pre_context_text": "",
                        "match_text": "",
                        "post_context_text": "",
                        "snippet_text": "",
                        "snippet_token_ids": doc.get("token_ids", []),
                    })

                snippets.append(snippet_info)
                docs_retrieved += 1
            except Exception as exc:
                _log.warning(
                    "Unexpected error retrieving doc (s=%s, rank=%s): %s: %s",
                    s, rank, type(exc).__name__, exc,
                )
                continue

    return snippets


# ===========================================================================
# Corpus label derivation
# ===========================================================================
def derive_corpus_label(args) -> str:
    """Map (--api_index | --index_dir) -> corpus label used in the output path.

      - ``--api_index v4_olmo-mix-1124_llama`` -> ``pretraining``
      - ``--index_dir ./index/dolmino-mix-1124`` -> ``mid_training``
      - ``--index_dir ./index/post-training/{size}`` -> ``post_training/{size}``

    Falls back to a generic label derived from the input if neither marker
    is present (so unusual indices still produce a deterministic path).
    """
    if args.index_dir:
        norm = os.path.normpath(args.index_dir).replace("\\", "/").lower()
        base = os.path.basename(os.path.normpath(args.index_dir)).lower()
        if "dolmino-mix-1124" in norm or base == "dolmino-mix-1124":
            return "mid_training"
        if "post-training" in norm or "post_training" in norm:
            # Post-training corpora are per-size (./index/post-training/{1b,7b,13b,32b}/),
            # so the segment after "post-training" is appended to keep results separated.
            parts = norm.split("/")
            for i, p in enumerate(parts):
                if p in ("post-training", "post_training") and i + 1 < len(parts):
                    size = parts[i + 1]
                    if size:
                        return f"post_training/{size}"
            return "post_training"
        return base or "local_index"

    api_index = (args.api_index or "").lower()
    if "olmo-mix-1124_llama" in api_index:
        return "pretraining"
    if any(m in api_index for m in ("post-training", "post_training", "posttrain")):
        return "post_training"
    return api_index.replace("/", "_") or "api_index"


# ===========================================================================
# Logger setup (mirrors utils.setup_logger pattern, but rooted under
# logs/prompt/{corpus}/ since prompt-level is model-independent).
# ===========================================================================
def setup_logger(corpus: str, config: str):
    log_dir = os.path.join("logs", "prompt", corpus)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(
        log_dir, f"e1_verbatim_trace_prompt_{config}.log"
    )

    logger_name = f"e1_verbatim_trace_prompt.{corpus}.{config}"
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


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 Verbatim Trace (prompt-level): maximal matching spans "
                    "between each HarmBench prompt and a single specified corpus."
    )

    parser.add_argument("--config", type=str, default="standard",
                        choices=["standard", "contextual", "copyright"],
                        help="HarmBench config (default: standard)")

    # Backend selection: --api_index XOR --index_dir (same shape as e1_verbatim_trace.py).
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--api_index", type=str, default=None,
                         help="API index name (e.g., v4_olmo-mix-1124_llama). "
                              "Mutually exclusive with --index_dir.")
    backend.add_argument("--index_dir", type=str, default=None,
                         help="Path to local infini-gram index directory "
                              "(e.g., ./index/dolmino-mix-1124). "
                              "Mutually exclusive with --api_index.")

    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index "
                             "(default: meta-llama/Llama-2-7b-hf, same as the "
                             "response-level pipeline).")

    parser.add_argument("--top_k_ratio", type=float, default=0.05,
                        help="Fraction of prompt tokens to keep as top-K spans "
                             "(OLMoTrace default: 0.05).")

    # Phase 2
    parser.add_argument("--retrieve_snippets", action="store_true",
                        help="Phase 2: retrieve corpus snippets for top-K spans.")
    parser.add_argument("--max_docs_per_span", type=int, default=10,
                        help="Max documents to retrieve per span (default: 10).")
    parser.add_argument("--max_disp_len", type=int, default=80,
                        help="Max tokens to display per snippet (default: 80).")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run on a single record only (use with --record_id).")
    parser.add_argument("--record_id", type=int, default=None,
                        help="Record id to process when --test is set.")

    args = parser.parse_args()

    # Default backend: API w/ olmo-mix-1124_llama (matches the response-level default).
    if args.api_index is None and args.index_dir is None:
        args.api_index = "v4_olmo-mix-1124_llama"

    if args.test and args.record_id is None:
        parser.error("--test requires --record_id")

    return args


# ===========================================================================
# Main
# ===========================================================================
def main():
    args = parse_args()

    corpus = derive_corpus_label(args)
    logger = setup_logger(corpus, args.config)

    logger.info("=== E1 Verbatim Trace (prompt-level) started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Corpus label: %s", corpus)
    logger.info("Arguments: %s", vars(args))

    # Resolve I/O paths.
    input_path = os.path.join(
        "data", REFERENCE_MODEL_DIR, f"harmbench_{args.config}_labeled.json"
    )
    output_path = os.path.join(
        "results", "prompt", corpus, f"e1_verbatim_{args.config}.json"
    )
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)

    if not os.path.isfile(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Tokenizer (Llama-2, same as response-level pipeline).
    logger.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    # Engine (init_engine reads args.index_dir / args.api_index).
    engine, num_shards, corpus_size = init_engine(args, tokenizer, logger)

    if args.index_dir is not None:
        logger.info("Backend: LOCAL engine (--index_dir=%s)", args.index_dir)
    else:
        logger.info("Backend: HTTP API (--api_index=%s)", args.api_index)
    logger.info("CORPUS_SIZE = %d", corpus_size)

    # Load HarmBench prompts.
    logger.info("Loading prompts from %s ...", input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    logger.info("Loaded %d records total", len(records))

    if args.test:
        target_records = [r for r in records if r.get("id") == args.record_id]
        if not target_records:
            logger.error("--test --record_id=%d: no record with that id in %s",
                         args.record_id, input_path)
            sys.exit(1)
        logger.info("--test mode: processing only record id=%s",
                    args.record_id)
    else:
        target_records = records
        logger.info("Processing ALL %d records (no compliance/refusal filter)",
                    len(target_records))

    start_time = time.time()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resume: load existing results if output file exists. Skip records that
    # already have a successful e1 entry. Test mode does not resume — it
    # overwrites only the targeted record.
    results = []
    completed_ids = set()
    if not args.test and os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                if "error" not in r.get("e1", {}):
                    completed_ids.add(r.get("record_id"))
            logger.info("Resumed from %s: %d existing results (%d successful)",
                        output_path, len(results), len(completed_ids))
        except Exception as exc:
            logger.warning("Could not load existing output (%s); starting fresh.", exc)
            results = []
            completed_ids = set()

    for rec_idx, record in enumerate(target_records):
        rec_id = record.get("id", rec_idx)
        prompt_text = record.get("prompt", "") or ""

        if rec_id in completed_ids:
            logger.info("[%d/%d] Skipping record_id=%s (already completed)",
                        rec_idx + 1, len(target_records), rec_id)
            continue

        logger.info("=" * 70)
        logger.info("[%d/%d] Processing record_id=%s",
                    rec_idx + 1, len(target_records), rec_id)
        logger.info("  Prompt: %s",
                    prompt_text[:120] + ("..." if len(prompt_text) > 120 else ""))

        rec_start = time.time()

        try:
            prompt_ids = tokenizer.encode(prompt_text)
            L = len(prompt_ids)
            logger.info("  Tokenized prompt: %d tokens", L)

            maximal_spans = compute_maximal_matching_spans(
                engine, prompt_ids, num_shards, logger
            )

            top_k_spans = filter_top_k_spans(
                engine, prompt_ids, maximal_spans, args.top_k_ratio,
                corpus_size, logger
            )

            e1_metrics = compute_e1_metrics(
                prompt_ids, maximal_spans, top_k_spans, tokenizer,
                token_len_field="prompt_token_len",
            )

            if args.retrieve_snippets:
                logger.info("  Phase 2: retrieving snippets for %d top-K spans...",
                            len(top_k_spans))
                snippets_by_span = []
                for span_idx, (b, e) in enumerate(top_k_spans):
                    snippets = retrieve_snippets_for_span_full_context(
                        engine, prompt_ids[b:e],
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

            result = {
                "record_id": rec_id,
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt_ids,
                "metadata": record.get("metadata", {}),
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
            logger.error("  Error processing record_id=%s: %s", rec_id, exc,
                         exc_info=True)
            result = {
                "record_id": rec_id,
                "prompt_text": prompt_text,
                "prompt_token_ids": [],
                "metadata": record.get("metadata", {}),
                "e1": {"error": f"{type(exc).__name__}: {exc}"},
            }

        results.append(result)

        logger.info("  Saving %d results to %s ...", len(results), output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    successful = [r for r in results if "error" not in r.get("e1", {})]
    logger.info("  Total in output: %d", len(results))
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

    # Test-mode visual inspection.
    if args.test and successful:
        r = next((rr for rr in successful if rr.get("record_id") == args.record_id), None)
        if r:
            print("\n" + "=" * 70)
            print(f"TEST MODE — record_id={r['record_id']}")
            print("=" * 70)
            print(f"prompt_text: {r['prompt_text']}")
            e = r["e1"]
            print(f"prompt_token_len:  {e.get('prompt_token_len')}")
            print(f"LongestMatchLen:   {e['LongestMatchLen']}")
            print(f"VerbatimCoverage:  {e['VerbatimCoverage']}")
            print(f"num_maximal_spans: {e['num_maximal_spans']}")
            print(f"num_top_k_spans:   {e['num_top_k_spans']}")
            print("top_k_spans:")
            for s in e["top_k_spans"]:
                print(f"  [{s['begin']}..{s['end']}] (len={s['length']}) {s['text']!r}")
            if "ExampleSnippets" in e and e["ExampleSnippets"]:
                first_with_snip = next(
                    (g for g in e["ExampleSnippets"] if g["snippets"]), None
                )
                if first_with_snip:
                    s0 = first_with_snip["snippets"][0]
                    print("\nTop snippet (from first top-K span with results):")
                    print(f"  span_text:         {first_with_snip['span_text']!r}")
                    print(f"  doc_ix:            {s0.get('doc_ix')}")
                    print(f"  doc_len:           {s0.get('doc_len')}")
                    print(f"  blocked:           {s0.get('blocked')}")

                    def _preview(text: str, n: int = 200) -> str:
                        text = text or ""
                        return text[:n] + ("..." if len(text) > n else "")

                    print(f"  pre_context_text:  {_preview(s0.get('pre_context_text', ''))!r}")
                    print(f"  match_text:        {s0.get('match_text', '')!r}")
                    print(f"  post_context_text: {_preview(s0.get('post_context_text', ''))!r}")
                    print(f"  snippet_text:      {_preview(s0.get('snippet_text', ''), 400)!r}")
                    print(f"  metadata:          {_preview(str(s0.get('metadata', '')), 200)!r}")
                else:
                    print("\nNo snippets returned for any top-K span.")

    elapsed_float = time.time() - start_time
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
