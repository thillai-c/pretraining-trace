#!/usr/bin/env python3
"""E1 Phase 2: Retrieve corpus snippets for existing Phase 1 results.

Reads Phase 1 output JSON (with top_k_spans), retrieves snippets from
infini-gram API/local engine, and adds ExampleSnippets field.
Skips records that already have ExampleSnippets.

Usage:
    # OLMo 2 (API backend; default index -> results/{model_dir}/pretraining/...)
    python e1_retrieve_snippets.py --model olmo2-1b

    # Multiple HarmBench configs in one run (same rules as e1_verbatim_trace.py for paths)
    python e1_retrieve_snippets.py --model olmo2-7b --configs standard contextual

    # GPT-J (local Pile index -> flat results/{model_dir}/..., same as Phase 1)
    python e1_retrieve_snippets.py --model gpt-j --index_dir ./index

    # Local dolmino -> results/{model_dir}/mid_training/...
    python e1_retrieve_snippets.py --model olmo2-7b --index_dir ./index/dolmino-mix-1124

    # Custom parameters
    python e1_retrieve_snippets.py --model olmo2-1b --max_docs_per_span 10 --max_disp_len 80
"""

import argparse
import json
import logging
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

from e1_verbatim_trace import (
    MODEL_CONFIGS,
    e1_results_subdir,
    init_engine,
    retrieve_snippets_for_span,
)


def setup_logger(model_key: str, config: str = "standard"):
    """Log file: logs/{model_dir}/e1_retrieve_snippets_{config}.log (aligned with e1_verbatim_trace)."""
    out_dir = MODEL_CONFIGS[model_key]["out_dir_name"]
    log_dir = os.path.join("logs", out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"e1_retrieve_snippets_{config}.log")

    logger_name = f"e1_retrieve_snippets.{model_key}.{config}"
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


def _resolve_configs(args) -> list:
    if args.configs:
        return list(args.configs)
    return [args.config]


def _resolve_input_path(args, model_dir: str, config: str, multi: bool) -> str:
    """Default input path matches Phase 1 output; explicit --input follows e1_verbatim_trace rules."""
    if args.input is None:
        sub = e1_results_subdir(args)
        if sub:
            return os.path.join(
                "results", model_dir, sub, f"e1_verbatim_{config}.json"
            )
        return os.path.join("results", model_dir, f"e1_verbatim_{config}.json")
    if "{config}" in args.input or "{model_dir}" in args.input:
        return args.input.format(model_dir=model_dir, config=config)
    if multi:
        base, ext = os.path.splitext(args.input)
        return f"{base}_{config}{ext}"
    return args.input


def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 Phase 2: Retrieve snippets for existing Phase 1 results"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model key (determines auto paths)")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench config name (default: standard). "
                             "For multiple configs in one run, use --configs.")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Run multiple HarmBench configs in one invocation. "
             "Example: --configs standard contextual copyright",
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Phase 1 results JSON. If omitted, default matches "
                             "e1_verbatim_trace output (subdir rules: API "
                             "olmo-mix-1124_llama -> pretraining/; dolmino -> "
                             "mid_training/; post-training markers -> "
                             "post_training/; else flat). "
                             "Optional template: {model_dir}, {config}. "
                             "With --configs and a single path without "
                             "placeholders, uses base_{config}.json.")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="Path to local infini-gram index directory. "
                             "If omitted, use HTTP API.")
    parser.add_argument("--api_index", type=str,
                        default="v4_olmo-mix-1124_llama",
                        help="API index name (default: v4_olmo-mix-1124_llama)")
    parser.add_argument("--tokenizer_name", type=str,
                        default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer matching the infini-gram index")
    parser.add_argument("--max_docs_per_span", type=int, default=10,
                        help="Max documents to retrieve per span (default: 10)")
    parser.add_argument("--max_disp_len", type=int, default=80,
                        help="Max tokens per document snippet (default: 80)")

    return parser.parse_args()


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

    logger0.info("Loading tokenizer: %s", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    engine, num_shards, corpus_size = init_engine(args, tokenizer, logger0)

    for config in configs:
        input_path = _resolve_input_path(args, model_dir, config, multi)
        logger = setup_logger(args.model, config)

        logger.info("=== E1 Phase 2: Snippet Retrieval started at %s ===",
                    datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
        logger.info("Config: %s", config)
        logger.info(
            "Arguments: %s",
            {**vars(args), "config": config, "input": input_path},
        )
        start_time = time.time()

        logger.info("Loading Phase 1 results from %s ...", input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info("Loaded %d records", len(results))

        for rec_idx, rec in enumerate(results):
            rec_id = rec.get("id", rec_idx)
            e1 = rec.get("e1", {})

            if "error" in e1:
                logger.info("[%d/%d] Skipping record id=%s (has error)",
                            rec_idx + 1, len(results), rec_id)
                continue

            top_k_spans = e1.get("top_k_spans", [])
            if not top_k_spans:
                logger.info("[%d/%d] Skipping record id=%s (no top_k_spans)",
                            rec_idx + 1, len(results), rec_id)
                continue

            existing_snippets = e1.get("ExampleSnippets", [])
            filled = [s for s in existing_snippets if len(s.get("snippets", [])) > 0]
            if len(filled) >= len(top_k_spans):
                logger.info("[%d/%d] Skipping record id=%s (all %d spans done)",
                            rec_idx + 1, len(results), rec_id, len(top_k_spans))
                continue

            response_ids = tokenizer.encode(rec.get("response", ""))

            start_span_idx = len(filled)

            logger.info("=" * 70)
            logger.info("[%d/%d] Retrieving snippets for record id=%s "
                        "(%d spans, resuming from %d)",
                        rec_idx + 1, len(results), rec_id,
                        len(top_k_spans), start_span_idx)

            rec_start = time.time()
            snippets_by_span = list(filled)

            RATE_LIMIT_COOLDOWN = 300
            MAX_SPAN_RETRIES = 5

            abort = False
            for span_idx, span in enumerate(top_k_spans):
                if span_idx < start_span_idx:
                    continue

                b = span["begin"]
                e_pos = span["end"]
                span_ids = response_ids[b:e_pos]

                for span_attempt in range(1, MAX_SPAN_RETRIES + 1):
                    try:
                        snippets = retrieve_snippets_for_span(
                            engine, span_ids,
                            max_docs=args.max_docs_per_span,
                            max_disp_len=args.max_disp_len,
                            tokenizer=tokenizer,
                        )
                        break
                    except requests.exceptions.HTTPError as exc:
                        if exc.response is not None and exc.response.status_code in (403, 429):
                            if span_attempt < MAX_SPAN_RETRIES:
                                logger.warning(
                                    "  Rate limit at span %d / %d (attempt %d/%d). "
                                    "Cooling down %ds ...",
                                    span_idx + 1, len(top_k_spans),
                                    span_attempt, MAX_SPAN_RETRIES,
                                    RATE_LIMIT_COOLDOWN)
                                time.sleep(RATE_LIMIT_COOLDOWN)
                                continue
                            else:
                                logger.error(
                                    "  Rate limit at span %d / %d exhausted %d retries. "
                                    "Saving partial results and stopping.",
                                    span_idx + 1, len(top_k_spans), MAX_SPAN_RETRIES)
                                e1["ExampleSnippets"] = snippets_by_span
                                e1["snippet_error"] = f"{type(exc).__name__}: {exc}"
                                abort = True
                                break
                        else:
                            logger.error("  Non-retryable error at span %d / %d: %s",
                                         span_idx + 1, len(top_k_spans), exc)
                            e1["ExampleSnippets"] = snippets_by_span
                            e1["snippet_error"] = f"{type(exc).__name__}: {exc}"
                            abort = True
                            break
                    except Exception as exc:
                        logger.error("  Unexpected error at span %d / %d: %s",
                                     span_idx + 1, len(top_k_spans), exc)
                        e1["ExampleSnippets"] = snippets_by_span
                        e1["snippet_error"] = f"{type(exc).__name__}: {exc}"
                        abort = True
                        break

                if abort:
                    logger.info("  Saving partial results (%d / %d spans) to %s ...",
                                len(snippets_by_span), len(top_k_spans), input_path)
                    with open(input_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.error("Exiting due to unrecoverable error. "
                                 "Re-run to resume from span %d.",
                                 len(snippets_by_span))
                    sys.exit(1)

                snippets_by_span.append({
                    "span_begin": b,
                    "span_end": e_pos,
                    "span_length": e_pos - b,
                    "span_text": span.get("text", ""),
                    "num_snippets": len(snippets),
                    "snippets": snippets,
                })

                if (span_idx + 1) % 10 == 0:
                    logger.info("  Span %d / %d done",
                                span_idx + 1, len(top_k_spans))

            e1["ExampleSnippets"] = snippets_by_span
            e1.pop("snippet_error", None)

            rec_elapsed = time.time() - rec_start
            total_snippets = sum(s.get("num_snippets", 0) for s in snippets_by_span)
            logger.info("  Done: %d / %d spans, %d snippets, %.1fs",
                        len(snippets_by_span), len(top_k_spans),
                        total_snippets, rec_elapsed)

            logger.info("  Saving to %s ...", input_path)
            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("=" * 70)
        logger.info("SUMMARY (config=%s)", config)
        logger.info("=" * 70)
        with_snippets = sum(
            1 for r in results if "ExampleSnippets" in r.get("e1", {})
        )
        logger.info("  Records with snippets: %d / %d", with_snippets, len(results))

        elapsed_float = time.time() - start_time
        elapsed = int(elapsed_float)
        days = elapsed // 86400
        hours = (elapsed % 86400) // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        logger.info("=== Finished at %s ===",
                    datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
        logger.info(
            "=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===",
            days, hours, minutes, seconds, elapsed_float,
        )


if __name__ == "__main__":
    main()