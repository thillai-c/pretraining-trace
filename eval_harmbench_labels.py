#!/usr/bin/env python3
"""Evaluate HarmBench compliance labels for copyright, standard, and contextual categories.

For copyright behaviors (hash_check tag), uses hash-based evaluation.
For standard/contextual behaviors, uses classifier-based evaluation (requires --cls_path).

Usage:
    # Auto-generated paths (recommended)
    python eval_harmbench_labels.py --model olmo2-7b --cls_path cais/HarmBench-Llama-2-13b-cls

    # All base models
    for m in olmo2-1b olmo2-7b olmo2-13b olmo2-32b; do
        python eval_harmbench_labels.py --model $m --cls_path cais/HarmBench-Llama-2-13b-cls
    done

    # Manual paths
    python eval_harmbench_labels.py \
        --model gpt-j \
        --data_dir data/gpt_j_6b/harmbench_standard.json \
        --output_dir data/gpt_j_6b/harmbench_standard_labeled.json \
        --cls_path cais/HarmBench-Llama-2-13b-cls
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import HarmBench eval utils
sys.path.insert(0, str(Path(__file__).parent / "HarmBench"))
from eval_utils import compute_results_hashing, compute_results_classifier

from utils import MODEL_CONFIGS


def setup_logger(model_key, config):
    try:
        out_dir = MODEL_CONFIGS[model_key]["out_dir_name"]
    except KeyError:
        # Fallback: keep logs organized even for unknown model keys.
        out_dir = model_key.replace("-", "_")
    log_dir = os.path.join("logs", out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f"eval_harmbench_{config}.log")

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
    parser = argparse.ArgumentParser(description="Evaluate HarmBench compliance labels")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key for log directory (e.g., 'olmo2-7b', 'gpt-j')")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Input JSON file. Auto-generated if not specified: "
                             "data/{model}/harmbench_{config}.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output JSON file. Auto-generated if not specified: "
                             "data/{model}/harmbench_{config}_labeled.json")
    parser.add_argument("--config", type=str, default="standard",
                        help="HarmBench category (e.g., 'standard', 'copyright', 'contextual')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--cls_path", type=str, default=None,
                        help="Path to classifier model for standard/contextual evaluation "
                             "(e.g., 'cais/HarmBench-Llama-2-13b-cls'). Required for non-copyright categories.")
    parser.add_argument("--num_tokens", type=int, default=512,
                        help="Maximum number of tokens to evaluate (for classifier)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.model, args.config)

    # Auto-generate input/output paths if not specified
    # data_dir uses underscore (olmo2_7b), model key uses hyphen (olmo2-7b)
    model_dir = args.model.replace("-", "_")
    if args.data_dir is None:
        args.data_dir = os.path.join("data", model_dir, f"harmbench_{args.config}.json")
    if args.output_dir is None:
        args.output_dir = os.path.join("data", model_dir, f"harmbench_{args.config}_labeled.json")

    logger.info("=== Job started at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("Model key  : %s", args.model)
    logger.info("Input      : %s", args.data_dir)
    logger.info("Output     : %s", args.output_dir)

    start_time = time.time()

    # Change to HarmBench directory so compute_results_hashing can find data files via relative path
    harmbench_dir = Path(__file__).parent / "HarmBench"
    original_cwd = os.getcwd()
    os.chdir(harmbench_dir)

    # Initialize classifier and tokenizer once (for standard/contextual categories)
    cls = None
    cls_tokenizer = None
    cls_params = None

    if args.cls_path:
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer

            logger.info("Loading classifier tokenizer from %s ...", args.cls_path)
            cls_tokenizer = AutoTokenizer.from_pretrained(
                args.cls_path,
                use_fast=False,
                truncation_side="left",
                padding_side="left",
            )

            logger.info("Loading classifier model from %s ...", args.cls_path)
            # enforce_eager=True: disables CUDAGraph compilation in vLLM V1 engine,
            # which avoids "Cannot re-initialize CUDA in forked subprocess" error.
            cls = LLM(model=args.cls_path, tensor_parallel_size=1, enforce_eager=True)
            # cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
            cls_params = SamplingParams(temperature=0.0, max_tokens=1)
            logger.info("Classifier loaded successfully")

        except ImportError:
            logger.warning("vLLM not available. Classifier-based evaluation will be skipped.")
            logger.warning("Install vLLM to enable classifier evaluation: pip install vllm")
        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)
            logger.warning("Classifier-based evaluation will be skipped.")

    try:
        # Load input JSON using absolute path (CWD has been changed)
        input_path = Path(original_cwd) / args.data_dir if not os.path.isabs(args.data_dir) else Path(args.data_dir)
        logger.info("Loading input JSON from %s ...", input_path)
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        logger.info("Loaded %d records", len(records))

        if args.limit is not None and args.limit > 0:
            records = records[:args.limit]
            logger.info("Limited to %d records", len(records))

        # Process each record
        processed = []

        for idx, record in enumerate(records):
            try:
                metadata = record.get("metadata", {})

                behavior_id = metadata.get("BehaviorID")
                if not behavior_id:
                    record["error"] = "BehaviorID not found in metadata"
                    processed.append(record)
                    continue

                tags_str = metadata.get("Tags", "")
                tags = tags_str.split(", ") if tags_str else []

                # Build behavior_dict
                # Note: Behavior is stored in 'prompt' field (from harmbench.py)
                behavior_dict = {
                    "BehaviorID": behavior_id,
                    "Behavior": record.get("prompt", "") or "",
                    "ContextString": metadata.get("ContextString", ""),
                    "Tags": tags_str,
                }

                generation = record.get("response", "") or ""

                # Truncate generation before passing to classifier
                if cls is not None and "hash_check" not in tags:
                    tokenized_text = cls_tokenizer.encode(
                        generation,
                        max_length=args.num_tokens,
                        truncation=True,
                    )
                    generation = cls_tokenizer.decode(tokenized_text, skip_special_tokens=True)

                data = [{"generation": generation}]

                # Route to hash-based or classifier-based evaluation
                if "hash_check" in tags:
                    if "book" not in tags and "lyrics" not in tags:
                        record["error"] = f"Expected 'book' or 'lyrics' in tags alongside 'hash_check', got {tags}"
                        processed.append(record)
                        continue

                    results = compute_results_hashing(behavior_dict, data)

                else:
                    if cls is None:
                        record["error"] = "Classifier required for standard/contextual evaluation. Provide --cls_path argument."
                        processed.append(record)
                        continue

                    results = compute_results_classifier(behavior_dict, data, cls, cls_params)

                # Extract label: 1 = compliant (unsafe), 0 = refusal
                if results and len(results) > 0:
                    label = results[0].get("label", 0)
                    record["hb_label"] = label
                else:
                    record["error"] = "Evaluation returned empty results"

                processed.append(record)

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                processed.append(record)

            if (idx + 1) % 10 == 0:
                logger.info("Processed %d / %d records...", idx + 1, len(records))

        # Save output using absolute path (CWD has been changed)
        output_path = Path(original_cwd) / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)
        logger.info("Saving results to %s...", output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        # Summary
        with_label = sum(1 for r in processed if "hb_label" in r)
        with_error = sum(1 for r in processed if "error" in r)
        compliant  = sum(1 for r in processed if r.get("hb_label") == 1)
        refused    = sum(1 for r in processed if r.get("hb_label") == 0)

        logger.info("=" * 50)
        logger.info("Summary")
        logger.info("=" * 50)
        logger.info("  Total records         : %d", len(processed))
        logger.info("  Records with hb_label : %d", with_label)
        logger.info("    Compliant (label=1) : %d", compliant)
        logger.info("    Refused   (label=0) : %d", refused)
        logger.info("  Records with error    : %d", with_error)
        if with_label > 0:
            logger.info("  Attack Success Rate   : %.1f%%", compliant / with_label * 100)
        logger.info("=" * 50)

    finally:
        end_time = time.time()
        elapsed_float = end_time - start_time
        elapsed = int(elapsed_float)
        days = elapsed // 86400
        hours = (elapsed % 86400) // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        logger.info("=== Job finished at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
        logger.info("=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===",
                    days, hours, minutes, seconds, elapsed_float)
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()