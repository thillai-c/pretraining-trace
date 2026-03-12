#!/usr/bin/env python3
"""
Generate HarmBench responses using GPT-J-6B or OLMo 2 family models.

Supports:
  - GPT-J-6B (original baseline)
  - OLMo 2 family: 1B, 7B, 13B, 32B (base and instruct variants)

For base models, the raw prompt is fed directly (text completion).
For instruct models, the prompt is wrapped with the model's chat template.
"""

import os
import sys
import json
import time
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Model Registry
# =============================================================================
# hf_id:        HuggingFace model ID
# max_ctx:      max token count (prompt + response combined)
# model_type:   "base" (raw text completion) or "instruct" (chat template)
# out_dir_name: output directory under data/

MODEL_REGISTRY = {
    # GPT-J (original baseline)
    "gpt-j": {
        "hf_id": "EleutherAI/gpt-j-6B",
        "max_ctx": 2048,
        "model_type": "base",
        "out_dir_name": "gpt_j_6b",
    },

    # OLMo 2 Base — cl100k tokenizer, seq len 4096 (OLMo 2 paper Table 1)
    "olmo2-1b": {
        "hf_id": "allenai/OLMo-2-0425-1B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_1b",
    },
    "olmo2-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_7b",
    },
    "olmo2-13b": {
        "hf_id": "allenai/OLMo-2-1124-13B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_13b",
    },
    "olmo2-32b": {
        "hf_id": "allenai/OLMo-2-0325-32B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_32b",
    },

    # OLMo 2 Instruct — chat template: <|user|>\n{prompt}\n<|assistant|>\n
    "olmo2-1b-instruct": {
        "hf_id": "allenai/OLMo-2-0425-1B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_1b_instruct",
    },
    "olmo2-7b-instruct": {
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_7b_instruct",
    },
    "olmo2-13b-instruct": {
        "hf_id": "allenai/OLMo-2-1124-13B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_13b_instruct",
    },
    "olmo2-32b-instruct": {
        "hf_id": "allenai/OLMo-2-0325-32B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_32b_instruct",
    },
}


def safe_value(v):
    """Convert pandas/numpy NaN/inf to None for JSON serialization."""
    if pd.isna(v):
        return None
    if isinstance(v, (float, np.floating)):
        if np.isinf(v) or np.isnan(v):
            return None
    return v


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = f"logs/harmbench_{timestamp}.log"

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
        description="Generate HarmBench responses using GPT-J or OLMo 2 family models"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to use. Available: " + ", ".join(MODEL_REGISTRY.keys()))
    parser.add_argument("--csv_path", type=str,
                        default="HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    parser.add_argument("--out_json", type=str, default=None,
                        help="Output JSON path (auto-generated if not specified)")
    parser.add_argument("--config", type=str, default="standard",
                        help="FunctionalCategory to filter (e.g., 'copyright', 'standard', 'contextual')")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    return parser.parse_args()


def format_prompt(prompt_text, model_info, tokenizer):
    """Format prompt: raw text for base models, chat template for instruct models."""
    if model_info["model_type"] == "instruct":
        messages = [{"role": "user", "content": prompt_text}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: manual OLMo 2 instruct template
            formatted_prompt = (
                f"<|endoftext|><|user|>\n{prompt_text}\n<|assistant|>\n"
            )
        return formatted_prompt
    else:
        return prompt_text


def load_model_and_tokenizer(model_info, logger):
    """Load model and tokenizer with size-appropriate strategy.

    <= 13B: single GPU, fp16, .to(device)
    32B:    device_map="auto" for multi-GPU sharding (requires accelerate)

    Returns (model, tokenizer, device). device is None when device_map="auto".
    """
    hf_id = model_info["hf_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    logger.info("Loading tokenizer: %s ...", hf_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s ...", hf_id)
    is_large_model = "32b" in model_info["out_dir_name"]

    if is_large_model and device == "cuda":
        logger.info("  Large model — using device_map='auto' for multi-GPU sharding")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        device = None
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if device == "cuda":
            model = model.to(device)

    model.eval()
    logger.info("Model loaded successfully: %s", hf_id)
    return model, tokenizer, device


def main():
    logger = setup_logger()
    args = parse_args()

    # Resolve model info
    model_info = MODEL_REGISTRY[args.model]
    hf_id = model_info["hf_id"]
    max_ctx = model_info["max_ctx"]

    # Auto-generate output path if not specified
    if args.out_json is None:
        args.out_json = os.path.join(
            "data", model_info["out_dir_name"], f"harmbench_{args.config}.json"
        )

    logger.info("=" * 70)
    logger.info("=== Job started at %s ===",
                datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("=" * 70)
    logger.info("Model key       : %s", args.model)
    logger.info("HuggingFace ID  : %s", hf_id)
    logger.info("Model type      : %s", model_info["model_type"])
    logger.info("Max context len : %d", max_ctx)
    logger.info("Max new tokens  : %d", args.max_new_tokens)
    logger.info("Config (category): %s", args.config)
    logger.info("Output JSON     : %s", args.out_json)
    logger.info("Seed            : %d", args.seed)

    start_time = time.time()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load and filter data
    df = pd.read_csv(args.csv_path)
    total_rows = len(df)
    df = df[df["FunctionalCategory"] == args.config].reset_index(drop=True)
    logger.info("Total rows in CSV: %d, After filtering '%s': %d",
                total_rows, args.config, len(df))

    if df.empty:
        logger.warning("No rows found for config '%s'. Exiting.", args.config)
        sys.exit(0)

    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)
        logger.info("Limited to first %d samples (--max_samples)", args.max_samples)

    # Replace NaN with None for JSON serialization
    for col in ["Behavior", "FunctionalCategory", "SemanticCategory",
                "Tags", "ContextString", "BehaviorID"]:
        if col in df.columns:
            df[col] = df[col].astype(object).where(pd.notna(df[col]), None)

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(model_info, logger)
    eos_token_id = tokenizer.eos_token_id

    # Generate
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
            raw_prompt = (
                str(row.get("Behavior", ""))
                if row.get("Behavior") is not None
                else ""
            )

            try:
                formatted_prompt = format_prompt(raw_prompt, model_info, tokenizer)

                input_ids = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=False,
                ).input_ids

                # Route input to correct device
                if device is None:
                    input_ids = input_ids.to(model.device)
                else:
                    input_ids = input_ids.to(device)

                prompt_len = input_ids.shape[1]

                # Context-limit-aware token budgeting
                effective_max_new_tokens = min(
                    args.max_new_tokens,
                    max_ctx - prompt_len
                )

                if effective_max_new_tokens <= 0:
                    raise ValueError(
                        f"Prompt token length ({prompt_len}) >= "
                        f"max context ({max_ctx}). Cannot generate."
                    )

                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        do_sample=False,
                        max_new_tokens=effective_max_new_tokens,
                        eos_token_id=eos_token_id,
                        pad_token_id=eos_token_id,
                    )

                generated_ids = output[0, prompt_len:]
                response_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                finish_reason = (
                    "eos"
                    if output[0, -1].item() == eos_token_id
                    else "length"
                )

                record = {
                    "id": idx,
                    "config": args.config,
                    "prompt": safe_value(row.get("Behavior", "")) or "",
                    "response": response_text,
                    "model": hf_id,
                    "model_key": args.model,
                    "model_type": model_info["model_type"],
                    "finish_reason": finish_reason,
                    "prompt_token_len": prompt_len,
                    "response_token_len": len(generated_ids),
                    "metadata": {
                        "FunctionalCategory": safe_value(row.get("FunctionalCategory", "")) or "",
                        "SemanticCategory": safe_value(row.get("SemanticCategory", "")) or "",
                        "Tags": safe_value(row.get("Tags", "")) or "",
                        "ContextString": safe_value(row.get("ContextString", "")) or "",
                        "BehaviorID": safe_value(row.get("BehaviorID", "")) or "",
                    },
                }

            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.error("id=%d — %s", idx, error_msg)
                record = {
                    "id": idx,
                    "prompt": safe_value(row.get("Behavior", "")) or "",
                    "error": error_msg,
                    "model": hf_id,
                    "model_key": args.model,
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            f.flush()

        f.write("\n]\n")

    logger.info("Results saved to: %s", args.out_json)

    # Summary
    try:
        with open(args.out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        successful = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info("  Model          : %s (%s)", args.model, hf_id)
        logger.info("  Config         : %s", args.config)
        logger.info("  Total records  : %d", len(results))
        logger.info("  Successful     : %d", len(successful))
        logger.info("  Errors         : %d", len(errors))
        if successful:
            eos_count = sum(1 for r in successful if r.get("finish_reason") == "eos")
            len_count = sum(1 for r in successful if r.get("finish_reason") == "length")
            resp_lens = [r.get("response_token_len", 0) for r in successful]
            logger.info("  Finish reason  : eos=%d, length=%d", eos_count, len_count)
            logger.info("  Response tokens: min=%d, max=%d, mean=%.1f",
                        min(resp_lens), max(resp_lens),
                        sum(resp_lens) / len(resp_lens))
    except Exception as exc:
        logger.warning("Could not read results for summary: %s", exc)

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
