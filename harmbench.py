#!/usr/bin/env python3
"""Generate HarmBench prompts using GPT-J-6B and save results as JSON."""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/gpt-j-6B"
GPT_J_MAX_CTX = 2048


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
    parser = argparse.ArgumentParser(description="Run GPT-J on HarmBench prompts")
    parser.add_argument("--csv_path", type=str, default="HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv")
    parser.add_argument("--out_json", type=str, default="data/gpt_j_6b/harmbench_copyright.json")
    parser.add_argument("--config", type=str, default="copyright", help="FunctionalCategory to filter (e.g., 'copyright', 'standard')")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    return parser.parse_args()


def main():
    logger = setup_logger()
    args = parse_args()
    
    logger.info("=== Job started at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
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
    logger.info("Total rows: %d, After filtering with '%s': %d", total_rows, args.config, len(df))
    
    if df.empty:
        logger.warning("No rows found for the specified config. Exiting.")
        sys.exit(0)
    
    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)
        logger.info("Limited to first %d samples for testing", args.max_samples)
    
    # Replace NaN with None for JSON serialization
    for col in ["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"]:
        if col in df.columns:
            df[col] = df[col].astype(object).where(pd.notna(df[col]), None)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("Loading model: %s ...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16 if device == "cuda" else torch.float32, low_cpu_mem_usage=True).to(device)
    model.eval()
    eos_token_id = tokenizer.eos_token_id
    
    # Generate
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
            prompt_text = str(row.get("Behavior", "")) if row.get("Behavior") is not None else ""
            try:
                input_ids = tokenizer(prompt_text, return_tensors="pt", truncation=False).input_ids.to(device)
                prompt_len = input_ids.shape[1]
                effective_max_new_tokens = min(args.max_new_tokens, GPT_J_MAX_CTX - prompt_len)
                if effective_max_new_tokens <= 0:
                    raise ValueError(f"Prompt token length ({prompt_len}) exceeds GPT-J max context ({GPT_J_MAX_CTX}).")
                
                with torch.no_grad():
                    output = model.generate(input_ids, do_sample=False, max_new_tokens=effective_max_new_tokens, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
                
                generated_ids = output[0, prompt_len:]
                response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                finish_reason = "eos" if output[0, -1].item() == eos_token_id else "length"
                
                record = {
                    "id": idx,
                    "config": args.config,
                    "prompt": safe_value(row.get("Behavior", "")) or "",
                    "response": response_text,
                    "model": MODEL_ID,
                    "finish_reason": finish_reason,
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
                record = {"id": idx, "prompt": safe_value(row.get("Behavior", "")) or "", "error": error_msg, "model": MODEL_ID}
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            if not first:
                fout.write(",\n")
            first = False
            fout.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            fout.flush()
        fout.write("\n]\n")
    
    logger.info("Results saved to: %s", args.out_json)
    
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


if __name__ == "__main__":
    main()