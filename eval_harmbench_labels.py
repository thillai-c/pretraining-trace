#!/usr/bin/env python3
"""Evaluate HarmBench compliance labels for copyright, standard, and contextual categories.

For copyright behaviors (hash_check tag), uses hash-based evaluation.
For standard/contextual behaviors, uses classifier-based evaluation (requires --cls_path).

Usage:
    # Copyright evaluation (hash-based, no classifier needed)
    python eval_harmbench_labels.py --data_dir data/gpt_j_6b/harmbench_copyright.json --output_dir results/gpt_j_6b/harmbench_copyright_labeled.json
    
    # Standard/contextual evaluation (classifier-based)
    python eval_harmbench_labels.py --data_dir data/gpt_j_6b/harmbench_standard.json --output_dir results/gpt_j_6b/harmbench_standard_labeled.json --cls_path cais/HarmBench-Llama-2-13b-cls
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import HarmBench eval utils
sys.path.insert(0, str(Path(__file__).parent / "HarmBench"))
from eval_utils import compute_results_hashing, compute_results_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HarmBench compliance labels")
    parser.add_argument("--data_dir", type=str, required=True, help="Input JSON file (list of records)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process")
    parser.add_argument("--cls_path", type=str, default=None, help="Path to classifier model for standard/contextual evaluation (e.g., 'cais/HarmBench-Llama-2-13b-cls'). Required for non-copyright categories.")
    parser.add_argument("--num_tokens", type=int, default=512, help="Maximum number of tokens to evaluate (for classifier)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Change to HarmBench directory for compute_results_hashing to find data files
    harmbench_dir = Path(__file__).parent / "HarmBench"
    original_cwd = os.getcwd()
    os.chdir(harmbench_dir)
    
    # Initialize classifier if needed (for standard/contextual categories)
    cls = None
    cls_params = None
    if args.cls_path:
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            print(f"Loading classifier from {args.cls_path}...")
            tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
            tokenizer.truncation_side = "right"
            
            cls = LLM(model=args.cls_path, tensor_parallel_size=1)
            cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
            cls_params = SamplingParams(temperature=0.0, max_tokens=1)
            print("Classifier loaded successfully")
        except ImportError:
            print("WARNING: vLLM not available. Classifier-based evaluation (standard/contextual) will be skipped.")
            print("Install vLLM to enable classifier evaluation: pip install vllm")
        except Exception as e:
            print(f"WARNING: Failed to load classifier: {e}")
            print("Classifier-based evaluation (standard/contextual) will be skipped.")
    
    try:
        # Load input JSON (use absolute path since we changed directory)
        input_path = Path(original_cwd) / args.data_dir if not os.path.isabs(args.data_dir) else args.data_dir
        print(f"Loading input JSON from {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"Loaded {len(records)} records")
        
        # Limit if specified
        if args.limit is not None and args.limit > 0:
            records = records[:args.limit]
            print(f"Limited to {len(records)} records")
        
        # Process each record
        processed = []
        for record in records:
            try:
                metadata = record.get("metadata", {})
                behavior_id = metadata.get("BehaviorID")
                if not behavior_id:
                    record["error"] = "BehaviorID not found in metadata"
                    processed.append(record)
                    continue
                
                tags_str = metadata.get("Tags", "")
                tags = tags_str.split(", ") if tags_str else []
                
                # Build behavior_dict from metadata
                # Note: Behavior is stored in 'prompt' field (from harmbench.py)
                behavior_dict = {
                    "BehaviorID": behavior_id,
                    "Behavior": record.get("prompt", "") or "",
                    "ContextString": metadata.get("ContextString", ""),
                    "Tags": tags_str,
                }
                
                generation = record.get("response", "")
                
                # Truncate generation if using classifier (to match HarmBench behavior)
                if cls is not None and "hash_check" not in tags:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
                    tokenized_text = tokenizer.encode(generation, max_length=args.num_tokens, truncation=True)
                    generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
                
                data = [{"generation": generation}]
                
                # Choose evaluation method based on tags
                if "hash_check" in tags:
                    # Copyright category: use hash-based evaluation
                    # Verify that 'book' or 'lyrics' is in tags (required by compute_results_hashing)
                    if "book" not in tags and "lyrics" not in tags:
                        record["error"] = f"Expected 'book' or 'lyrics' in tags alongside 'hash_check', got {tags}"
                        processed.append(record)
                        continue
                    
                    # Call compute_results_hashing
                    results = compute_results_hashing(behavior_dict, data)
                else:
                    # Standard or contextual category: use classifier-based evaluation
                    if cls is None:
                        record["error"] = "Classifier required for standard/contextual evaluation. Provide --cls_path argument."
                        processed.append(record)
                        continue
                    
                    # Call compute_results_classifier
                    results = compute_results_classifier(behavior_dict, data, cls, cls_params)
                
                # Extract label and add to record
                if results and len(results) > 0:
                    label = results[0].get("label", 0)
                    record["hb_label"] = label
                else:
                    record["error"] = "Evaluation returned empty results"
                
                processed.append(record)
                
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                processed.append(record)
        
        # Save output (use absolute path since we changed directory)
        output_path = Path(original_cwd) / args.output_dir if not os.path.isabs(args.output_dir) else args.output_dir
        print(f"Saving results to {output_path}...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        # Print summary
        with_label = sum(1 for r in processed if "hb_label" in r)
        with_error = sum(1 for r in processed if "error" in r)
        print(f"\nSummary:")
        print(f"  Total records: {len(processed)}")
        print(f"  Records with hb_label: {with_label}")
        print(f"  Records with error: {with_error}")
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()