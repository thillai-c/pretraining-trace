#!/usr/bin/env python3
"""
For all unique doc_ixs in 9 partially compliant records,
fetch the full document from the local InfiniGram engine and save as JSON.

Usage:
    python e1_retrieve_full_docs.py \
        --input_json results/gpt_j_6b/e1_verbatim_standard_9cases.json \
        --index_dir ./index \
        --output_json data/gpt_j_6b/full_docs_9cases.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer

# Path setup (same as e1_retrieve_snippets.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from infini_gram.engine import InfiniGramEngine

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve full documents for all unique doc_ix in 9 cases JSON."
    )
    parser.add_argument(
        "--input_json", type=str, required=True,
        help="Path to e1_verbatim_standard_9cases.json"
    )
    parser.add_argument(
        "--index_dir", type=str, required=True,
        help="Path to local infini-gram index directory (The Pile)"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path to save full documents JSON"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer name — must match the tokenizer used to build the "
             "infini-gram index (default: meta-llama/Llama-2-7b-hf)"
    )
    return parser.parse_args()


def collect_unique_doc_ix(records):
    """Collect all unique doc_ix from ExampleSnippets across all records.
    Also collect doc_len for reference."""
    doc_info = {}  # doc_ix -> doc_len
    for rec in records:
        for es in rec['e1']['ExampleSnippets']:
            for sn in es['snippets']:
                dix = sn['doc_ix']
                dlen = sn['doc_len']
                if dix not in doc_info:
                    doc_info[dix] = dlen
    return doc_info


def retrieve_full_doc(engine, tokenizer, doc_ix, expected_doc_len):
    """Retrieve full document using get_doc_by_rank().
    
    Returns:
        dict with keys: doc_ix, doc_len, metadata, full_text, full_token_ids
        or None if retrieval fails.
    """
    try:
        # s=0: shard index (only 1 shard)
        # max_disp_len: must be >= doc_len to get full document
        result = engine.get_doc_by_rank(
            s=0,
            rank=doc_ix,
            max_disp_len=max(expected_doc_len + 100, 10000)
        )
        
        if 'error' in result:
            print(f"  ERROR doc_ix={doc_ix}: {result['error']}", file=sys.stderr)
            return None
        
        token_ids = result.get('token_ids', [])
        metadata = result.get('metadata', '')
        
        # Decode token IDs to text
        full_text = tokenizer.decode(token_ids)
        
        return {
            'doc_ix': result.get('doc_ix', doc_ix),
            'doc_len': result.get('doc_len', len(token_ids)),
            'disp_len': result.get('disp_len', 0),
            'metadata': metadata,
            # 'full_token_ids': token_ids,
            'full_text': full_text
        }
    except Exception as e:
        print(f"  ERROR retrieving doc_ix={doc_ix}: {e}", file=sys.stderr)
        return None


def main():
    args = parse_args()
    
    # 1. Load input JSON
    print(f"Loading input: {args.input_json}")
    with open(args.input_json, 'r') as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")
    
    # 2. Collect unique doc_ix
    doc_info = collect_unique_doc_ix(records)
    print(f"  Found {len(doc_info)} unique doc_ix")
    
    # 3. Load existing output (resume support)
    output_path = Path(args.output_json)
    existing_docs = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_docs = json.load(f)
        print(f"  Loaded {len(existing_docs)} existing docs from output (resume)")
    
    remaining = {dix: dlen for dix, dlen in doc_info.items()
                 if str(dix) not in existing_docs}
    print(f"  Remaining to retrieve: {len(remaining)}")
    
    if len(remaining) == 0:
        print("  All docs already retrieved. Done.")
        return
    
    # 4. Initialize local engine
    print(f"Initializing InfiniGramEngine...")
    print(f"  index_dir: {args.index_dir}")
    print(f"  tokenizer: {args.tokenizer}")
    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=args.index_dir,
        eos_token_id=50256,  # GPT-J EOS token
        version=4,
        token_dtype="u16",
    )
    print(f"  Engine ready.")

    # 5. Load tokenizer (once, outside loop)
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  Tokenizer ready. vocab_size={tokenizer.vocab_size}")
    
    # 6. Retrieve full documents
    print(f"\nRetrieving {len(remaining)} documents...")
    start_time = time.time()
    success = 0
    errors = 0
    
    for i, (doc_ix, expected_len) in enumerate(sorted(remaining.items())):
        doc_result = retrieve_full_doc(engine, tokenizer, doc_ix, expected_len)
        
        if doc_result is not None:
            existing_docs[str(doc_ix)] = doc_result
            success += 1
            
            # Sanity check: doc_len match
            if doc_result['doc_len'] != expected_len:
                print(f"  WARNING doc_ix={doc_ix}: expected doc_len={expected_len}, "
                      f"got {doc_result['doc_len']}")
        else:
            errors += 1
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(remaining)}] "
                  f"success={success}, errors={errors}, "
                  f"rate={rate:.1f} docs/sec, "
                  f"elapsed={elapsed:.1f}s")
        
        # Save every 50 docs (incremental save)
        if (i + 1) % 50 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(existing_docs, f, ensure_ascii=False)
            print(f"  [Incremental save: {len(existing_docs)} docs]")
    
    # 7. Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(existing_docs, f, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\nDone. Retrieved {success} docs, {errors} errors, "
          f"total {len(existing_docs)} docs saved.")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {args.output_json}")


if __name__ == '__main__':
    main()