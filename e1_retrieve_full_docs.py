#!/usr/bin/env python3
"""
e1_retrieve_full_docs_v2.py — Retrieve full documents for all unique doc_ix
in the 9 partially-compliant records.

=== v1 BUG ===
In v1, get_doc_by_rank(s=0, rank=doc_ix) was called. However, the rank parameter of get_doc_by_rank() is the rank within the suffix array, not the document index (doc_ix). As a result, a completely different document was retrieved.

=== v2 Fixing Strategy ===
1. For each unique doc_ix, obtain the snippet_token_ids of a snippet that contains that doc_ix from e1_verbatim_standard_9cases.json.
2. Use find(snippet_token_ids) → segment_by_shard to find locations in the suffix array. (snippet_token_ids are 51~80 tokens and nearly unique in the corpus)
3. For each occurrence, check doc_ix using get_doc_by_rank(s, rank, max_disp_len=small).
4. If it matches the target doc_ix, fetch the full document with get_doc_by_rank(s, rank, max_disp_len=doc_len).

Usage:
    python e1_retrieve_full_docs_v2.py \
        --input_json results/gpt_j_6b/e1_verbatim_standard_9cases.json \
        --index_dir ./index \
        --output_json data/gpt_j_6b/full_docs_9cases.json \
        --tokenizer EleutherAI/gpt-j-6B
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from infini_gram.engine import InfiniGramEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve full documents for all unique doc_ix in 9 cases JSON (v2: correct retrieval)."
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
        "--tokenizer", type=str, default="EleutherAI/gpt-j-6B",
        help="Tokenizer name (default: EleutherAI/gpt-j-6B)"
    )
    parser.add_argument(
        "--max_disp_len_cap", type=int, default=50000,
        help="Maximum disp_len to request (cap for very large documents). "
             "Default: 50000 tokens"
    )
    return parser.parse_args()


def collect_doc_ix_to_snippet(records):
    """Collect mapping: doc_ix -> {doc_len, snippet_token_ids}.

    For each unique doc_ix, we pick the first snippet that references it.
    The snippet_token_ids (51~80 tokens) will be used to re-find the document
    in the corpus via find().
    """
    doc_map = {}  # doc_ix -> {doc_len, snippet_token_ids}
    for rec in records:
        for es in rec["e1"]["ExampleSnippets"]:
            for snip in es["snippets"]:
                dix = snip["doc_ix"]
                if dix not in doc_map:
                    doc_map[dix] = {
                        "doc_ix": dix,
                        "doc_len": snip["doc_len"],
                        "snippet_token_ids": snip["snippet_token_ids"],
                    }
    return doc_map


def retrieve_full_doc(engine, tokenizer, doc_ix, expected_doc_len,
                      snippet_token_ids, max_disp_len_cap):
    """Retrieve the full document for a given doc_ix.

    Strategy:
      1. find(snippet_token_ids) → get occurrences in suffix array
      2. For each occurrence, get_doc_by_rank(s, rank) → check if doc_ix matches
      3. If match, re-call get_doc_by_rank with large max_disp_len to get full text

    Args:
        engine: InfiniGramEngine instance
        tokenizer: tokenizer for decoding
        doc_ix: target document index to find
        expected_doc_len: expected document length (tokens) from original snippet
        snippet_token_ids: token IDs of the snippet (51~80 tokens, used as search key)
        max_disp_len_cap: maximum disp_len to request

    Returns:
        dict with {doc_ix, doc_len, disp_len, metadata, full_text} or None
    """
    # Step 1: find() with snippet_token_ids to locate in suffix array
    find_result = engine.find(input_ids=snippet_token_ids)
    segments = find_result.get("segment_by_shard", [])

    # Step 2: iterate occurrences, find matching doc_ix
    for s, (start_rank, end_rank) in enumerate(segments):
        count_in_shard = end_rank - start_rank
        if count_in_shard <= 0:
            continue

        for rank in range(start_rank, end_rank):
            try:
                # First call: small disp_len just to check doc_ix
                doc = engine.get_doc_by_rank(
                    s=s, rank=rank, max_disp_len=1
                )
                if "error" in doc:
                    continue

                returned_doc_ix = doc.get("doc_ix")
                if returned_doc_ix != doc_ix:
                    continue

                # Match found! Now get full document
                actual_doc_len = doc.get("doc_len", expected_doc_len)
                full_disp_len = min(actual_doc_len, max_disp_len_cap)

                full_doc = engine.get_doc_by_rank(
                    s=s, rank=rank, max_disp_len=full_disp_len
                )
                if "error" in full_doc:
                    print(f"  ERROR: full retrieval failed for doc_ix={doc_ix}: "
                          f"{full_doc['error']}", file=sys.stderr)
                    return None

                token_ids = full_doc.get("token_ids", [])
                full_text = tokenizer.decode(token_ids)

                return {
                    "doc_ix": full_doc.get("doc_ix", doc_ix),
                    "doc_len": full_doc.get("doc_len", len(token_ids)),
                    "disp_len": full_doc.get("disp_len", len(token_ids)),
                    "metadata": full_doc.get("metadata", ""),
                    "full_text": full_text,
                }

            except Exception as e:
                print(f"  WARNING: error at s={s}, rank={rank}: {e}",
                      file=sys.stderr)
                continue

    # Not found
    print(f"  WARNING: doc_ix={doc_ix} not found via snippet_token_ids "
          f"(len={len(snippet_token_ids)})", file=sys.stderr)
    return None


def main():
    args = parse_args()

    # 1. Load input JSON
    print(f"Loading input: {args.input_json}")
    with open(args.input_json, "r") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records")

    # 2. Collect doc_ix → snippet mapping
    doc_map = collect_doc_ix_to_snippet(records)
    print(f"  Found {len(doc_map)} unique doc_ix")

    # 3. Load existing output (resume support)
    output_path = Path(args.output_json)
    existing_docs = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            existing_docs = json.load(f)
        # Filter out docs that might have been incorrectly retrieved in v1
        # by checking doc_len consistency
        valid_docs = {}
        invalid_count = 0
        for dix_str, fd in existing_docs.items():
            dix_int = int(dix_str)
            if dix_int in doc_map:
                expected_len = doc_map[dix_int]["doc_len"]
                if fd.get("doc_len") == expected_len:
                    valid_docs[dix_str] = fd
                else:
                    invalid_count += 1
            else:
                valid_docs[dix_str] = fd
        existing_docs = valid_docs
        if invalid_count > 0:
            print(f"  Discarded {invalid_count} docs with doc_len mismatch (v1 bug)")
        print(f"  Loaded {len(existing_docs)} valid existing docs (resume)")

    remaining = {dix: info for dix, info in doc_map.items()
                 if str(dix) not in existing_docs}
    print(f"  Remaining to retrieve: {len(remaining)}")

    if len(remaining) == 0:
        print("  All docs already retrieved. Done.")
        return

    # 4. Initialize local engine
    print(f"Initializing InfiniGramEngine...")
    print(f"  index_dir: {args.index_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  Tokenizer: {args.tokenizer} (vocab_size={tokenizer.vocab_size})")

    engine = InfiniGramEngine(
        s3_names=[],
        index_dir=args.index_dir,
        eos_token_id=tokenizer.eos_token_id,
        version=4,
        token_dtype="u16",
    )
    print(f"  Engine ready.")

    # 5. Retrieve full documents
    print(f"\nRetrieving {len(remaining)} documents...")
    start_time = time.time()
    success = 0
    not_found = 0
    errors = 0

    for i, (doc_ix, info) in enumerate(sorted(remaining.items())):
        doc_result = retrieve_full_doc(
            engine=engine,
            tokenizer=tokenizer,
            doc_ix=doc_ix,
            expected_doc_len=info["doc_len"],
            snippet_token_ids=info["snippet_token_ids"],
            max_disp_len_cap=args.max_disp_len_cap,
        )

        if doc_result is not None:
            existing_docs[str(doc_ix)] = doc_result
            success += 1

            # Sanity check
            if doc_result["doc_len"] != info["doc_len"]:
                print(f"  WARNING doc_ix={doc_ix}: expected doc_len={info['doc_len']}, "
                      f"got {doc_result['doc_len']}")
        else:
            not_found += 1

        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(remaining)}] "
                  f"success={success}, not_found={not_found}, "
                  f"rate={rate:.1f} docs/sec, "
                  f"elapsed={elapsed:.1f}s")

        # Incremental save every 50 docs
        if (i + 1) % 50 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(existing_docs, f, ensure_ascii=False)
            print(f"  [Incremental save: {len(existing_docs)} docs]")

    # 6. Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(existing_docs, f, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\nDone.")
    print(f"  Success: {success}")
    print(f"  Not found: {not_found}")
    print(f"  Total in output: {len(existing_docs)} docs")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {args.output_json}")


if __name__ == "__main__":
    main()
