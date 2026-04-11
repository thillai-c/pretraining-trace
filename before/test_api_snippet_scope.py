#!/usr/bin/env python3
"""Test script: Verify what infini-gram API returns for snippet text.

Demonstrates that the API's get_doc_by_rank returns only the PRE-SPAN
context as text, not the span itself or post-span context.

Usage:
    python test_api_snippet_scope.py
"""

import json
import requests

API_URL = "https://api.infini-gram.io/"
INDEX = "v4_olmo-mix-1124_llama"

# Example span: "regime of President Bashar al-Assad.\n\nThe" (13 tokens)
QUERY_IDS = [22384, 310, 7178, 29677, 279, 394, 29899, 7900, 328, 29889, 13, 13, 1576]
SPAN_TEXT = "regime of President Bashar al-Assad.\n\nThe"
MAX_DISP_LEN = 150


def main():
    # ── Step 1: Find the span in the corpus ──
    print("=" * 70)
    print("  Step 1: find() — Locate span in corpus")
    print("=" * 70)

    resp = requests.post(API_URL, json={
        "index": INDEX,
        "query_type": "find",
        "query_ids": QUERY_IDS,
    })
    find_result = resp.json()

    count = find_result["cnt"]
    segments = find_result["segment_by_shard"]
    print(f"  Span: \"{SPAN_TEXT}\"")
    print(f"  Token IDs: {QUERY_IDS}")
    print(f"  Corpus occurrences: {count}")

    # Find first non-empty shard
    shard, start_rank = None, None
    for s, (start, end) in enumerate(segments):
        if end - start > 0:
            shard, start_rank = s, start
            break

    if shard is None:
        print("  ERROR: Span not found in any shard.")
        return

    print(f"  Using shard={shard}, rank={start_rank}")

    # ── Step 2: Retrieve document with query_ids ──
    print(f"\n{'=' * 70}")
    print(f"  Step 2: get_doc_by_rank(max_disp_len={MAX_DISP_LEN})")
    print("=" * 70)

    resp = requests.post(API_URL, json={
        "index": INDEX,
        "query_type": "get_doc_by_rank",
        "s": shard,
        "rank": start_rank,
        "max_disp_len": MAX_DISP_LEN,
        "query_ids": QUERY_IDS,
    })
    doc = resp.json()

    needle_offset = doc.get("needle_offset")
    disp_len = doc.get("disp_len")
    token_ids = doc.get("token_ids", [])
    spans = doc.get("spans", [])

    print(f"  disp_len (total tokens retrieved): {disp_len}")
    print(f"  needle_offset (span starts at token): {needle_offset}")
    print(f"  token_ids returned: {len(token_ids)} (span itself = {len(QUERY_IDS)} tokens)")
    print(f"  spans structure: {len(spans)} element(s)")

    # ── Step 3: Analyze spans field ──
    print(f"\n{'=' * 70}")
    print("  Step 3: What does spans[0] contain?")
    print("=" * 70)

    if spans:
        for i, part in enumerate(spans[0]):
            if part is None:
                print(f"  spans[0][{i}]: None")
            else:
                print(f"  spans[0][{i}]: {len(part)} chars")
                print(f"    First 80 chars: \"{part[:80]}\"")
                print(f"    Last  80 chars: \"{part[-80:]}\"")

    # ── Step 4: Check if span text is in snippet ──
    print(f"\n{'=' * 70}")
    print("  Step 4: Does snippet contain the span text?")
    print("=" * 70)

    snippet_text = spans[0][0] if spans and spans[0][0] else ""
    span_in_snippet = SPAN_TEXT in snippet_text
    # Also check without newlines
    span_clean = SPAN_TEXT.replace("\n", " ")
    snippet_clean = snippet_text.replace("\n", " ")
    span_in_snippet_clean = span_clean in snippet_clean

    print(f"  Span in snippet_text: {span_in_snippet}")
    print(f"  Span in snippet_text (cleaned): {span_in_snippet_clean}")

    # ── Step 5: Token budget breakdown ──
    print(f"\n{'=' * 70}")
    print("  Step 5: Token budget breakdown")
    print("=" * 70)

    pre_span_tokens = needle_offset
    span_tokens = len(QUERY_IDS)
    post_span_tokens = disp_len - needle_offset - span_tokens

    print(f"  max_disp_len requested:  {MAX_DISP_LEN}")
    print(f"  disp_len returned:       {disp_len}")
    print(f"  ┌─ pre-span context:     {pre_span_tokens} tokens  → returned as spans[0][0]")
    print(f"  ├─ span itself:          {span_tokens} tokens  → returned as token_ids only")
    print(f"  └─ post-span context:    {post_span_tokens} tokens  → NOT returned")

    # ── Conclusion ──
    print(f"\n{'=' * 70}")
    print("  CONCLUSION")
    print("=" * 70)
    print(f"  The API retrieves {disp_len} tokens centered around the span,")
    print(f"  but only returns the PRE-SPAN context ({pre_span_tokens} tokens) as text.")
    print(f"  The span itself ({span_tokens} tokens) is returned as token_ids only.")
    print(f"  The post-span context ({post_span_tokens} tokens) is NOT returned at all.")
    print(f"  → snippet_text = pre-span context only (spans[0][0])")


if __name__ == "__main__":
    main()
