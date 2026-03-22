import json
import numpy as np

def display_record(e1_result, id=None, index=None, show_e1=True, indent=2):
    # Find record by id or index
    record = None
    record_index = None
    
    if id is not None:
        # Search by ID
        for idx, r in enumerate(e1_result):
            if r.get('id') == id:
                record = r
                record_index = idx
                break
        if record is None:
            print(f"\n{'='*80}")
            print(f"Error: Record with ID {id} not found")
            print(f"{'='*80}\n")
            return
    elif index is not None:
        # Use index
        if index < 0 or index >= len(e1_result):
            print(f"\n{'='*80}")
            print(f"Error: Index {index} is out of range. Valid range: 0 to {len(e1_result) - 1}")
            print(f"{'='*80}\n")
            return
        record = e1_result[index]
        record_index = index
    else:
        # Default to index 0
        record = e1_result[0]
        record_index = 0
    
    hb_label = record.get('hb_label')
    if hb_label == 1:
        label_str = "Compliant (hb_label=1)"
    elif hb_label == 0:
        label_str = "Non-Compliant (hb_label=0)"
    else:
        label_str = f"Unknown (hb_label={hb_label})"
    title = f"{label_str} - Record ID: {record.get('id', 'N/A')} (from {len(e1_result)} total records)"\
    
    record_id = record.get('id')
    
    # Display the record (without e1 if show_e1 is False)
    print(f"\n{'='*80}")
    if id is not None:
        print(f"{title}")
    else:
        print(f"{title} - Displaying index {record_index}")
    print(f"{'='*80}\n")
    
    if not show_e1:
        # Display record excluding 'e1' key
        record_copy = record.copy()
        record_copy.pop('e1', None)  # Remove 'e1' key if exists
        print(f"{'─'*80}")
        print(f"Record at index {record_index} (ID: {record_copy.get('id', 'N/A')})")
        print(f"{'─'*80}")
        print(json.dumps(record_copy, indent=indent, ensure_ascii=False))
        print()
    else:
        # Display record excluding 'e1' key first
        record_copy = record.copy()
        record_copy.pop('e1', None)  # Remove 'e1' key if exists
        print(f"{'─'*80}")
        print(f"Record ID: {record_copy.get('id', 'N/A')}")
        print(f"{'─'*80}")
        print(json.dumps(record_copy, indent=indent, ensure_ascii=False))
        print()
        
        # Then display e1 data if available
        if 'e1' not in record:
            print(f"\n{'='*80}")
            print(f"Note: Record at index {record_index} does not have 'e1' key")
            print(f"{'='*80}\n")
            return
        
        e1_data = record['e1']
        
        # Filter to show only specified keys
        e1_filtered = {}
        e1_keys_to_show = [
            'response_token_len',
            'LongestMatchLen',
            'VerbatimCoverage',
            'num_maximal_spans',
            'num_top_k_spans',
            'span_length_distribution'
        ]
        for key in e1_keys_to_show:
            if key in e1_data:
                e1_filtered[key] = e1_data[key]
        
        # Add ExampleSnippets (unique span_text only, first occurrence, without 'snippets' key)
        if 'ExampleSnippets' in e1_data:
            example_snippets = e1_data['ExampleSnippets']
            # Count occurrences of each span_text
            span_text_counts = {}
            for snippet in example_snippets:
                span_text = snippet.get('span_text', '')
                span_text_counts[span_text] = span_text_counts.get(span_text, 0) + 1
            
            # Keep only unique span_text (first occurrence of each unique text) and add count
            seen_texts = set()
            example_snippets_filtered = []
            for snippet in example_snippets:
                span_text = snippet.get('span_text', '')
                if span_text not in seen_texts:
                    seen_texts.add(span_text)
                    snippet_copy = snippet.copy()
                    snippet_copy.pop('snippets', None)  # Remove 'snippets' key
                    snippet_copy['count'] = span_text_counts[span_text]  # Add count
                    example_snippets_filtered.append(snippet_copy)
            e1_filtered['ExampleSnippets'] = example_snippets_filtered
        
        # Display the filtered e1 data
        print(f"\n{'='*80}")
        title_parts = ["E1 Metrics"]
        if id is not None:
            title_parts.append(f"ID: {id}")
        print(" - ".join(title_parts))
        print(f"{'='*80}\n")

        print(json.dumps(e1_filtered, indent=indent, ensure_ascii=False))
        print()


def build_row(r):
    """Extract E1 metrics from a single record into a flat dict."""
    e = r['e1']
    sd = e['span_length_distribution']
    topk = e['top_k_spans']
    topk_lens = [s['length'] for s in topk]

    # Word-level 4-gram repetition ratio (seq-rep-4)
    words = r['response'].split()
    n = 4
    if len(words) >= n:
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)] # sliding window
        repetition = 1.0 - len(set(ngrams)) / len(ngrams)
    else:
        repetition = 0.0

    return {
        'id': r['id'],
        'hb_label': r.get('hb_label', -1),
        'compliance': 'Compliant' if r.get('hb_label') == 1 else 'Non-Compliant',
        'prompt': r['prompt'],
        'response_token_len': e['response_token_len'],
        'LongestMatchLen': e['LongestMatchLen'],
        'VerbatimCoverage': e['VerbatimCoverage'],
        'num_maximal_spans': e['num_maximal_spans'],
        'num_top_k_spans': e['num_top_k_spans'],
        'span_min': sd['min'],
        'span_max': sd['max'],
        'span_mean': sd['mean'],
        'span_median': sd['median'],
        'topk_min': min(topk_lens) if topk_lens else 0,
        'topk_max': max(topk_lens) if topk_lens else 0,
        'topk_mean': np.mean(topk_lens) if topk_lens else 0,
        'topk_ge8': sum(1 for l in topk_lens if l >= 8),
        'topk_ge10': sum(1 for l in topk_lens if l >= 10),
        'repetition_ratio': repetition,
    }

# ============================================================
# Phase 1: Build summary table
# ============================================================

def extract_unique_spans(record):
    """Extract deduplicated spans from ExampleSnippets.
    Returns list of dicts with span info + snippets."""
    e1 = record['e1']
    snippets_list = e1.get('ExampleSnippets', [])
    
    seen_texts = set()
    unique = []
    for sp in snippets_list:
        text = sp.get('span_text', '')
        if text not in seen_texts:
            seen_texts.add(text)
            unique.append(sp)
    return unique

def extract_unique_snippets(span_info):
    """Extract deduplicated snippets from a span.
    Returns list of unique snippet dicts."""
    seen = set()
    unique = []
    for snip in span_info.get('snippets', []):
        stxt = snip.get('snippet_text', '')
        if stxt not in seen:
            seen.add(stxt)
            unique.append(snip)
    return unique

def display_snippets(record, unique_spans, max_snippet_chars=None):
    """Display a record's spans and snippets for labeling reference."""
    rid = record['id']
    e1 = record['e1']
    prompt = record.get('prompt', '')
    response = record.get('response', '')

    # Header
    print("=" * 80)
    print(f"Record id={rid}")
    print(f"  Prompt:           {prompt}")
    print(f"  Response:         {len(response)} chars")
    print(f"  LongestMatchLen:  {e1.get('LongestMatchLen')}")
    print(f"  VerbatimCoverage: {e1.get('VerbatimCoverage')}")
    print(f"  Maximal spans:    {e1.get('num_maximal_spans')}")
    print(f"  Top-K spans:      {e1.get('num_top_k_spans')}")
    print(f"  Unique spans:     {len(unique_spans)}")
    print(f"  Response tokens:  {e1.get('response_token_len', 'N/A')}")
    print("=" * 80)

    for i, span_info in enumerate(unique_spans):
        span_text = span_info.get('span_text', '')
        span_len = span_info.get('span_length', 0)
        span_begin = span_info.get('span_begin', '?')
        span_end = span_info.get('span_end', '?')
        num_snips = span_info.get('num_snippets', 0)

        # Span header with clear separator
        print()
        print("-" * 60)
        unique_snip_texts = set(s.get('snippet_text', '') for s in span_info.get('snippets', []))
        print(f"Span {i} (tokens {span_begin}:{span_end}, "
              f"length={span_len}, {len(unique_snip_texts)} unique / {num_snips} total snippet(s))")
        print(f"  span_text: \"{span_text}\"")
        print("-" * 60)

        # Count occurrences of each unique snippet text
        snip_text_counts = {}
        for snip in span_info.get('snippets', []):
            st = snip.get('snippet_text', '')
            snip_text_counts[st] = snip_text_counts.get(st, 0) + 1

        # Show unique snippets (deduplicated, show all)
        seen_snip_texts = set()
        shown = 0
        for j, snip in enumerate(span_info.get('snippets', [])):
            snippet_text = snip.get('snippet_text', '')
            if snippet_text in seen_snip_texts:
                continue
            seen_snip_texts.add(snippet_text)

            if max_snippet_chars and len(snippet_text) > max_snippet_chars:
                display_text = snippet_text[:max_snippet_chars] + '...'
            else:
                display_text = snippet_text

            doc_ix = snip.get('doc_ix', '?')
            doc_len = snip.get('doc_len', '?')
            metadata = snip.get('metadata', '')
            count = snip_text_counts[snippet_text]

            print(f"\n    Snippet {shown} (appears {count}x):")
            print(f"      doc_ix:   {doc_ix}")
            print(f"      doc_len:  {doc_len} tokens")
            if metadata:
                print(f"      metadata: {metadata}")
            print(f"      text:     \"{display_text}\"")

            shown += 1

    print(f"\n{'=' * 80}")
    total_snips = sum(s.get('num_snippets', 0) for s in unique_spans)
    unique_total = sum(len(set(s.get('snippet_text', '') for s in sp.get('snippets', []))) for sp in unique_spans)
    print(f"Total: {len(unique_spans)} unique spans, {unique_total} unique / {total_snips} total snippets")
    print()

# ============================================================
# Phase 2: Interactive Labeling
# ============================================================

def load_existing_labels(csv_path):
    """Load existing labels from CSV. Returns dict of (record_id, span_idx, doc_ix) → row_dict."""
    existing = {}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = (int(row['record_id']), int(row['span_idx']), int(row['doc_ix']))
            existing[key] = row.to_dict()
    return existing

def save_all_labels(csv_path, existing_dict):
    """Overwrite entire CSV from dict. Enables delete/replace of individual rows."""
    columns = ['record_id', 'span_idx', 'doc_ix',
               'context_topic', 'context_safety', 'span_safety_label']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for key in sorted(existing_dict.keys()):
            row = existing_dict[key]
            writer.writerow({c: row[c] for c in columns})

def label_record(record_id, max_snippet_chars=None, max_doc_chars=1500, relabel=False):
    """Interactive labeling for one record — saves per snippet.
    
    Args:
        record_id: which record to label
        max_snippet_chars: truncate snippet display (None = show all)
        max_doc_chars: truncate full document display (default 1500 chars, None = show all)
        relabel: if True, re-label already labeled snippets instead of skipping them
    
    Enter 'q' at any prompt to stop. Each snippet is saved immediately.
    """
    # Find record
    rec = None
    for r in all_records:
        if r['id'] == record_id:
            rec = r
            break
    if rec is None:
        print(f"Record id={record_id} not found.")
        return
    
    unique_spans = all_spans_by_record[record_id]
    existing = load_existing_labels(LABEL_CSV_PATH)
    
    # Count already labeled
    already_done = 0
    total_snips = 0
    for sp_idx, sp in enumerate(unique_spans):
        for snip in extract_unique_snippets(sp):
            total_snips += 1
            if (record_id, sp_idx, snip.get('doc_ix', 0)) in existing:
                already_done += 1
    
    if already_done == total_snips and total_snips > 0 and not relabel:
        print(f"Record id={record_id}: all {total_snips} snippets already labeled. Skipping.")
        print(f"  Tip: use label_record({record_id}, relabel=True) to re-label.")
        return
    elif already_done > 0 and not relabel:
        print(f"Record id={record_id}: {already_done}/{total_snips} snippets already labeled.")
        print(f"  Only unlabeled snippets will be shown.")
        print(f"  Tip: use label_record({record_id}, relabel=True) to re-label.\n")
    elif relabel and already_done > 0:
        print(f"Record id={record_id}: RE-LABELING mode. {already_done} existing labels will be overwritten.")
        print(f"  (Old labels are replaced only for snippets you re-label; others kept as-is.)\n")
    
    # Shortcuts
    context_shortcuts = {'u': 'unsafe_context', 's': 'safe_context', 'a': 'ambiguous_context'}
    span_shortcuts = {'u': 'unsafe', 's': 'safe_but_relevant', 't': 'trivial'}
    valid_context = set(context_shortcuts.values())
    valid_span = set(span_shortcuts.values())
    
    # Header
    e1 = rec['e1']
    print("=" * 80)
    print(f"LABELING Record id={record_id}" + (" [RE-LABEL MODE]" if relabel else ""))
    print(f"  Prompt:           {rec['prompt']}")
    print(f"  LongestMatchLen:  {e1.get('LongestMatchLen')}")
    print(f"  Unique spans:     {len(unique_spans)}")
    print(f"  Total snippets:   {total_snips}")
    print(f"  Already labeled:  {already_done}")
    print(f"  (Enter 'q' at any prompt to stop. Each snippet is saved immediately.)")
    print("=" * 80)
    
    total_saved = 0
    
    for i, sp in enumerate(unique_spans):
        span_text = sp.get('span_text', '')
        span_len = sp.get('span_length', 0)
        span_begin = sp.get('span_begin', '?')
        span_end = sp.get('span_end', '?')
        
        unique_snips = extract_unique_snippets(sp)
        
        # Check if ALL snippets in this span are already labeled
        all_done = all(
            (record_id, i, snip.get('doc_ix', 0)) in existing
            for snip in unique_snips
        )
        if all_done and not relabel:
            print(f"\n[Span {i} already labeled, skipping]")
            continue
        
        # Show span header
        print()
        print("-" * 60)
        print(f"Span {i} (tokens {span_begin}:{span_end}, "
              f"length={span_len}, {len(unique_snips)} unique snippet(s))")
        print(f"  span_text: \"{span_text}\"")
        print("-" * 60)
        
        # Label each snippet individually and save immediately
        for si, snip in enumerate(unique_snips):
            doc_ix = snip.get('doc_ix', '?')
            doc_len = snip.get('doc_len', '?')
            snippet_text = snip.get('snippet_text', '')
            key = (record_id, i, doc_ix)
            
            # Skip if already labeled (per snippet)
            if key in existing and not relabel:
                print(f"\n    [Snippet {si} (doc_ix={doc_ix}) already labeled, skipping]")
                continue
            
            # Show old labels when re-labeling
            if key in existing and relabel:
                old = existing[key]
                print(f"\n    [Snippet {si} (doc_ix={doc_ix}) — OLD LABELS: "
                      f"topic=\"{old.get('context_topic','')}\", "
                      f"ctx_safety={old.get('context_safety','')}, "
                      f"span_safety={old.get('span_safety_label','')}]")
            
            # Display snippet text
            if max_snippet_chars and len(snippet_text) > max_snippet_chars:
                display_snippet = snippet_text[:max_snippet_chars] + '...'
            else:
                display_snippet = snippet_text
            
            print(f"\n    Snippet {si}:")
            print(f"      doc_ix:       {doc_ix}")
            print(f"      doc_len:      {doc_len} tokens")
            print(f"      snippet_text: \"{display_snippet}\"")
            
            # Show full document text
            doc_ix_str = str(doc_ix)
            if doc_ix_str in full_docs:
                full_text = full_docs[doc_ix_str].get('full_text', '')
                if max_doc_chars and len(full_text) > max_doc_chars:
                    display_doc = full_text[:max_doc_chars] + f'... [{len(full_text)} chars total]'
                else:
                    display_doc = full_text
                print(f"      --- FULL DOCUMENT (doc_ix={doc_ix}) ---")
                print(f"      {display_doc}")
                print(f"      --- END FULL DOCUMENT ---")
            else:
                print(f"      [Full document not available for doc_ix={doc_ix}]")
            
            # Input: context_topic
            context_topic = input(f"      context_topic: ").strip()
            if context_topic.lower() == 'q':
                print(f"\n      [Stopped. {total_saved} snippets saved this session.]")
                return
            
            # Input: context_safety
            while True:
                cs = input(f"      context_safety (u=unsafe / s=safe / a=ambiguous): ").strip()
                if cs.lower() == 'q':
                    print(f"\n      [Stopped. {total_saved} snippets saved this session.]")
                    return
                cs = context_shortcuts.get(cs, cs)
                if cs in valid_context:
                    break
                print(f"        Invalid. Enter u, s, a, or q.")
            
            # Input: span_safety_label (per snippet now)
            while True:
                sl = input(f"      span_safety_label (u=unsafe / s=safe_but_relevant / t=trivial): ").strip()
                if sl.lower() == 'q':
                    print(f"\n      [Stopped. {total_saved} snippets saved this session.]")
                    return
                sl = span_shortcuts.get(sl, sl)
                if sl in valid_span:
                    break
                print(f"        Invalid. Enter u, s, t, or q.")
            
            # Save this snippet immediately
            existing[key] = {
                'record_id': record_id,
                'span_idx': i,
                'doc_ix': doc_ix,
                'context_topic': context_topic,
                'context_safety': cs,
                'span_safety_label': sl,
            }
            save_all_labels(LABEL_CSV_PATH, existing)
            total_saved += 1
            print(f"      [Saved snippet {si} (doc_ix={doc_ix}) → "
                  f"total this session: {total_saved}]")
    
    print(f"\n{'=' * 80}")
    if total_saved > 0:
        print(f"Record id={record_id} labeling complete. {total_saved} snippets saved.")
    else:
        print(f"No new labels to save.")


# ============================================================
# Phase 3: Display with Labels
# ============================================================

import json, pandas as pd
from collections import OrderedDict

_label_df = pd.read_csv("span_safety_labels.csv")
_e1_records = json.load(open("../results/gpt_j_6b/e1_verbatim_standard_9cases.json"))
_full_docs  = json.load(open("../data/gpt_j_6b/full_docs_9cases.json"))

# record lookup dict: {id: record}
_rec_map = {r["id"]: r for r in _e1_records}


def view_span(record_id: int, span_idx: int = None):
    # record validation
    assert record_id in _rec_map, (
        f"record_id={record_id} not found. "
        f"Available: {sorted(_rec_map.keys())}"
    )
    rec = _rec_map[record_id]
    e1  = rec["e1"]

    snippets_list = e1["ExampleSnippets"]

    # span_idx가 None이면 모든 span, 아니면 해당 span만
    if span_idx is not None:
        assert 0 <= span_idx < len(snippets_list), (
            f"span_idx={span_idx} out of range (0–{len(snippets_list)-1}) "
            f"for record_id={record_id}"
        )
        span_indices = [span_idx]
    else:
        span_indices = list(range(len(snippets_list)))

    # CSV filter for this record
    rec_mask = _label_df["record_id"] == record_id
    rec_df = _label_df[rec_mask].reset_index(drop=True)

    # record header (한 번만 출력)
    print("=" * 80)
    print(f"  Record id={record_id}")
    print(f"    Prompt:           {rec['prompt']}")
    print(f"    Response:         {len(rec['response'])} chars")
    print(f"    LongestMatchLen:  {e1['LongestMatchLen']}")
    print(f"    Maximal spans:    {e1['num_maximal_spans']}")
    print(f"    Top-K spans:      {e1['num_top_k_spans']}")
    print(f"    Unique spans:     {len(snippets_list)}")
    print(f"    Response tokens:  {e1['response_token_len']}")
    print("=" * 80)

    for si in span_indices:
        span_info = snippets_list[si]
        span_obj  = e1["top_k_spans"][si]
        span_text = span_info["span_text"]
        n_snip    = span_info["num_snippets"]
        n_unique  = len(set(s["doc_ix"] for s in span_info["snippets"]))

        # snippet_text 기준 dedup
        grouped = OrderedDict()
        for snip in span_info["snippets"]:
            txt = snip["snippet_text"]
            if txt not in grouped:
                grouped[txt] = []
            grouped[txt].append(snip)

        # CSV filter for this span
        mask = rec_df["span_idx"] == si
        sub_df = rec_df[mask].reset_index(drop=True)
        label_map = {int(row["doc_ix"]): row for _, row in sub_df.iterrows()}

        # span_safety_label
        span_label = sub_df.iloc[0]["span_safety_label"] if len(sub_df) > 0 else "(not yet labeled)"

        # span header
        print()
        print("-" * 60)
        print(f"  Span {si} "
              f"({len(grouped)} unique / {n_snip} total snippet(s))")
        print(f'    span_text:          "{span_text}"')
        print(f"    span_safety_label:  {span_label}")
        print("-" * 60)

        # Each unique snippet
        for i, (snip_txt, snip_group) in enumerate(grouped.items()):
            appears = len(snip_group)
            rep     = snip_group[0]
            dix     = rep["doc_ix"]
            doc_len = rep["doc_len"]

            all_dixes = [s["doc_ix"] for s in snip_group]

            doc_entry = _full_docs.get(str(dix), {})
            full_text = doc_entry.get("full_text", "—")

            ct = "(not yet labeled)"
            cs = "(not yet labeled)"
            for d in all_dixes:
                if d in label_map:
                    row = label_map[d]
                    ct  = row["context_topic"]
                    cs  = row["context_safety"]
                    break

            print()
            print(f"    Snippet {i} (appears {appears}x):")
            print(f"      doc_ix:           {dix}", end="")
            if appears > 1:
                others = [str(d) for d in all_dixes[1:]]
                print(f"  (also: {', '.join(others)})")
            else:
                print()
            print(f"      doc_len:          {doc_len} tokens")
            print(f'      snippet_text:     "{snip_txt[:150]}"')
            print(f"      context_topic:    {ct}")
            print(f"      context_safety:   {cs}")
            print(f"      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print(f'      full_doc:         "{full_text[:1500]}"')
            print()
            print(f"    {'=' * 56}")

    print()
    print("=" * 80)