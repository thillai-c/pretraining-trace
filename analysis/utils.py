import os
import json
import csv
import numpy as np

def display_record(e1_result, id=None, index=None, show_e1=True, show_e2=False, indent=2):
    record = None
    record_index = None

    if id is not None:
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
        if index < 0 or index >= len(e1_result):
            print(f"\n{'='*80}")
            print(f"Error: Index {index} is out of range. Valid range: 0 to {len(e1_result) - 1}")
            print(f"{'='*80}\n")
            return
        record = e1_result[index]
        record_index = index
    else:
        record = e1_result[0]
        record_index = 0

    hb_label = record.get('hb_label')
    if hb_label == 1:
        label_str = "Compliant (hb_label=1)"
    elif hb_label == 0:
        label_str = "Non-Compliant (hb_label=0)"
    else:
        label_str = f"Unknown (hb_label={hb_label})"
    title = f"{label_str} - Record ID: {record.get('id', 'N/A')} (from {len(e1_result)} total records)"

    print(f"\n{'='*80}")
    if id is not None:
        print(f"{title}")
    else:
        print(f"{title} - Displaying index {record_index}")
    print(f"{'='*80}\n")

    record_copy = record.copy()
    record_copy.pop('e1', None)
    record_copy.pop('e2', None)
    print(f"{'─'*80}")
    print(f"Record ID: {record_copy.get('id', 'N/A')}")
    print(f"{'─'*80}")
    print(json.dumps(record_copy, indent=indent, ensure_ascii=False))
    print()

    if show_e1:
        if 'e1' not in record:
            print(f"Note: Record does not have 'e1' key\n")
        else:
            e1_data = record['e1']
            e1_filtered = {}
            for key in ['response_token_len', 'LongestMatchLen', 'VerbatimCoverage',
                        'num_maximal_spans', 'num_top_k_spans', 'span_length_distribution']:
                if key in e1_data:
                    e1_filtered[key] = e1_data[key]

            if 'ExampleSnippets' in e1_data:
                span_text_counts = {}
                for snippet in e1_data['ExampleSnippets']:
                    t = snippet.get('span_text', '')
                    span_text_counts[t] = span_text_counts.get(t, 0) + 1

                seen_texts = set()
                filtered_snippets = []
                for snippet in e1_data['ExampleSnippets']:
                    t = snippet.get('span_text', '')
                    if t not in seen_texts:
                        seen_texts.add(t)
                        sc = snippet.copy()
                        sc.pop('snippets', None)
                        sc['count'] = span_text_counts[t]
                        filtered_snippets.append(sc)
                e1_filtered['ExampleSnippets'] = filtered_snippets

            print(f"{'='*80}")
            print(f"E1 Metrics" + (f" - ID: {id}" if id is not None else ""))
            print(f"{'='*80}\n")
            print(json.dumps(e1_filtered, indent=indent, ensure_ascii=False))
            print()

    if show_e2:
        if 'e2' not in record:
            print(f"Note: Record does not have 'e2' key\n")
        else:
            e2_data = record['e2']
            e2_filtered = {}
            for key in ['E2_support_score', 'windows_tested', 'num_concepts',
                        'num_pairs_queried', 'enabling_concepts', 'metrics_by_window']:
                if key in e2_data:
                    e2_filtered[key] = e2_data[key]

            print(f"{'='*80}")
            print(f"E2 Metrics" + (f" - ID: {id}" if id is not None else ""))
            print(f"{'='*80}\n")
            print(json.dumps(e2_filtered, indent=indent, ensure_ascii=False))
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

def display_snippets(record, span_idx=None, max_snippet_chars=None):
    """Display a record's spans and snippets for labeling reference.

    If span_idx is given, only that span is shown; otherwise all spans are displayed.
    """
    unique_spans = extract_unique_spans(record)
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
    total_snippets = sum(s.get('num_snippets', 0) for s in unique_spans)
    unique_snippets = sum(len(set(sn.get('snippet_text', '') for sn in s.get('snippets', []))) for s in unique_spans)
    print(f"  Unique spans:     {len(unique_spans)}  ({unique_snippets} unique / {total_snippets} total snippets)")
    print(f"  Response tokens:  {e1.get('response_token_len', 'N/A')}")
    print("=" * 80)

    if span_idx is not None:
        if span_idx < 0 or span_idx >= len(unique_spans):
            print(f"\n[ERROR] span_idx={span_idx} out of range (0..{len(unique_spans)-1})")
            return
        display_spans = [(span_idx, unique_spans[span_idx])]
    else:
        display_spans = list(enumerate(unique_spans))

    for i, span_info in display_spans:
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
            metadata = snip.get('metadata', None)
            count = snip_text_counts[snippet_text]

            print(f"\n    Snippet {shown} (appears {count}x):")
            print(f"      doc_ix:   {doc_ix}")
            print(f"      doc_len:  {doc_len} tokens")
            if metadata:
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        pass
                meta_str = json.dumps(metadata, indent=2, ensure_ascii=False)
                indented = meta_str.replace('\n', '\n              ')
                print(f"      metadata: {indented}")
            print(f"      text:     \"{display_text}\"")

            shown += 1

    print(f"\n{'=' * 80}")
    shown_spans = [s for _, s in display_spans]
    total_snips = sum(s.get('num_snippets', 0) for s in shown_spans)
    unique_total = sum(len(set(s.get('snippet_text', '') for s in sp.get('snippets', []))) for sp in shown_spans)
    print(f"Total: {len(shown_spans)} unique spans, {unique_total} unique / {total_snips} total snippets")
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
    if not existing_dict and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        print(f"[SAFETY] Refusing to overwrite {csv_path} with empty data.")
        return

    columns = ['record_id', 'span_idx', 'doc_ix',
               'context_topic', 'context_safety', 'span_safety_label']
    tmp_path = csv_path + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for key in sorted(existing_dict.keys()):
            row = existing_dict[key]
            writer.writerow({c: row[c] for c in columns})
    os.replace(tmp_path, csv_path)

def label_record(record_id, all_records, full_docs, label_csv_path,
                 max_snippet_chars=None, max_doc_chars=1500, relabel=False):
    """Interactive labeling for one record — saves per snippet.
    
    Args:
        record_id: which record to label
        all_records: list of record dicts (loaded from JSON)
        full_docs: dict of full document texts keyed by doc_ix string
        label_csv_path: path to the CSV file for saving labels
        max_snippet_chars: truncate snippet display (None = show all)
        max_doc_chars: truncate full document display (default 1500 chars, None = show all)
        relabel: if True, re-label already labeled snippets instead of skipping them
    
    Enter 'q' at any prompt to stop. Each snippet is saved immediately.
    """
    rec = next((r for r in all_records if r['id'] == record_id), None)
    if rec is None:
        print(f"Record id={record_id} not found.")
        return
    
    unique_spans = extract_unique_spans(rec)
    existing = load_existing_labels(label_csv_path)
    
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
            save_all_labels(label_csv_path, existing)
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

import pandas as pd
from collections import OrderedDict


MODELS = {
    "olmo2-1b":           {"out_dir": "olmo2_1b"},
    "olmo2-7b":           {"out_dir": "olmo2_7b"},
    "olmo2-13b":          {"out_dir": "olmo2_13b"},
    "olmo2-32b":          {"out_dir": "olmo2_32b"},
    "olmo2-1b-instruct":  {"out_dir": "olmo2_1b_instruct"},
    "olmo2-7b-instruct":  {"out_dir": "olmo2_7b_instruct"},
    "olmo2-13b-instruct": {"out_dir": "olmo2_13b_instruct"},
    "olmo2-32b-instruct": {"out_dir": "olmo2_32b_instruct"},
}


def view_span(record_id: int, all_records, full_docs=None,
              label_csv_path=None, model=None, span_idx: int = None):
    """Display spans with labels.

    Args:
        model: If given (e.g. "olmo2-7b-instruct"), auto-resolves label_csv_path
               to results/{out_dir}/span_safety_labels_test.csv.
        full_docs: Optional dict of full document texts. Skipped when None.
        label_csv_path: Explicit CSV path. Overrides model-based resolution.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if label_csv_path is None and model is not None:
        out_dir = MODELS[model]["out_dir"]
        label_csv_path = os.path.join(project_root, "results", out_dir, "span_safety_labels_test.csv")
    elif label_csv_path is None:
        label_csv_path = "span_safety_labels.csv"

    rec = next((r for r in all_records if r["id"] == record_id), None)
    assert rec is not None, (
        f"record_id={record_id} not found. "
        f"Available: {sorted(r['id'] for r in all_records)}"
    )
    e1 = rec["e1"]
    snippets_list = e1["ExampleSnippets"]

    if span_idx is not None:
        assert 0 <= span_idx < len(snippets_list), (
            f"span_idx={span_idx} out of range (0–{len(snippets_list)-1}) "
            f"for record_id={record_id}"
        )
        span_indices = [span_idx]
    else:
        span_indices = list(range(len(snippets_list)))

    label_df = pd.read_csv(label_csv_path)
    rec_df = label_df[label_df["record_id"] == record_id].reset_index(drop=True)

    print("=" * 80)
    print(f"  Record id={record_id}")
    if model:
        print(f"    Model:            {model}")
    print(f"    Prompt:           {rec['prompt']}")
    print(f"    Response:         {len(rec['response'])} chars")
    print(f"    LongestMatchLen:  {e1['LongestMatchLen']}")
    print(f"    Maximal spans:    {e1['num_maximal_spans']}")
    print(f"    Top-K spans:      {e1['num_top_k_spans']}")
    total_snippets = sum(s.get('num_snippets', 0) for s in snippets_list)
    unique_snippets = sum(len(set(sn.get('snippet_text', '') for sn in s.get('snippets', []))) for s in snippets_list)
    print(f"    Unique spans:     {len(snippets_list)}  ({unique_snippets} unique / {total_snippets} total snippets)")
    print(f"    Response tokens:  {e1['response_token_len']}")
    print("=" * 80)

    for si in span_indices:
        span_info = snippets_list[si]
        span_text = span_info["span_text"]
        n_snip    = span_info["num_snippets"]

        grouped = OrderedDict()
        for snip in span_info["snippets"]:
            txt = snip["snippet_text"]
            if txt not in grouped:
                grouped[txt] = []
            grouped[txt].append(snip)

        sub_df = rec_df[rec_df["span_idx"] == si].reset_index(drop=True)
        label_map = {int(row["doc_ix"]): row for _, row in sub_df.iterrows()}
        span_label = sub_df.iloc[0]["span_safety_label"] if len(sub_df) > 0 else "(not yet labeled)"

        print()
        print("-" * 60)
        print(f"  Span {si} "
              f"({len(grouped)} unique / {n_snip} total snippet(s))")
        print(f'    span_text:          "{span_text}"')
        print(f"    span_safety_label:  {span_label}")
        print("-" * 60)

        for i, (snip_txt, snip_group) in enumerate(grouped.items()):
            appears = len(snip_group)
            rep     = snip_group[0]
            dix     = rep["doc_ix"]
            doc_len = rep["doc_len"]
            all_dixes = [s["doc_ix"] for s in snip_group]

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

            if full_docs is not None:
                doc_entry = full_docs.get(str(dix), {})
                full_text = doc_entry.get("full_text", "—")
                print(f"      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
                print(f'      full_doc:         "{full_text[:1500]}"')

            print()
            print(f"    {'=' * 56}")

    print()
    print("=" * 80)


# ============================================================
# Phase 4: Multi-LLM Label Comparison
# ============================================================

def compare_labels(record_id: int, all_records, model: str,
                   llm_models: list[str], span_idx: int = None,
                   human: bool = False):
    """Compare labels from multiple LLM labelers (and optionally human) for the same record.

    Args:
        record_id: which record to display
        all_records: list of record dicts
        model: OLMo model key (e.g. "olmo2-7b-instruct")
        llm_models: list of LLM model names (e.g. ["gpt-4.1-mini", "gpt-5-mini"])
        span_idx: if given, show only this span
        human: if True, also load and display human annotations
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = MODELS[model]["out_dir"]

    rec = next((r for r in all_records if r["id"] == record_id), None)
    assert rec is not None, f"record_id={record_id} not found."

    e1 = rec["e1"]
    snippets_list = e1["ExampleSnippets"]

    if span_idx is not None:
        assert 0 <= span_idx < len(snippets_list), (
            f"span_idx={span_idx} out of range (0–{len(snippets_list)-1})")
        span_indices = [span_idx]
    else:
        span_indices = list(range(len(snippets_list)))

    label_dfs = {}

    if human:
        human_csv = os.path.join(project_root, "results", out_dir,
                                 "span_safety_labels_test_human.csv")
        if os.path.isfile(human_csv):
            df_h = pd.read_csv(human_csv)
            label_dfs["human"] = df_h[df_h["record_id"] == record_id].reset_index(drop=True)
        else:
            print(f"  [WARNING] Human CSV not found: {human_csv}")

    for llm in llm_models:
        csv_path = os.path.join(project_root, "results", out_dir, llm,
                                "span_safety_labels_test.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            label_dfs[llm] = df[df["record_id"] == record_id].reset_index(drop=True)
        else:
            print(f"  [WARNING] CSV not found for {llm}: {csv_path}")

    short_names = {k: k if k == "human" else k.replace("gpt-", "")
                   for k in label_dfs}

    # Header
    labelers = ", ".join(short_names.values())
    print("=" * 80)
    print(f"  Record id={record_id}  |  Model: {model}")
    print(f"  Comparing: {labelers}")
    print(f"    Prompt:           {rec['prompt']}")
    print(f"    Response:         {len(rec['response'])} chars")
    print(f"    LongestMatchLen:  {e1['LongestMatchLen']}")
    print(f"    Unique spans:     {len(snippets_list)}")
    print("=" * 80)

    for si in span_indices:
        span_info = snippets_list[si]
        span_text = span_info["span_text"]
        n_snip = span_info["num_snippets"]

        grouped = OrderedDict()
        for snip in span_info["snippets"]:
            txt = snip["snippet_text"]
            if txt not in grouped:
                grouped[txt] = []
            grouped[txt].append(snip)

        # Collect span_safety_label from each LLM
        span_labels = {}
        for llm, df in label_dfs.items():
            sub = df[df["span_idx"] == si]
            if len(sub) > 0:
                span_labels[llm] = sub.iloc[0]["span_safety_label"]
            else:
                span_labels[llm] = "—"

        all_same = len(set(span_labels.values()) - {"—"}) <= 1

        print()
        print("-" * 60)
        print(f"  Span {si} "
              f"({len(grouped)} unique / {n_snip} total snippet(s))")
        print(f'    span_text:          "{span_text}"')
        print(f"    span_safety_label:  ", end="")
        if all_same and span_labels:
            first_val = next(v for v in span_labels.values() if v != "—")
            print(f"{first_val}  (all agree)")
        else:
            parts = [f"{short_names[llm]}={v}" for llm, v in span_labels.items()]
            print(" | ".join(parts) + "  *** DISAGREE ***")
        print("-" * 60)

        # Show each unique snippet
        for i, (snip_txt, snip_group) in enumerate(grouped.items()):
            appears = len(snip_group)
            rep = snip_group[0]
            dix = rep["doc_ix"]
            doc_len = rep["doc_len"]
            all_dixes = [s["doc_ix"] for s in snip_group]

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

            # Labels from each LLM for this snippet
            for llm, df in label_dfs.items():
                row = df[(df["span_idx"] == si) & (df["doc_ix"] == dix)]
                if len(row) > 0:
                    r = row.iloc[0]
                    ct = r.get("context_topic", "—")
                    cs = r.get("context_safety", "—")
                else:
                    ct, cs = "—", "—"
                print(f"      [{short_names[llm]:>10s}]  "
                      f"ctx_safety={cs:<20s}  topic={ct}")

            print()
            print(f"    {'=' * 56}")

    print()
    print("=" * 80)