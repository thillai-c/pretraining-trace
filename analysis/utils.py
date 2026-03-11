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