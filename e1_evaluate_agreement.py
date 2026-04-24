import csv
import argparse
import sys
from collections import defaultdict

def calculate_metrics(y_true, y_pred, labels):
    """Calculate Precision, Recall, and F1 for each label using standard libs."""
    metrics = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in y_true if t == label)
        }
    return metrics

def print_report(title, y_true, y_pred, labels):
    print("\n" + "="*60)
    print(f"{title.upper()} AGREEMENT REPORT")
    print("="*60)
    
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    overall_acc = correct / total if total > 0 else 0
    
    print(f"Overall Accuracy: {overall_acc:.2%}")
    print("-" * 60)
    
    metrics = calculate_metrics(y_true, y_pred, labels)
    print(f"{'Label':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    for label in labels:
        m = metrics[label]
        print(f"{label:<25} {m['precision']:<10.2%} {m['recall']:<10.2%} {m['f1']:<10.2%} {m['support']:<8}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate labeling agreement (Standard Library Version).")
    parser.add_argument("--human", required=True, help="Path to human ground truth CSV")
    parser.add_argument("--model", required=True, help="Path to model-generated labels CSV")
    args = parser.parse_args()

    def load_csv(path):
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Key: (record_id, span_idx, doc_ix)
                try:
                    key = (int(row['record_id']), int(row['span_idx']), int(row['doc_ix']))
                    data[key] = row
                except (ValueError, KeyError):
                    continue
        return data

    try:
        data_h = load_csv(args.human)
        data_m = load_csv(args.model)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        sys.exit(1)

    # Find overlapping keys (Inner Join)
    common_keys = sorted(list(set(data_h.keys()) & set(data_m.keys())))

    if not common_keys:
        print("Error: No overlapping records found based on (record_id, span_idx, doc_ix).")
        sys.exit(1)

    print(f"Alignment successful. Evaluated pairs: {len(common_keys)}")
    print(f"(Total rows in human={len(data_h)}, model={len(data_m)})")

    y_true_span = [data_h[k]['span_safety_label'] for k in common_keys]
    y_pred_span = [data_m[k]['span_safety_label'] for k in common_keys]
    
    y_true_ctx = [data_h[k]['context_safety'] for k in common_keys]
    y_pred_ctx = [data_m[k]['context_safety'] for k in common_keys]

    # Evaluate Span Safety Label
    span_labels = ['unsafe', 'safe_but_relevant', 'trivial']
    print_report("Span Safety Label", y_true_span, y_pred_span, span_labels)

    # Evaluate Context Safety Label
    ctx_labels = ['unsafe_context', 'ambiguous_context', 'safe_context']
    print_report("Context Safety Label", y_true_ctx, y_pred_ctx, ctx_labels)

if __name__ == "__main__":
    main()
