import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# train_data = pd.read_csv("./HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
# print(train_data.columns)

# print(train_data[train_data['FunctionalCategory']=="copyright"])
# print("Train data:", len(train_data[train_data['FunctionalCategory']=="copyright"]))

# ============================================================================
# Args: choose HarmBench category once (standard/contextual/copyright)
# ============================================================================
parser = argparse.ArgumentParser(description="Inspect HarmBench outputs and labeled results")
parser.add_argument(
    "--config",
    choices=["standard", "contextual", "copyright", "all"],
    default="contextual",
    help="HarmBench category to inspect (default: contextual). Use 'all' to summarize all categories.",
)
args = parser.parse_args()
CONFIG = args.config

# Model directory used for the single-record preview in Part 1/2.
# (Part 3 always aggregates across the fixed 8 model dirs below.)
PREVIEW_MODEL_DIR = "olmo2_7b_instruct"

CONFIGS = ["standard", "contextual", "copyright"] if CONFIG == "all" else [CONFIG]


# ============================================================================
# Part 1: Display original results (harmbench_copyright.json)
# ============================================================================
# Load and display JSON results
if CONFIG == "all":
    print(f"\n{'='*80}")
    print("Preview sections (Part 1/2) are skipped when --config=all.")
    print("Use --config standard|contextual|copyright to preview individual records.")
    print(f"{'='*80}\n")
    json_path = None
else:
    json_path = f"./data/{PREVIEW_MODEL_DIR}/harmbench_{CONFIG}.json"

if json_path and os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"HarmBench {CONFIG} Results - Total: {len(data)} records (showing first 1)")
    print(f"{'='*80}\n")
    
    for i, record in enumerate(data[:1], 1):
        print(f"{'─'*80}")
        print(f"Record {i} (ID: {record.get('id', 'N/A')})")
        print(f"{'─'*80}")
        print(f"Config: {record.get('config', 'N/A')}")
        print(f"Model: {record.get('model', 'N/A')}")
        print(f"Finish Reason: {record.get('finish_reason', 'N/A')}")
        
        print(f"\nPrompt:")
        prompt = record.get('prompt', 'N/A')
        print(f"   {prompt}")
        
        print(f"\nResponse ({len(record.get('response', ''))} chars):")
        response = record.get('response', 'N/A')
        if response != 'N/A':
            # Truncate long responses for readability
            if len(response) > 300:
                # Show first 300 chars and last 100 chars
                print(f"   {response[:300]}...")
                print(f"   ... (truncated {len(response) - 400} chars) ...")
                print(f"   ...{response[-100:]}")
            else:
                # Show full response with proper indentation
                for line in response.split('\n'):
                    print(f"   {line}")
        else:
            print(f"   {response}")
        
        if 'metadata' in record and record['metadata']:
            print(f"\nMetadata:")
            for key, value in record['metadata'].items():
                if value:  # Only show non-empty values
                    print(f"   {key}: {value}")
        
        if 'error' in record:
            print(f"\n❌ Error: {record['error']}")
        
        print()
else:
    if json_path:
        print(f"❌ JSON file not found: {json_path}")


# ============================================================================
# Part 2: Display labeled results (harmbench_standard_labeled.json)
# ============================================================================
# Load and display labeled JSON results
if CONFIG == "all":
    labeled_json_path = None
else:
    labeled_json_path = f"./data/{PREVIEW_MODEL_DIR}/harmbench_{CONFIG}_labeled.json"

if labeled_json_path and os.path.exists(labeled_json_path):
    with open(labeled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"HarmBench {CONFIG} Labeled Results - Total: {len(data)} records")
    print(f"{'='*80}\n")
    
    for i, record in enumerate(data, 1):
        print(f"{'─'*80}")
        print(f"Record {i} (ID: {record.get('id', 'N/A')})")
        print(f"{'─'*80}")
        print(f"Config: {record.get('config', 'N/A')}")
        print(f"Model: {record.get('model', 'N/A')}")
        print(f"Finish Reason: {record.get('finish_reason', 'N/A')}")
        
        # Show hb_label if available
        if 'hb_label' in record:
            label = record['hb_label']
            label_str = "Compliant (1)" if label == 1 else "Not Compliant (0)"
            print(f"HB Label: {label} - {label_str}")
        
        print(f"\nPrompt:")
        prompt = record.get('prompt', 'N/A')
        print(f"   {prompt}")
        
        print(f"\nResponse ({len(record.get('response', ''))} chars):")
        response = record.get('response', 'N/A')
        if response != 'N/A':
            # Truncate long responses for readability
            if len(response) > 200:
                # Show first 200 chars and last 100 chars
                print(f"   {response[:200]}...")
                print(f"   ... (truncated {len(response) - 300} chars) ...")
                print(f"   ...{response[-100:]}")
            else:
                # Show full response with proper indentation
                for line in response.split('\n'):
                    print(f"   {line}")
        else:
            print(f"   {response}")
        
        if 'metadata' in record and record['metadata']:
            print(f"\nMetadata:")
            for key, value in record['metadata'].items():
                if value:  # Only show non-empty values
                    print(f"   {key}: {value}")
        
        if 'error' in record:
            print(f"\n❌ Error: {record['error']}")
        
        print()
        if i >= 5:
            break
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    
    total = len(data)
    compliant = sum(1 for r in data if r.get('hb_label') == 1)
    not_compliant = sum(1 for r in data if r.get('hb_label') == 0)
    with_errors = sum(1 for r in data if 'error' in r)
    no_label = sum(1 for r in data if 'hb_label' not in r and 'error' not in r)
    
    print(f"Total records: {total}")
    print(f"Compliant (hb_label=1): {compliant} ({compliant/total*100:.1f}%)")
    print(f"Not Compliant (hb_label=0): {not_compliant} ({not_compliant/total*100:.1f}%)")
    if with_errors > 0:
        print(f"Records with errors: {with_errors}")
    if no_label > 0:
        print(f"Records without label: {no_label}")
    print(f"{'='*80}\n")
else:
    if labeled_json_path:
        print(f"❌ Labeled JSON file not found: {labeled_json_path}")


# ============================================================================
# Part 3: Aggregate stats across 8 OLMo 2 models (per CONFIG)
# ============================================================================
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_records(records):
    total = len(records)
    with_label = sum(1 for r in records if "hb_label" in r)
    compliant = sum(1 for r in records if r.get("hb_label") == 1)
    refused = sum(1 for r in records if r.get("hb_label") == 0)
    with_errors = sum(1 for r in records if "error" in r)
    no_label = sum(1 for r in records if "hb_label" not in r and "error" not in r)

    asr = (compliant / with_label * 100.0) if with_label > 0 else float("nan")
    refusal_rate = (refused / with_label * 100.0) if with_label > 0 else float("nan")

    return {
        "total": total,
        "with_label": with_label,
        "compliant": compliant,
        "refused": refused,
        "errors": with_errors,
        "no_label": no_label,
        "asr_pct": asr,
        "refusal_pct": refusal_rate,
    }


MODEL_DIRS = [
    "olmo2_1b",
    "olmo2_7b",
    "olmo2_13b",
    "olmo2_32b",
    "olmo2_1b_instruct",
    "olmo2_7b_instruct",
    "olmo2_13b_instruct",
    "olmo2_32b_instruct",
]

overall_rows = []

for cfg in CONFIGS:
    rows = []
    missing = []
    for model_dir in MODEL_DIRS:
        labeled_path = os.path.join("data", model_dir, f"harmbench_{cfg}_labeled.json")
        if not os.path.exists(labeled_path):
            missing.append(labeled_path)
            continue

        records = _load_json(labeled_path)
        stats = _summarize_records(records)
        stats["model_dir"] = model_dir
        stats["path"] = labeled_path
        rows.append(stats)

    print(f"\n{'='*80}")
    print(f"HarmBench {cfg} labeled summary across models")
    print(f"{'='*80}")

    if missing:
        print("Missing labeled files:")
        for p in missing:
            print(f"  - {p}")
        print()

    if rows:
        df = pd.DataFrame(rows)
        df["model_dir"] = pd.Categorical(df["model_dir"], categories=MODEL_DIRS, ordered=True)
        df = df.sort_values("model_dir")
        display_cols = [
            "model_dir",
            "total",
            "with_label",
            "compliant",
            "refused",
            "errors",
            "no_label",
            "asr_pct",
            "refusal_pct",
        ]
        df_disp = df[display_cols].copy()
        df_disp["asr_pct"] = df_disp["asr_pct"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")
        df_disp["refusal_pct"] = df_disp["refusal_pct"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")

        # Print per-model summary
        print(df_disp.to_string(index=False))

        # Print overall summary (micro-average across all loaded models)
        all_records = []
        for r in rows:
            all_records.extend(_load_json(r["path"]))
        overall = _summarize_records(all_records)
        overall["config"] = cfg
        overall_rows.append(overall)

        print(f"\n{'-'*80}")
        print("Overall (micro-average across loaded models)")
        print(f"  Total records         : {overall['total']}")
        print(f"  Records with hb_label : {overall['with_label']}")
        print(f"    Compliant (label=1) : {overall['compliant']} ({overall['asr_pct']:.1f}%)")
        print(f"    Refused   (label=0) : {overall['refused']} ({overall['refusal_pct']:.1f}%)")
        print(f"  Records with error    : {overall['errors']}")
        if overall["no_label"] > 0:
            print(f"  Records without label : {overall['no_label']}")
    else:
        print("No labeled files found to summarize.")

if len(CONFIGS) > 1 and overall_rows:
    print(f"\n{'='*80}")
    print("Overall summary by config (micro-average)")
    print(f"{'='*80}")
    df_overall = pd.DataFrame(overall_rows)
    df_overall = df_overall[
        [
            "config",
            "total",
            "with_label",
            "compliant",
            "refused",
            "errors",
            "no_label",
            "asr_pct",
            "refusal_pct",
        ]
    ].sort_values("config")
    df_overall["asr_pct"] = df_overall["asr_pct"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")
    df_overall["refusal_pct"] = df_overall["refusal_pct"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")
    print(df_overall.to_string(index=False))