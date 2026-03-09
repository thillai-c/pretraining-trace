import os
import json
import pandas as pd
from dotenv import load_dotenv

# csv_path = "./HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
# df = pd.read_csv(csv_path)

# print(df.columns)

# print("\n Standard category:")
# print(df[df['FunctionalCategory'] == 'standard'])
# print(df[df['FunctionalCategory'] == 'standard'].shape) # (200, 6)

# print("\n Contextual category:")
# print(df[df['FunctionalCategory'] == 'contextual'])
# print(df[df['FunctionalCategory'] == 'contextual'].shape) # (100, 6)

# print("\n Copyright category:")
# print(df[df['FunctionalCategory'] == 'copyright'])
# print(df[df['FunctionalCategory'] == 'copyright'].shape) # (100, 6)


# ============================================================================
# Part 2: Display labeled results (harmbench_standard_labeled.json)
# ============================================================================
# Load and display labeled JSON results
labeled_json_path = "./results/gpt_j_6b/harmbench_standard_labeled.json"
if os.path.exists(labeled_json_path):
    with open(labeled_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"HarmBench Copyright Labeled Results - Total: {len(data)} records")
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
    print(f"❌ Labeled JSON file not found: {labeled_json_path}")