import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# train_data = pd.read_csv("./HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv")
# print(train_data.columns)

# print(train_data[train_data['FunctionalCategory']=="copyright"])
# print("Train data:", len(train_data[train_data['FunctionalCategory']=="copyright"]))

# ============================================================================
# Part 1: Display original results (harmbench_copyright.json)
# ============================================================================
# Load and display JSON results
json_path = "./data/gpt_j_6b/harmbench_standard.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"HarmBench Copyright Results - Total: {len(data)} records (showing first 1)")
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
    print(f"❌ JSON file not found: {json_path}")