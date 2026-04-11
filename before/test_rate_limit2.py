import requests
import time

API_URL = "https://api.infini-gram.io/"
INDEX = "v4_olmo-mix-1124_llama"

test_queries = [
    {"index": INDEX, "query_type": "find", "query_ids": list(range(100, 600))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 350))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 225))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 163))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 132))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 116))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 108))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 104))},
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 102))},
    {"index": INDEX, "query_type": "count", "query_ids": [100]},
]

for delay in [0.6, 0.7, 0.8]:
    success = 0
    fail = 0
    start = time.time()

    for pos in range(50):  # 50 positions = ~500 calls
        for q in test_queries:
            resp = requests.post(API_URL, json=q, timeout=30)
            if resp.status_code == 200:
                success += 1
            else:
                fail += 1
                break
            time.sleep(delay)
        if fail > 0:
            break

    elapsed = time.time() - start
    total = success + fail
    print(f"delay={delay:.1f}s | {total} calls in {elapsed:.1f}s | "
          f"success={success} fail={fail} | "
          f"positions={pos}{'+' if fail == 0 else ''}")

    if fail > 0:
        print(f"  → FAILED at position {pos}")

    print("  (waiting 120s...)")
    time.sleep(120)