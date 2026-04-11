import requests
import time

API_URL = "https://api.infini-gram.io/"
INDEX = "v4_olmo-mix-1124_llama"

# Simulate real E1 pattern: find(long) + count(various lengths)
test_queries = [
    {"index": INDEX, "query_type": "find", "query_ids": list(range(100, 600))},   # find, 500 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 350))},   # count, 250 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 225))},   # count, 125 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 163))},   # count, 63 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 132))},   # count, 32 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 116))},   # count, 16 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 108))},   # count, 8 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 104))},   # count, 4 tokens
    {"index": INDEX, "query_type": "count", "query_ids": list(range(100, 102))},   # count, 2 tokens
    {"index": INDEX, "query_type": "count", "query_ids": [100]},                   # count, 1 token
]
# 1 position = ~10 calls (1 find + ~9 binary search counts)

for delay in [0.0, 0.1, 0.2, 0.3, 0.5]:
    success = 0
    fail = 0
    start = time.time()

    # Simulate 20 positions (= ~200 API calls)
    for pos in range(20):
        for q in test_queries:
            resp = requests.post(API_URL, json=q, timeout=30)
            if resp.status_code == 200:
                success += 1
            else:
                fail += 1
                break  # stop this delay test on first failure
            time.sleep(delay)
        if fail > 0:
            break

    elapsed = time.time() - start
    total = success + fail
    print(f"delay={delay:.1f}s | {total} calls in {elapsed:.1f}s | "
          f"success={success} fail={fail} | "
          f"{success/elapsed:.1f} req/s")

    if fail > 0:
        print(f"  → FAILED at position {pos}, resetting...")

    # Wait between tests to reset rate limit
    print("  (waiting 120s before next test...)")
    time.sleep(120)