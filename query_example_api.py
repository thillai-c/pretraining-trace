#!/usr/bin/env python3
"""Run infini-gram query example using API: count and find phrase in index."""
import logging
import os
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load .env
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

# API configuration
API_ENDPOINT = "https://api.infini-gram.io/"
INDEX_NAME = "v4_olmo-mix-1124_llama"
QUERY_STR = "natural language processing"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filepath = f"logs/run_query_example_api_{timestamp}.log"
    # log_filepath = "logs/run_query_example_api.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filepath, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to file: %s", log_filepath)
    return logger


def api_request(payload, logger):
    """Make API request with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_ENDPOINT, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                logger.error("API error: %s", result["error"])
                return None
            
            return result
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning("Request failed (attempt %d/%d): %s. Retrying...", 
                              attempt + 1, MAX_RETRIES, e)
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error("Request failed after %d attempts: %s", MAX_RETRIES, e)
                return None
    return None


def main():
    logger = setup_logger()
    
    logger.info("=== Job started at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    start_time = time.time()
    
    logger.info("Using infini-gram API endpoint: %s", API_ENDPOINT)
    logger.info("Index: %s", INDEX_NAME)
    logger.info("Query: '%s'", QUERY_STR)
    
    # Step 1: Count query
    logger.info("=== Step 1: Count query ===")
    count_payload = {
        "index": INDEX_NAME,
        "query_type": "count",
        "query": QUERY_STR,
    }
    
    count_result = api_request(count_payload, logger)
    if count_result is None:
        logger.error("Failed to get count result")
        return
    
    logger.info("Count result: %s", count_result)
    logger.info("Count: %s (approx: %s)", count_result.get("count"), count_result.get("approx"))
    logger.info("Token IDs: %s", count_result.get("token_ids"))
    logger.info("Tokens: %s", count_result.get("tokens"))
    logger.info("Latency: %s ms", count_result.get("latency"))
    
    # Get token IDs for subsequent queries
    token_ids = count_result.get("token_ids", [])
    if not token_ids:
        logger.error("No token IDs returned from count query")
        return
    
    # Step 2: Find query (simple query)
    logger.info("=== Step 2: Find query ===")
    find_payload = {
        "index": INDEX_NAME,
        "query_type": "find",
        "query": QUERY_STR,
    }
    
    find_result = api_request(find_payload, logger)
    if find_result is None:
        logger.error("Failed to get find result")
        return
    
    logger.info("Find result: %s", find_result)
    logger.info("Count: %s", find_result.get("cnt"))
    logger.info("Segment by shard: %s", find_result.get("segment_by_shard"))
    
    # Step 3: Get document by rank
    segment_by_shard = find_result.get("segment_by_shard")
    if segment_by_shard and len(segment_by_shard) > 0:
        if len(segment_by_shard[0]) >= 2:
            start_rank, end_rank = segment_by_shard[0][0], segment_by_shard[0][1]
            if start_rank < end_rank:
                logger.info("=== Step 3: Get document by rank ===")
                s = 0  # shard index
                rank = start_rank
                
                get_doc_payload = {
                    "index": INDEX_NAME,
                    "query_type": "get_doc_by_rank",
                    "query": QUERY_STR,
                    "s": s,
                    "rank": rank,
                    "max_disp_len": 20,
                }
                
                doc_result = api_request(get_doc_payload, logger)
                if doc_result is None:
                    logger.warning("Failed to get document by rank")
                else:
                    logger.info("Document (shard %s, rank %s): %s", s, rank, doc_result)
                    logger.info("Doc index: %s", doc_result.get("doc_ix"))
                    logger.info("Doc length: %s", doc_result.get("doc_len"))
                    logger.info("Display length: %s", doc_result.get("disp_len"))
                    logger.info("Token IDs: %s", doc_result.get("token_ids"))
                    logger.info("Spans: %s", doc_result.get("spans"))
            else:
                logger.info("No matching documents found (start_rank >= end_rank)")
        else:
            logger.warning("Invalid segment_by_shard format")
    else:
        logger.info("No matching documents found")
    
    end_time = time.time()
    elapsed_float = end_time - start_time
    elapsed = int(elapsed_float)
    days = elapsed // 86400
    hours = (elapsed % 86400) // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    
    logger.info("=== Job finished at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    logger.info("=== Elapsed time: %d days %02d:%02d:%02d (total %.3f seconds) ===", 
                days, hours, minutes, seconds, elapsed_float)


if __name__ == "__main__":
    main()