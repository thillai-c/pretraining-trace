#!/usr/bin/env python3
"""Run infini-gram query example: count and find phrase in index."""
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

# Load .env and add infini-gram pkg to path BEFORE importing infini_gram
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

PKG_DIR = os.path.join(SCRIPT_DIR, "infini-gram", "pkg")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from transformers import AutoTokenizer
from infini_gram.engine import InfiniGramEngine

# Config (same as --save_dir when you ran indexing.py)
INDEX_DIR = os.path.join(SCRIPT_DIR, "index")
# Tokenizer must match the one used when building the index. Set to None for byte-level index.
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"
QUERY_STR = "natural language processing"


def setup_logger():
    os.makedirs("logs", exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_filepath = f"logs/run_query_example_{timestamp}.log"
    log_filepath = "logs/run_query_example.log"

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


def main():
    logger = setup_logger()
    
    logger.info("=== Job started at %s ===", datetime.now().strftime("%a %b %d %I:%M:%S %p %Z %Y"))
    start_time = time.time()

    if not os.path.isdir(INDEX_DIR):
        logger.error("Index directory not found: %s", INDEX_DIR)
        logger.error("Point INDEX_DIR to the directory that contains tokenized.*, offset.*, table.*")
        sys.exit(1)

    if TOKENIZER_NAME is None:
        logger.info("Using byte-level index (tokenizer=None during indexing)")
        eos_token_id = 0
        token_dtype = "u8"
        vocab_size = 256
        input_ids = list(QUERY_STR.encode("utf-8"))
        logger.info("Loading engine from index: %s", INDEX_DIR)
        engine = InfiniGramEngine(
            s3_names=[],
            index_dir=INDEX_DIR,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            version=4,
            token_dtype=token_dtype,
        )
    else:
        logger.info("Loading tokenizer: %s", TOKENIZER_NAME)
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_NAME,
            token=os.environ.get("HF_TOKEN"),
            use_fast=False,
            add_bos_token=False,
            add_eos_token=False,
        )
        eos_token_id = tokenizer.eos_token_id
        logger.info("Loading engine from index: %s", INDEX_DIR)
        engine = InfiniGramEngine(
            s3_names=[],
            index_dir=INDEX_DIR,
            eos_token_id=eos_token_id,
            version=4,
            token_dtype="u16",
        )
        input_ids = tokenizer.encode(QUERY_STR)

    logger.info("Query: '%s'", QUERY_STR)
    logger.info("Token IDs: %s", input_ids)

    result = engine.count(input_ids=input_ids)
    logger.info("Count: %s", result)

    find_result = engine.find(input_ids=input_ids)
    if "error" in find_result:
        logger.error("Find error: %s", find_result["error"])
        return
    logger.info("Find (segment_by_shard): %s", find_result)

    if find_result.get("segment_by_shard") and find_result["segment_by_shard"][0][0] < find_result["segment_by_shard"][0][1]:
        s, (start_rank, end_rank) = 0, find_result["segment_by_shard"][0]
        rank = start_rank
        doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=20)
        if "error" not in doc:
            logger.info("First matching doc (shard %s, rank %s): %s", s, rank, doc)
        else:
            logger.warning("Doc retrieval: %s", doc.get("error", doc))
    
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
