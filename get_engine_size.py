#!/usr/bin/env python3
"""Get infini-gram engine size using engine.count(input_ids=[])."""
import logging
import os
import sys

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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not os.path.isdir(INDEX_DIR):
        logger.error("Index directory not found: %s", INDEX_DIR)
        sys.exit(1)

    if TOKENIZER_NAME is None:
        logger.info("Using byte-level index")
        eos_token_id = 0
        token_dtype = "u8"
        vocab_size = 256
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
        engine = InfiniGramEngine(
            s3_names=[],
            index_dir=INDEX_DIR,
            eos_token_id=eos_token_id,
            version=4,
            token_dtype="u16",
        )

    logger.info("Getting engine size...")
    engine_size = engine.count(input_ids=[])
    logger.info("Engine size: %s", engine_size)
    print(f"Engine size: {engine_size}")


if __name__ == "__main__":
    main()
