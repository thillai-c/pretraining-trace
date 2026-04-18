#!/usr/bin/env python3
"""Shared utilities for the E2 pipeline (extraction + ranking).

This module contains code that is used by BOTH:
    - e2_extract_concepts.py  (Stage 1: concept extraction from responses)
    - e2_rank_concepts.py     (Stage 2: LLM-based ranking of extracted concepts)

Anything that is specific to a single stage (e.g., the Stage 1 system prompt,
Stage 1 sanity flags, Stage 2 system prompt, Stage 2 output schema) lives in
the corresponding stage file, not here.

Contents:
    - MODELS / REASONING_MODELS / DEFAULT_EXTRACTION_MODEL: model registry
    - TRAINING_PHASES_BASE / TRAINING_PHASES_INSTRUCT /
      TRAINING_PHASE_ALL / training_phases_when_all /
      model_results_root / label_llm_dirname / label_run_root:
      phase folders under results/{out_dir}/ and ``--training-phase all`` expansion
    - _is_reasoning_model / get_model_params: OpenAI param selection
    - setup_logger: per-stage logger factory
    - compute_rep_ratio: word-level 4-gram repetition ratio
    - load_e1_results / filter_compliant: E1 input loading
    - get_semantic_category: safe metadata accessor
    - parse_llm_json: generic JSON extraction from LLM output
    - save_output_json: standard JSON dump
"""

import json
import logging
import os
import re
import sys


# ============================================================
# Model registry
# ============================================================

MODEL_CONFIGS = {
    # GPT-J (original baseline)
    "gpt-j": {
        "hf_id": "EleutherAI/gpt-j-6B",
        "max_ctx": 2048,
        "model_type": "base",
        "out_dir_name": "gpt_j_6b",
    },

    # OLMo 2 Base
    "olmo2-1b": {
        "hf_id": "allenai/OLMo-2-0425-1B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_1b",
    },
    "olmo2-7b": {
        "hf_id": "allenai/OLMo-2-1124-7B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_7b",
    },
    "olmo2-13b": {
        "hf_id": "allenai/OLMo-2-1124-13B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_13b",
    },
    "olmo2-32b": {
        "hf_id": "allenai/OLMo-2-0325-32B",
        "max_ctx": 4096,
        "model_type": "base",
        "out_dir_name": "olmo2_32b",
    },

    # OLMo 2 Instruct
    "olmo2-1b-instruct": {
        "hf_id": "allenai/OLMo-2-0425-1B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_1b_instruct",
    },
    "olmo2-7b-instruct": {
        "hf_id": "allenai/OLMo-2-1124-7B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_7b_instruct",
    },
    "olmo2-13b-instruct": {
        "hf_id": "allenai/OLMo-2-1124-13B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_13b_instruct",
    },
    "olmo2-32b-instruct": {
        "hf_id": "allenai/OLMo-2-0325-32B-Instruct",
        "max_ctx": 4096,
        "model_type": "instruct",
        "out_dir_name": "olmo2_32b_instruct",
    },
}

# E2 pipeline expects MODELS[model_key]["out_dir"].
MODELS = {
    k: {"out_dir": v["out_dir_name"]}
    for k, v in MODEL_CONFIGS.items()
    if k.startswith("olmo2-")
}

# When tools use ``--training-phase all``: base OLMo checkpoints only have
# pretraining + mid_training; *-instruct models also have post_training.
TRAINING_PHASES_BASE = ("pretraining", "mid_training")
TRAINING_PHASES_INSTRUCT = ("pretraining", "mid_training", "post_training")
TRAINING_PHASE_ALL = "all"


def training_phases_when_all(model_key: str) -> tuple[str, ...]:
    """Phases to iterate for ``--training-phase all`` (E2, E1 auto-label, co-occurrence)."""
    if model_key.endswith("-instruct"):
        return TRAINING_PHASES_INSTRUCT
    return TRAINING_PHASES_BASE


def model_results_root(model_key: str, training_phase: str) -> str:
    """Return ``results/{out_dir}/{training_phase}/``."""
    out_dir = MODELS[model_key]["out_dir"]
    return os.path.join("results", out_dir, training_phase)


def label_llm_dirname(llm_model: str) -> str:
    """Filesystem-safe directory name for a labeling model id."""
    s = llm_model.strip().lower().replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^a-z0-9._+\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def label_run_root(model_key: str, training_phase: str, llm_model: str) -> str:
    """Root directory for one (training_phase, label LLM) experiment."""
    return os.path.join(
        model_results_root(model_key, training_phase),
        label_llm_dirname(llm_model),
    )


# Default OpenAI model used for both extraction (Stage 1) and ranking (Stage 2);
# overridable at the CLI via --extraction_model / --rank_model.
DEFAULT_EXTRACTION_MODEL = "gpt-5-mini"

# Reasoning models use `reasoning_effort` + `max_completion_tokens`
# instead of `temperature` + `max_tokens`.
REASONING_MODELS = {
    "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gpt-5.4-nano", "gpt-5.4-mini",
    "o1", "o3", "o4-mini",
}


# ============================================================
# Model parameter helpers
# ============================================================

def _is_reasoning_model(model_name: str) -> bool:
    """Check whether the given OpenAI model is a reasoning model
    (uses reasoning_effort instead of temperature)."""
    return any(model_name.startswith(prefix) for prefix in REASONING_MODELS)


def get_model_params(model_name: str) -> dict:
    """Return model-specific API parameters based on the chosen model.

    Reasoning models get `reasoning_effort=medium` and a large
    `max_completion_tokens`. Standard models get `temperature=0.0` for
    deterministic output and a tighter `max_tokens`.
    """
    if _is_reasoning_model(model_name):
        return {
            "reasoning_effort": "medium",
            "max_completion_tokens": 65000,
        }
    return {
        "temperature": 0.0,
        "max_tokens": 16000,
    }


# ============================================================
# Logger setup
# ============================================================

def setup_logger(
    model_key: str,
    stage_name: str,
    training_phase: str | None = None,
    *,
    config: str | None = None,
):
    """Create a logger with a model-based log directory.

    Args:
        model_key: a key from MODELS (e.g., "olmo2-7b-instruct")
        stage_name: short name used for both the log file and the logger,
                    e.g., "e2_extract_concepts" or "e2_rank_concepts".
        training_phase: if set, log under ``logs/{out_dir}/{training_phase}/``.
        config: if set (e.g. HarmBench config), log file is
                ``{stage_name}_{config}.log`` so runs with different configs do not
                overwrite the same file.

    Log file path: ``logs/{out_dir}/{stage_name}.log``,
    ``logs/{out_dir}/{stage_name}_{config}.log``,
    or ``logs/{out_dir}/{training_phase}/{stage_name}.log`` (and with ``_{config}``
    before ``.log`` when ``config`` is set).
    """
    out_dir = MODELS[model_key]["out_dir"]
    if training_phase:
        log_dir = os.path.join("logs", out_dir, training_phase)
    else:
        log_dir = os.path.join("logs", out_dir)
    os.makedirs(log_dir, exist_ok=True)
    base = f"{stage_name}_{config}" if config else stage_name
    log_path = os.path.join(log_dir, f"{base}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(stage_name)
    logger.info("Logging to %s", log_path)
    return logger


# ============================================================
# Response text utilities
# ============================================================

def compute_rep_ratio(response_text: str, n: int = 4) -> float:
    """Compute word-level n-gram repetition ratio (seq-rep-4).

    Returns 1.0 - (unique n-grams / total n-grams). A value of 0.0 means
    no repetition; higher values mean more repetition. Used downstream
    for degenerate-response detection.
    """
    words = response_text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


# ============================================================
# E1 input loading
# ============================================================

def load_e1_results(model_key: str, input_path: str = None,
                    training_phase: str = None, config: str = "standard"):
    """Load E1 verbatim trace JSON for ``model_key``.

    Default path: ``results/{out_dir}/[{training_phase}/]e1_verbatim_{config}.json``.
    """
    if input_path is None:
        out_dir = MODELS[model_key]["out_dir"]
        name = f"e1_verbatim_{config}.json"
        if training_phase:
            input_path = os.path.join(
                "results", out_dir, training_phase, name
            )
        else:
            input_path = os.path.join(
                "results", out_dir, name
            )
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_compliant(records):
    """Filter to HarmBench-compliant records (``hb_label == 1``).

    Returns a list of ``(record, rep_ratio)`` tuples, where ``rep_ratio``
    is the word-level 4-gram repetition ratio of the response.
    """
    filtered = []
    for r in records:
        if r.get("hb_label") != 1:
            continue
        rep = compute_rep_ratio(r["response"])
        filtered.append((r, rep))
    return filtered


def get_semantic_category(record) -> str:
    """Extract ``SemanticCategory`` from a record's metadata field.
    Returns an empty string if missing."""
    meta = record.get("metadata", {}) or {}
    return meta.get("SemanticCategory", "") or ""


# ============================================================
# Generic LLM JSON parsing
# ============================================================

def parse_llm_json(response_text: str, logger) -> dict:
    """Parse a JSON object from the LLM response text.

    Handles:
      - Leading/trailing markdown fences (```json ... ```)
      - Control characters that break JSON parsing
      - Logs the raw response on failure for debugging

    Returns the parsed dict on success, or None on failure.
    NOTE: This function does NOT validate the schema — each stage should
    do its own schema validation on top of the raw parse.
    """
    text = response_text.strip()

    # Strip markdown fences if present (e.g., ```json ... ```)
    if text.startswith("```"):
        text_lines = text.split("\n")
        if text_lines[0].startswith("```"):
            text_lines = text_lines[1:]
        if text_lines and text_lines[-1].strip() == "```":
            text_lines = text_lines[:-1]
        text = "\n".join(text_lines)

    # Clean control characters that break JSON parsing
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        logger.error("Raw response (first 500 chars): %s", text[:500])
        return None

    if not isinstance(parsed, dict):
        logger.error("Expected JSON object, got %s", type(parsed).__name__)
        return None

    return parsed


# ============================================================
# Output JSON save
# ============================================================

def save_output_json(rows: list, output_path: str):
    """Save a list of records as a pretty-printed JSON file.

    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
