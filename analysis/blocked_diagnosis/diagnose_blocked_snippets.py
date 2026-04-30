#!/usr/bin/env python3
"""Diagnose infini-gram API ``blocked``-doc behavior in existing E1 Phase 2
results, then probe a small sample of suspect spans to characterize what
triggers blocking.

Tier 1 plan (zero-API analysis + ≤300 API calls). Reads existing results in
``results/olmo2_*/pretraining/e1_verbatim_standard.json`` for all 8 OLMo 2
variants, computes a lower-bound "blocked rate" from saved ``num_snippets``,
then for the 20 worst-undershoot spans:

  - re-runs ``engine.find(span_ids)`` to get the true corpus count
  - re-fetches up to 5 ranks NOT present in stored snippets via
    ``engine.get_doc_by_rank()`` and saves the FULL raw API response

Outputs (under ``analysis/blocked_diagnosis/``):
  - ``analysis.json``  — per-model + aggregate stats
  - ``probe_raw_responses.json`` — raw API payloads for the 20 probe spans
  - ``report.md`` — human-readable summary with the four required deliverables

Usage:
    # Zero-API analysis only
    python analysis/diagnose_blocked_snippets.py --analyze-only

    # Default: analysis + probe 20 suspect spans
    python analysis/diagnose_blocked_snippets.py

    # Custom probe size
    python analysis/diagnose_blocked_snippets.py --probe 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Resolve repo root (analysis/ is one level deep).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


MODEL_KEYS = [
    "olmo2_1b",
    "olmo2_7b",
    "olmo2_13b",
    "olmo2_32b",
    "olmo2_1b_instruct",
    "olmo2_7b_instruct",
    "olmo2_13b_instruct",
    "olmo2_32b_instruct",
]

MAX_DOCS_PER_SPAN = 10  # The default in retrieve_snippets_for_span.
DEFAULT_MAX_DISP_LEN = 80
DEFAULT_PROBE_PER_SPAN = 5  # how many ranks to probe per probe-span
DEFAULT_PROBE_TOTAL = 20    # how many spans to probe overall


# ============================================================
# Logging
# ============================================================
def setup_logger() -> logging.Logger:
    log_dir = os.path.join(REPO_ROOT, "analysis", "blocked_diagnosis")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "diagnose_blocked_snippets.log")

    logger = logging.getLogger("diagnose_blocked_snippets")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("Logging to %s", log_path)
    return logger


# ============================================================
# Phase 1 — zero-API analysis
# ============================================================
def model_results_path(model_dir: str) -> str:
    return os.path.join(
        REPO_ROOT, "results", model_dir, "pretraining",
        "e1_verbatim_standard.json",
    )


def analyze_one_model(model_dir: str, logger) -> dict:
    """Aggregate Phase-2 stats from one model's standard-config JSON."""
    path = model_results_path(model_dir)
    if not os.path.isfile(path):
        logger.warning("Missing: %s — skipping.", path)
        return {"model": model_dir, "missing": True}

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total_records = len(records)
    records_with_phase2 = 0
    records_all_undershoot = 0
    records_partial_undershoot = 0
    records_no_undershoot = 0

    total_spans = 0
    undershoot_spans = 0
    zero_snippet_spans = 0
    num_snippets_hist = Counter()  # 0..MAX_DOCS_PER_SPAN

    affected_record_ids = []  # records with ≥1 undershoot span
    zero_record_ids = []      # records with ≥1 zero-snippet span

    for rec in records:
        e1 = rec.get("e1", {})
        snippets_lists = e1.get("ExampleSnippets")
        if "error" in e1 or not snippets_lists:
            continue
        records_with_phase2 += 1

        # Skip records that explicitly errored mid-phase2
        if "snippet_error" in e1:
            continue

        rec_total = 0
        rec_under = 0
        rec_zero = 0
        for sp in snippets_lists:
            n = sp.get("num_snippets", 0)
            num_snippets_hist[min(n, MAX_DOCS_PER_SPAN)] += 1
            total_spans += 1
            rec_total += 1
            if n < MAX_DOCS_PER_SPAN:
                undershoot_spans += 1
                rec_under += 1
            if n == 0:
                zero_snippet_spans += 1
                rec_zero += 1

        if rec_total == 0:
            continue
        if rec_under == 0:
            records_no_undershoot += 1
        elif rec_under == rec_total:
            records_all_undershoot += 1
            affected_record_ids.append(rec.get("id"))
        else:
            records_partial_undershoot += 1
            affected_record_ids.append(rec.get("id"))
        if rec_zero > 0:
            zero_record_ids.append(rec.get("id"))

    stats = {
        "model": model_dir,
        "missing": False,
        "total_records": total_records,
        "records_with_phase2": records_with_phase2,
        "records_no_undershoot": records_no_undershoot,
        "records_partial_undershoot": records_partial_undershoot,
        "records_all_undershoot": records_all_undershoot,
        "affected_records": records_partial_undershoot + records_all_undershoot,
        "affected_record_ids": affected_record_ids,
        "zero_record_ids": zero_record_ids,
        "total_spans": total_spans,
        "undershoot_spans": undershoot_spans,
        "zero_snippet_spans": zero_snippet_spans,
        "undershoot_pct": (undershoot_spans / total_spans * 100.0) if total_spans else 0.0,
        "zero_snippet_pct": (zero_snippet_spans / total_spans * 100.0) if total_spans else 0.0,
        "num_snippets_hist": {str(k): num_snippets_hist[k]
                              for k in range(MAX_DOCS_PER_SPAN + 1)},
    }
    return stats


def aggregate(stats_list: list) -> dict:
    keys_sum = ["total_records", "records_with_phase2",
                "records_no_undershoot", "records_partial_undershoot",
                "records_all_undershoot", "affected_records",
                "total_spans", "undershoot_spans", "zero_snippet_spans"]
    agg = {k: 0 for k in keys_sum}
    hist = Counter()
    for s in stats_list:
        if s.get("missing"):
            continue
        for k in keys_sum:
            agg[k] += s.get(k, 0)
        for k, v in s.get("num_snippets_hist", {}).items():
            hist[int(k)] += v
    agg["undershoot_pct"] = (
        agg["undershoot_spans"] / agg["total_spans"] * 100.0
        if agg["total_spans"] else 0.0
    )
    agg["zero_snippet_pct"] = (
        agg["zero_snippet_spans"] / agg["total_spans"] * 100.0
        if agg["total_spans"] else 0.0
    )
    agg["num_snippets_hist"] = {str(k): hist[k]
                                 for k in range(MAX_DOCS_PER_SPAN + 1)}
    return agg


# ============================================================
# Probe-span selection
# ============================================================
def candidate_probe_spans(stats_list: list, probe_total: int, logger) -> list:
    """Select ``probe_total`` worst-undershoot spans, distributing across models.

    Priority: zero_snippet spans first (gap = 10), then progressively higher
    num_snippets values. Ensures each model contributes at least 1 if it has
    undershoot. Returns a list of dicts with model + record_id + span_idx +
    begin/end/text.
    """
    # Per model: build sorted list of (gap, record_id, span_idx, begin, end, text, num_snippets)
    by_model: dict = {}
    for s in stats_list:
        if s.get("missing"):
            continue
        model_dir = s["model"]
        path = model_results_path(model_dir)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        cands = []
        for rec in records:
            e1 = rec.get("e1", {})
            snippets_lists = e1.get("ExampleSnippets")
            if "error" in e1 or not snippets_lists:
                continue
            for span_idx, sp in enumerate(snippets_lists):
                n = sp.get("num_snippets", 0)
                if n >= MAX_DOCS_PER_SPAN:
                    continue
                gap = MAX_DOCS_PER_SPAN - n
                cands.append({
                    "gap": gap,
                    "model": model_dir,
                    "record_id": rec.get("id"),
                    "span_idx": span_idx,
                    "begin": sp.get("span_begin"),
                    "end": sp.get("span_end"),
                    "span_text": sp.get("span_text", ""),
                    "stored_num_snippets": n,
                    "stored_doc_ixs": [
                        s2.get("doc_ix") for s2 in sp.get("snippets", [])
                        if s2.get("doc_ix") is not None
                    ],
                })
        # sort: gap desc, then by record_id (deterministic)
        cands.sort(key=lambda d: (-d["gap"], d.get("record_id") or 0))
        by_model[model_dir] = cands

    # round-robin pick to ensure model coverage
    selected = []
    seen = set()
    if not by_model:
        return []
    # First pass: 1 per model (top gap), then continue round-robin until probe_total reached.
    while len(selected) < probe_total:
        any_pick = False
        for model_dir, cands in by_model.items():
            if len(selected) >= probe_total:
                break
            while cands:
                c = cands.pop(0)
                key = (c["model"], c["record_id"], c["span_idx"])
                if key in seen:
                    continue
                seen.add(key)
                selected.append(c)
                any_pick = True
                break
        if not any_pick:
            break

    logger.info("Selected %d probe spans across %d models",
                len(selected), len({c["model"] for c in selected}))
    return selected


# ============================================================
# Phase 2 — API probe
# ============================================================
def probe_spans(probe_list: list, max_disp_len: int, probe_per_span: int,
                logger) -> tuple[list, int]:
    """For each probe span: find() once, then up to ``probe_per_span``
    get_doc_by_rank() on ranks NOT already in stored_doc_ixs.

    Returns (results, api_call_count).
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))

    PKG_DIR = os.path.join(REPO_ROOT, "infini-gram", "pkg")
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)

    from transformers import AutoTokenizer
    from infini_gram_api import InfiniGramAPIEngine

    logger.info("Loading Llama-2 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
        add_bos_token=False,
        add_eos_token=False,
    )

    engine = InfiniGramAPIEngine(index="v4_olmo-mix-1124_llama")
    api_calls = 0

    # Cache loaded JSONs by model to avoid re-reads.
    record_cache: dict = {}

    def _records_for(model_dir: str):
        if model_dir not in record_cache:
            with open(model_results_path(model_dir), "r", encoding="utf-8") as f:
                record_cache[model_dir] = json.load(f)
        return record_cache[model_dir]

    rng = random.Random(42)
    results = []

    for i, probe in enumerate(probe_list):
        logger.info("Probe [%d/%d] model=%s record=%s span=%d gap=%d text=%r",
                    i + 1, len(probe_list), probe["model"], probe["record_id"],
                    probe["span_idx"], probe["gap"], probe["span_text"][:80])

        # Re-tokenize the response to recover token IDs for this span.
        records = _records_for(probe["model"])
        rec = next((r for r in records if r.get("id") == probe["record_id"]), None)
        if rec is None:
            logger.warning("  Record id=%s not found in %s — skipping.",
                           probe["record_id"], probe["model"])
            continue

        response_text = rec.get("response", "") or ""
        try:
            response_ids = tokenizer.encode(response_text)
        except Exception as exc:
            logger.warning("  Tokenize failed: %s", exc)
            continue

        b = probe["begin"]
        e = probe["end"]
        if b is None or e is None or e > len(response_ids):
            logger.warning("  Invalid span [%s..%s] for response of %d tokens",
                           b, e, len(response_ids))
            continue
        span_ids = response_ids[b:e]

        # Sanity: decoded text should match stored span_text.
        try:
            decoded = tokenizer.decode(span_ids, skip_special_tokens=False)
            if decoded != probe["span_text"]:
                logger.info("  Note: decoded span differs slightly: stored=%r decoded=%r",
                            probe["span_text"][:60], decoded[:60])
        except Exception:
            pass

        # find() to get true corpus count
        try:
            fr = engine.find(input_ids=span_ids)
            api_calls += 1
        except Exception as exc:
            logger.warning("  find() failed: %s", exc)
            continue
        true_count = fr.get("cnt", 0)
        sbs = fr.get("segment_by_shard", [])

        stored_doc_ixs = set(probe["stored_doc_ixs"])

        # Build candidate (shard_idx, rank) list, shuffle, then probe up to N
        # ranks NOT in stored_doc_ixs (until we get probe_per_span fresh docs).
        candidates = []
        # Cap candidate enumeration to avoid huge ranges.
        CAND_CAP = 200
        for s_idx, seg in enumerate(sbs):
            if not seg or seg[1] - seg[0] <= 0:
                continue
            for r in range(seg[0], seg[1]):
                candidates.append((s_idx, r))
                if len(candidates) >= CAND_CAP:
                    break
            if len(candidates) >= CAND_CAP:
                break
        rng.shuffle(candidates)

        probe_docs = []
        probed_count = 0
        attempts = 0
        for (s_idx, r) in candidates:
            if probed_count >= probe_per_span:
                break
            if attempts >= probe_per_span * 3:  # safety cap
                break
            attempts += 1
            try:
                doc = engine.get_doc_by_rank(
                    s=s_idx, rank=r, max_disp_len=max_disp_len, query_ids=span_ids,
                )
                api_calls += 1
            except Exception as exc:
                logger.info("  get_doc_by_rank(s=%d, rank=%d) raised %s: %s",
                            s_idx, r, type(exc).__name__, exc)
                continue

            doc_ix = doc.get("doc_ix")
            if doc_ix in stored_doc_ixs:
                # Already retrieved successfully — skip (less informative).
                continue

            spans_field = doc.get("spans") or []
            none_count = 0
            if spans_field and isinstance(spans_field[0], list):
                none_count = sum(1 for x in spans_field[0] if x is None)

            probe_docs.append({
                "shard_s": s_idx,
                "rank": r,
                "doc_ix": doc_ix,
                "doc_len": doc.get("doc_len"),
                "blocked": bool(doc.get("blocked", False)),
                "metadata": doc.get("metadata", ""),
                "needle_offset": doc.get("needle_offset"),
                "spans_none_count": none_count,
                "spans_total": len(spans_field[0]) if spans_field and isinstance(spans_field[0], list) else 0,
                "raw_response_keys": list(doc.keys()),
                # Keep a small text preview if reconstructable
                "text_preview": (
                    "".join((t if isinstance(t, str) else "[NONE]") for t in spans_field[0])[:200]
                    if spans_field and isinstance(spans_field[0], list) else ""
                ),
            })
            probed_count += 1

        results.append({
            "model": probe["model"],
            "record_id": probe["record_id"],
            "span_idx": probe["span_idx"],
            "span_text": probe["span_text"],
            "stored_num_snippets": probe["stored_num_snippets"],
            "stored_doc_ixs": probe["stored_doc_ixs"],
            "true_corpus_count": true_count,
            "true_minus_stored": true_count - probe["stored_num_snippets"],
            "probe_docs": probe_docs,
            "n_probed": probed_count,
            "n_blocked": sum(1 for d in probe_docs if d["blocked"]),
            "n_with_none": sum(1 for d in probe_docs if d["spans_none_count"] > 0),
        })
        logger.info("  -> true_count=%d  stored=%d  probed=%d  blocked=%d  with_none=%d",
                    true_count, probe["stored_num_snippets"], probed_count,
                    sum(1 for d in probe_docs if d["blocked"]),
                    sum(1 for d in probe_docs if d["spans_none_count"] > 0))

    return results, api_calls


# ============================================================
# Report writer
# ============================================================
def write_report(out_dir: str, per_model: list, agg: dict,
                 probe: list, api_calls: int, probe_per_span: int) -> str:
    path = os.path.join(out_dir, "report.md")

    lines = []
    lines.append("# Blocked-snippet diagnosis (Tier 1)\n")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("")
    lines.append("Lower-bound diagnosis from existing E1 Phase-2 results "
                 f"(`MAX_DOCS_PER_SPAN={MAX_DOCS_PER_SPAN}`); a span is "
                 "**undershooting** if `num_snippets < 10`. The true blocked "
                 "rate is ≤ undershoot rate (some undershoot is from genuinely "
                 "rare spans). API probe of suspect spans confirms whether "
                 "the gap is dominated by `blocked=true` docs.\n")

    # Section 1: Lower bound
    lines.append("## 1. Lower-bound blocked rate (`num_snippets < 10`)\n")
    lines.append("| Model | Records (Phase-2) | Total spans | "
                 "Undershoot spans | Undershoot % | Zero-snippet spans | Zero-snippet % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in per_model:
        if s.get("missing"):
            lines.append(f"| {s['model']} | (missing) |  |  |  |  |  |")
            continue
        lines.append(
            f"| {s['model']} | {s['records_with_phase2']}/{s['total_records']} "
            f"| {s['total_spans']} | {s['undershoot_spans']} "
            f"| {s['undershoot_pct']:.2f}% "
            f"| {s['zero_snippet_spans']} "
            f"| {s['zero_snippet_pct']:.2f}% |"
        )
    lines.append(
        f"| **AGGREGATE** |  | **{agg['total_spans']}** "
        f"| **{agg['undershoot_spans']}** "
        f"| **{agg['undershoot_pct']:.2f}%** "
        f"| **{agg['zero_snippet_spans']}** "
        f"| **{agg['zero_snippet_pct']:.2f}%** |"
    )

    # Histogram
    lines.append("\n### num_snippets histogram (aggregate)\n")
    lines.append("| n | count |")
    lines.append("|---:|---:|")
    for k in range(MAX_DOCS_PER_SPAN + 1):
        lines.append(f"| {k} | {agg['num_snippets_hist'].get(str(k), 0)} |")

    # Section 2: Affected records
    lines.append("\n## 2. Affected records (≥1 undershoot span)\n")
    lines.append("| Model | Total | Phase-2 | No undershoot | Partial | All undershoot | Affected % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in per_model:
        if s.get("missing"):
            continue
        denom = max(s["records_with_phase2"], 1)
        pct = (s["affected_records"] / denom * 100.0) if denom else 0.0
        lines.append(
            f"| {s['model']} | {s['total_records']} | {s['records_with_phase2']} "
            f"| {s['records_no_undershoot']} | {s['records_partial_undershoot']} "
            f"| {s['records_all_undershoot']} | {pct:.1f}% |"
        )
    aff_pct = (agg["affected_records"] / max(agg["records_with_phase2"], 1) * 100.0)
    lines.append(
        f"| **AGGREGATE** | {agg['total_records']} | {agg['records_with_phase2']} "
        f"| {agg['records_no_undershoot']} | {agg['records_partial_undershoot']} "
        f"| {agg['records_all_undershoot']} | **{aff_pct:.1f}%** |"
    )

    # Section 3: Probe results
    lines.append("\n## 3. Probe of suspect spans (raw API response)\n")
    if not probe:
        lines.append("_No probe data (--analyze-only)._\n")
    else:
        lines.append(f"Probed **{len(probe)}** spans, up to {probe_per_span} "
                     f"missing docs per span. Total API calls: **{api_calls}**.\n")
        n_blocked = sum(p["n_blocked"] for p in probe)
        n_probed = sum(p["n_probed"] for p in probe)
        n_with_none = sum(p["n_with_none"] for p in probe)
        n_blocked_with_none = sum(
            1 for p in probe for d in p["probe_docs"]
            if d["blocked"] and d["spans_none_count"] > 0
        )
        lines.append(f"- Total docs probed: {n_probed}")
        lines.append(f"- `blocked=true`: {n_blocked} ({(n_blocked/n_probed*100.0) if n_probed else 0:.1f}%)")
        lines.append(f"- Has `None` in spans[0]: {n_with_none} "
                     f"({(n_with_none/n_probed*100.0) if n_probed else 0:.1f}%)")
        lines.append(f"- `blocked=true` AND has `None`: {n_blocked_with_none}")

        # Metadata clustering
        lines.append("\n### `blocked=true` doc metadata clusters (top 15)\n")
        meta_cluster = Counter()
        for p in probe:
            for d in p["probe_docs"]:
                if not d["blocked"]:
                    continue
                meta = d.get("metadata") or ""
                # try to extract dataset/path prefix
                if isinstance(meta, dict):
                    key = meta.get("path") or meta.get("source") or json.dumps(meta)[:80]
                else:
                    key = str(meta)[:120]
                # bucket by leading path component
                bucket = key.split("/")[0].split(".")[0] if key else "(empty)"
                meta_cluster[bucket] += 1
        if meta_cluster:
            lines.append("| Bucket (path prefix) | Count |")
            lines.append("|---|---:|")
            for bucket, c in meta_cluster.most_common(15):
                lines.append(f"| `{bucket}` | {c} |")
        else:
            lines.append("_No `blocked=true` docs found in probe sample._")

        # Per-span detail (compact)
        lines.append("\n### Probe-span detail\n")
        lines.append("| # | model | rec_id | gap | true_cnt | stored | n_probed | n_blocked | n_None | span_text |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for i, p in enumerate(probe):
            txt = p["span_text"][:60].replace("|", "\\|")
            lines.append(
                f"| {i+1} | {p['model']} | {p['record_id']} "
                f"| {MAX_DOCS_PER_SPAN - p['stored_num_snippets']} "
                f"| {p['true_corpus_count']} | {p['stored_num_snippets']} "
                f"| {p['n_probed']} | {p['n_blocked']} | {p['n_with_none']} | `{txt}` |"
            )

    # Section 4: Escalation judgment
    lines.append("\n## 4. Tier-2/3 escalation judgment\n")
    rate = agg["undershoot_pct"]
    if probe:
        n_blocked_total = sum(p["n_blocked"] for p in probe)
        n_probed_total = sum(p["n_probed"] for p in probe)
        blocked_share = (n_blocked_total / n_probed_total * 100.0) if n_probed_total else 0.0
    else:
        n_blocked_total = n_probed_total = 0
        blocked_share = 0.0

    lines.append(f"- **Lower-bound rate** (undershoot/total spans): **{rate:.2f}%**")
    lines.append(f"- **Probe blocked share** (blocked/probed): **{blocked_share:.1f}%**")
    lines.append("")

    if rate < 5 and blocked_share >= 80 and probe:
        verdict = "**Tier 1 sufficient.** Apply patch, re-run Phase 2 for affected records, no further escalation."
    elif rate >= 5 and rate < 15:
        verdict = "**Tier 2 recommended.** Re-query find() for ALL top-K spans (~2-3K calls) to derive precise corpus counts."
    elif rate >= 15:
        verdict = "**Tier 3 recommended.** Lower-bound rate is high; re-fetch all missing docs to fully characterize blocked vs other failure modes."
    else:
        verdict = "**Inconclusive.** Re-run probe with larger sample (try `--probe 50`) before deciding."

    lines.append(verdict)
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip the API probe; only run zero-API analysis.")
    p.add_argument("--probe", type=int, default=DEFAULT_PROBE_TOTAL,
                   help=f"Number of spans to probe (default: {DEFAULT_PROBE_TOTAL}).")
    p.add_argument("--probe-per-span", type=int, default=DEFAULT_PROBE_PER_SPAN,
                   help=f"Docs to probe per span (default: {DEFAULT_PROBE_PER_SPAN}).")
    p.add_argument("--max-disp-len", type=int, default=DEFAULT_MAX_DISP_LEN,
                   help=f"max_disp_len for get_doc_by_rank (default: {DEFAULT_MAX_DISP_LEN}).")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    out_dir = os.path.join(REPO_ROOT, "analysis", "blocked_diagnosis")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("=== Phase 1: Zero-API analysis ===")
    per_model = [analyze_one_model(m, logger) for m in MODEL_KEYS]
    agg = aggregate(per_model)
    logger.info("Aggregate: total_spans=%d undershoot=%d (%.2f%%) zero=%d (%.2f%%) "
                "affected_records=%d/%d",
                agg["total_spans"], agg["undershoot_spans"], agg["undershoot_pct"],
                agg["zero_snippet_spans"], agg["zero_snippet_pct"],
                agg["affected_records"], agg["records_with_phase2"])

    analysis_json = os.path.join(out_dir, "analysis.json")
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump({"per_model": per_model, "aggregate": agg}, f,
                  ensure_ascii=False, indent=2)
    logger.info("Wrote %s", analysis_json)

    probe_results: list = []
    api_calls = 0
    if not args.analyze_only:
        logger.info("=== Phase 2: Probe %d suspect spans ===", args.probe)
        probe_list = candidate_probe_spans(per_model, args.probe, logger)
        probe_results, api_calls = probe_spans(
            probe_list, args.max_disp_len, args.probe_per_span, logger,
        )
        probe_json = os.path.join(out_dir, "probe_raw_responses.json")
        with open(probe_json, "w", encoding="utf-8") as f:
            json.dump(probe_results, f, ensure_ascii=False, indent=2)
        logger.info("Wrote %s (api_calls=%d)", probe_json, api_calls)

    report_path = write_report(out_dir, per_model, agg, probe_results,
                                api_calls, args.probe_per_span)
    logger.info("Wrote %s", report_path)
    logger.info("Done. Total API calls: %d", api_calls)


if __name__ == "__main__":
    main()
