"""HTTP client for the infini-gram public API (https://api.infini-gram.io/).

Uses JSON POST with ``query_ids`` (token ID lists) so tokenization matches the
local ``InfiniGramEngine`` workflow. Exposes ``count``, ``find``,
``get_doc_by_rank``, and ``count_cnf`` (CNF AND co-occurrence; API-only).
"""

from __future__ import annotations

import time
from typing import Optional

import requests

# Tighter defaults for high-volume E2 co-occurrence batches (optional at construct time).
E2_DEFAULT_MAX_RETRIES = 5
E2_DEFAULT_RETRY_DELAY = 2.0


class InfiniGramAPIEngine:
    """Mimic local ``InfiniGramEngine`` (count, find, get_doc_by_rank, count_cnf)."""

    API_URL = "https://api.infini-gram.io/"

    def __init__(
        self,
        index: str,
        max_retries: int = 8,
        retry_delay: float = 5.0,
    ):
        self.index = index
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def _post(self, payload: dict) -> dict:
        """POST with exponential-backoff retry."""
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.API_URL,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        raise RuntimeError(f"API error: {data['error']}")
                    time.sleep(1)  # rate-limit courtesy delay
                    return data

                if resp.status_code in (403, 429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()

            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError("API request failed after all retries")

    def count(self, input_ids: list) -> dict:
        """Count token sequence occurrences. Empty list → total corpus size."""
        if len(input_ids) == 0:
            payload = {
                "index": self.index,
                "query_type": "count",
                "query": "",
            }
        else:
            payload = {
                "index": self.index,
                "query_type": "count",
                "query_ids": input_ids,
            }
        return self._post(payload)

    def find(self, input_ids: list) -> dict:
        """Locate the token ID sequence in the corpus."""
        payload = {
            "index": self.index,
            "query_type": "find",
            "query_ids": input_ids,
        }
        return self._post(payload)

    def get_doc_by_rank(
        self,
        s: int,
        rank: int,
        max_disp_len: int = 80,
        query_ids: Optional[list] = None,
    ) -> dict:
        payload = {
            "index": self.index,
            "query_type": "get_doc_by_rank",
            "s": s,
            "rank": rank,
            "max_disp_len": max_disp_len,
        }
        if query_ids is not None:
            payload["query_ids"] = query_ids
        return self._post(payload)

    def count_cnf(
        self,
        cnf: list,
        max_clause_freq: int = 500000,
        max_diff_tokens: int = 1000,
    ) -> dict:
        """CNF AND co-occurrence query.

        cnf: [[[ids_A]], [[ids_B]]] — triply-nested list of token IDs.
        
        max_clause_freq: per-clause subsampling threshold. If a clause has
            more than this many corpus matches, the API returns an
            approximate count (result['approx']=True). Default 500,000 is the
            API maximum and minimizes approximation. Explicitly passing None
            falls back to the API server default (50,000). API-allowed
            range: [1, 500000].

        max_diff_tokens: co-occurrence window in tokens. API hard limit is
            1000; values above are silently clamped.
        """
        api_max_diff = min(max_diff_tokens, 1000)
        payload = {
            "index": self.index,
            "query_type": "count",
            "query_ids": cnf,
            "max_diff_tokens": api_max_diff,
        }
        if max_clause_freq is not None:
            payload["max_clause_freq"] = max_clause_freq
        return self._post(payload)