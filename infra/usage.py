"""Lightweight LLM token + cost accounting.

An AI-engineering project that calls a paid model on every scan should be able to say
what a run cost. This module accumulates token usage across all agent calls in-process
and reports an estimated dollar cost, so the framework and eval scripts can log it.

Prices are USD per 1M tokens and are easy to update as the price sheet changes; an
unknown model contributes tokens but $0 cost (and is flagged in the report).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Iterable

# USD per 1M tokens (input, output). Keep in sync with the provider price sheet.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
}


@dataclass
class UsageTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _unpriced_models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, usage) -> None:
        """Record one LLM response's usage. ``usage`` is the SDK usage object (or None)."""
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", 0)
        prompt = prompt or 0
        completion = completion or 0
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            if model not in PRICING:
                self._unpriced_models.add(model)

    def merge(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        calls: int = 0,
        unpriced_models: Iterable[str] = (),
    ) -> None:
        """Fold in a batch of usage that was accumulated elsewhere.

        The deal-hunting agents (scanner/pricer/messenger) run inside the MCP server
        *subprocess*, where they record to a separate UsageTracker instance. The client
        pulls those totals back via the get_run_usage tool and merges them here so the
        orchestrator's per-run cost report reflects the full spend, not just the
        orchestration loop's own calls. Token counts are summed directly (no SDK usage
        object), and ``calls`` is added as a batch rather than incremented by one."""
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.calls += calls
            self._unpriced_models.update(unpriced_models)

    @property
    def unpriced_models(self) -> list[str]:
        with self._lock:
            return sorted(self._unpriced_models)

    @property
    def estimated_cost(self) -> float:
        # Cost is approximate: it applies the dominant model's rate uniformly. For a
        # single-model project (all gpt-4o-mini) this is exact; mixed models are a rough
        # upper-bound on the cheaper tier. Good enough for a per-run budget signal.
        in_rate, out_rate = PRICING.get(LLM_MODEL_FOR_COST(), (0.0, 0.0))
        return (self.prompt_tokens * in_rate + self.completion_tokens * out_rate) / 1_000_000

    def report(self) -> str:
        line = (
            f"LLM usage: {self.calls} calls, "
            f"{self.prompt_tokens:,} in + {self.completion_tokens:,} out tokens, "
            f"~${self.estimated_cost:.4f}"
        )
        if self._unpriced_models:
            line += f" (no price for: {', '.join(sorted(self._unpriced_models))})"
        return line

    def log(self) -> None:
        logging.info(self.report())

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = self.completion_tokens = self.calls = 0
            self._unpriced_models.clear()


def LLM_MODEL_FOR_COST() -> str:
    # Imported lazily to avoid a circular import with infra.config at module load.
    from infra.config import LLM_MODEL

    return LLM_MODEL


# Process-wide singleton; agents call usage.TRACKER.record(...).
TRACKER = UsageTracker()
