"""In-process deal-hunting pipeline: scan -> estimate -> deterministically select -> notify.

This is the **default** scan path. It calls the same agents directly, skipping the MCP
subprocess and the LLM tool-calling loop (both kept in ``app.mcp_client`` as an optional demo
of MCP orchestration, reachable via ``SCAN_MODE=agent``). Going direct removes the three
fragile seams the indirection created:

  * PYTHONPATH injection into a spawned child process,
  * merging token-usage counters back across a process boundary, and
  * re-pairing the model's tool-call arguments to the deals that were scanned.

In-process, the agents record straight into the shared ``usage.TRACKER`` and each estimate is
already attached to its deal, so none of that machinery is needed. The MCP server
(``app.mcp_server``) still exposes scan/estimate/notify for any external MCP client.

The LLM is not the decision-maker here (nor is it in the agent loop): candidates are gathered,
then the single best deal is chosen deterministically by ``core.scoring.best_opportunity`` over
the list-price-capped savings.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from core.scoring import best_opportunity
from core.source_ids import deal_id
from domain.deal import Opportunity
from infra.config import ESTIMATE_MISMATCH_RATIO, RAG_MIN_CONFIDENCE


class DealPipeline:
    """Runs the scan->estimate->select->notify flow in-process against a Chroma collection.

    Agents are built lazily and cached for the pipeline's lifetime (the estimator loads a
    Sentence-Transformers model, so we avoid rebuilding it every scan). Tests inject stubs by
    setting the private ``_scanner`` / ``_pricer`` / ``_messenger`` attributes."""

    def __init__(self, collection):
        self.collection = collection
        self._scanner = None
        self._pricer = None
        self._messenger = None

    @property
    def scanner(self):
        if self._scanner is None:
            from agents.scanner_agent import ScannerAgent

            self._scanner = ScannerAgent()
        return self._scanner

    @property
    def pricer(self):
        if self._pricer is None:
            from agents.pricer_agent import PricerAgent

            self._pricer = PricerAgent(self.collection)
        return self._pricer

    @property
    def messenger(self):
        if self._messenger is None:
            from agents.messaging_agent import MessagingAgent

            self._messenger = MessagingAgent()
        return self._messenger

    def run(self, memory: list) -> Tuple[list, Optional[Opportunity]]:
        """Scan, estimate every candidate, pick the best deterministically, and notify.

        Returns ``(memory, best_opportunity_or_None)`` to match ``mcp_client.run_sync`` so the
        orchestrator can use either path interchangeably. The best deal is always returned (and
        thus saved by the caller); the push notification is withheld when its estimate rests on
        a weak RAG match (confidence below ``RAG_MIN_CONFIDENCE``)."""
        selection = self.scanner.scan(memory=memory)
        if not selection or not selection.deals:
            return memory, None

        candidates: List[Opportunity] = []
        confidence_by_id: dict[str, float] = {}
        for deal in selection.deals:
            try:
                estimate, confidence = self.pricer.estimate_with_confidence(deal.product_description)
            except ValueError as exc:
                # No usable estimate (uninformative RAG context); skip rather than surface a
                # fabricated value. See PricerAgent.price for the guard details.
                logging.info("Skipping deal with no usable estimate: %s", exc)
                continue
            opportunity = Opportunity(
                deal=deal, estimate=estimate, retrieval_confidence=confidence
            )
            if opportunity.is_comparables_mismatch(ESTIMATE_MISMATCH_RATIO):
                # The estimate is a multiple of the list price: the RAG neighbors were the
                # wrong kind of product, so the estimate deserves no trust. Zero the
                # confidence (blocks the push via the RAG_MIN_CONFIDENCE gate below) but keep
                # the candidate -- its savings are list-price-capped and therefore real.
                logging.info(
                    "Comparables mismatch (estimate %.2f > %.1fx list %.2f): zeroing "
                    "confidence for %s",
                    estimate, ESTIMATE_MISMATCH_RATIO, deal.list_price, deal.url,
                )
                confidence = 0.0
                opportunity.retrieval_confidence = confidence
            candidates.append(opportunity)
            confidence_by_id[deal_id(deal.url)] = confidence

        best = best_opportunity(candidates)
        if best is None:
            return memory, None

        confidence = confidence_by_id.get(deal_id(best.deal.url))
        if confidence is not None and confidence < RAG_MIN_CONFIDENCE:
            logging.info(
                "Withholding push: estimate confidence %.2f < threshold %.2f for %s",
                confidence, RAG_MIN_CONFIDENCE, best.deal.url,
            )
        else:
            self.messenger.notify(
                best.deal.product_description,
                best.deal.price,
                best.estimate,
                best.deal.url,
                best.deal.list_price,
                best.deal.quantity,
            )
        return memory, best
