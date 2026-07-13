"""Model-tiering experiment: can a cheaper model run the scanner?

The scanner's job (pick 5 deals from RSS listings and summarize them) is the pipeline's
easiest LLM task, making it the natural candidate for a cheaper tier. This script feeds the
same scraped batch to each candidate model (the on-disk HTTP cache keeps the inputs
identical across runs), then scores every selected deal with ScanJudge — is the summary and
extracted price faithful to the raw listing? — and prices each model's tokens from its own
rate sheet. Judging always uses JUDGE_MODEL, so scanner quality is compared by a fixed
referee regardless of which model produced the summary.
"""
import argparse
import json

from dotenv import load_dotenv

from auto_deal_hunter.infra import usage

load_dotenv(override=True)

# USD per 1M tokens (input, output) for the *scanner* candidates compared here. Priced
# locally because usage.TRACKER applies one dominant-model rate, which is exactly what a
# mixed-model comparison can't use.
CANDIDATE_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
}


def main():
    parser = argparse.ArgumentParser(description="Compare scanner quality and cost across models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4.1-nano"],
        help="Scanner models to compare (default: current default vs. a nano tier).",
    )
    parser.add_argument("--output-json", help="Optional path for per-model results.")
    args = parser.parse_args()

    from auto_deal_hunter.agents.scanner_agent import ScannerAgent
    from auto_deal_hunter.core.source_ids import deal_id
    from auto_deal_hunter.evaluation.judge import ScanJudge

    judge = ScanJudge()
    results = []
    for model in args.models:
        agent = ScannerAgent()
        agent.MODEL = model  # instance attribute shadows the class default (SCANNER_MODEL)

        usage.TRACKER.reset()
        selection = agent.scan([])
        scan_tokens = (usage.TRACKER.prompt_tokens, usage.TRACKER.completion_tokens)
        in_rate, out_rate = CANDIDATE_PRICING.get(model, (0.0, 0.0))
        scan_cost = (scan_tokens[0] * in_rate + scan_tokens[1] * out_rate) / 1_000_000

        deals = selection.deals if selection else []
        # Re-scrape resolves from the HTTP cache, so this is the same batch the model saw.
        scraped_by_id = {deal_id(s.url): s for s in agent.fetch_deals([])}
        verdicts = []
        for deal in deals:
            source = scraped_by_id.get(deal_id(deal.url))
            if source is None:
                verdicts.append({"url": deal.url, "error": "no scraped source matched"})
                continue
            verdict = judge.judge(source.describe(), deal)
            verdicts.append(
                {
                    "url": deal.url,
                    "faithful": verdict.faithful,
                    "score": verdict.score,
                    "issues": verdict.issues,
                }
            )
            status = "faithful" if verdict.faithful else "UNFAITHFUL"
            print(f"  [{model}] {status} score={verdict.score} {deal.url}")

        judged = [v for v in verdicts if "faithful" in v]
        summary = {
            "model": model,
            "n_selected": len(deals),
            "n_judged": len(judged),
            "faithfulness_rate": (
                sum(1 for v in judged if v["faithful"]) / len(judged) if judged else None
            ),
            "mean_score": (
                sum(v["score"] for v in judged) / len(judged) if judged else None
            ),
            "scan_prompt_tokens": scan_tokens[0],
            "scan_completion_tokens": scan_tokens[1],
            "scan_cost_usd": round(scan_cost, 6),
            "verdicts": verdicts,
        }
        results.append(summary)
        rate = f"{summary['faithfulness_rate']:.0%}" if judged else "n/a"
        score = f"{summary['mean_score']:.2f}" if judged else "n/a"
        print(
            f"{model}: {len(deals)} selected, faithfulness={rate} mean_score={score} "
            f"scan cost ~${scan_cost:.6f} ({scan_tokens[0]:,} in + {scan_tokens[1]:,} out)\n"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
