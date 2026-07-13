import argparse
import json
import os

from dotenv import load_dotenv

from auto_deal_hunter.agents.messaging_agent import MessagingAgent
from auto_deal_hunter.core.opportunity_store import OpportunityStore
from auto_deal_hunter.evaluation.judge import MessageJudge, corrupted_variants
from auto_deal_hunter.infra.paths import DEFAULT_DEALS_DB_PATH

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="Generate and judge saved deal notification messages.")
    parser.add_argument("--size", type=int, default=20, help="Number of saved opportunities to evaluate.")
    parser.add_argument(
        "--db",
        default=os.getenv("DEALS_DB_PATH", str(DEFAULT_DEALS_DB_PATH)),
        help="Path to the opportunities SQLite database.",
    )
    parser.add_argument("--output-json", help="Optional path to write per-message verdicts and aggregate metrics.")
    parser.add_argument(
        "--negative-control",
        action="store_true",
        help="Also judge deliberately corrupted variants of each message to measure judge "
        "recall (can the judge actually catch violations, not just pass clean messages).",
    )
    args = parser.parse_args()

    opportunities = OpportunityStore(args.db).list_opportunities()[: args.size]
    if not opportunities:
        print("No saved opportunities to evaluate.")
        return

    messenger = MessagingAgent()
    judge = MessageJudge()
    rows = []
    corrupted_rows = []
    for opportunity in opportunities:
        message = messenger.craft_message(
            opportunity.deal.product_description,
            opportunity.deal.price,
            opportunity.estimate,
        )
        verdict = judge.judge(opportunity, message)
        rows.append(
            {
                "url": opportunity.deal.url,
                "message": message,
                "faithful": verdict.faithful,
                "score": verdict.score,
                "issues": verdict.issues,
            }
        )
        status = "faithful" if verdict.faithful else "unfaithful"
        print(f"{status} score={verdict.score} issues={len(verdict.issues)} {opportunity.deal.url}")

        if args.negative_control:
            for name, corrupted in corrupted_variants(message).items():
                bad_verdict = judge.judge(opportunity, corrupted)
                caught = not bad_verdict.faithful
                corrupted_rows.append(
                    {
                        "url": opportunity.deal.url,
                        "corruption": name,
                        "caught": caught,
                        "score": bad_verdict.score,
                        "issues": bad_verdict.issues,
                    }
                )
                print(f"  [{name}] {'caught' if caught else 'MISSED'} score={bad_verdict.score}")

    faithful = sum(1 for row in rows if row["faithful"])
    aggregate = {
        "n": len(rows),
        "faithfulness_rate": faithful / len(rows),
        "mean_score": sum(row["score"] for row in rows) / len(rows),
    }
    print(
        f"\nfaithfulness_rate={aggregate['faithfulness_rate']:.0%} "
        f"mean_score={aggregate['mean_score']:.2f} n={aggregate['n']}"
    )
    if corrupted_rows:
        caught = sum(1 for row in corrupted_rows if row["caught"])
        aggregate["negative_control"] = {
            "n_corrupted": len(corrupted_rows),
            "judge_recall": caught / len(corrupted_rows),
            "by_corruption": {
                name: {
                    "n": sum(1 for r in corrupted_rows if r["corruption"] == name),
                    "caught": sum(1 for r in corrupted_rows if r["corruption"] == name and r["caught"]),
                }
                for name in sorted({r["corruption"] for r in corrupted_rows})
            },
        }
        print(
            f"negative control: judge_recall={aggregate['negative_control']['judge_recall']:.0%} "
            f"on {len(corrupted_rows)} corrupted messages "
            f"(clean pass rate above bounds the false-positive rate)"
        )
    from auto_deal_hunter.infra import usage

    print(usage.TRACKER.report())

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {"aggregate": aggregate, "messages": rows, "corrupted": corrupted_rows},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
