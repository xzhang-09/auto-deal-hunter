"""Pure metric helpers for the end-to-end pricer evaluation.

Importable, unit-tested logic (see tests/test_eval_pricers.py); the CLI that runs the
PricerAgent over the holdout and prints these lives in scripts/eval_pricers.py.
"""
import math


def summarize(guesses: list[float], truths: list[float]) -> dict[str, float]:
    """Accuracy plus bias diagnostics against ground-truth prices.

    MAE/RMSE measure raw accuracy; `bias` (mean signed error) and `over_rate` (share of
    estimates above the truth) expose directional bias. A pricer that systematically
    over-predicts is the root cause behind estimates exceeding a deal's list price, so
    these are the metrics to watch when calibrating."""
    n = len(guesses)
    if n == 0:
        raise ValueError("summarize() requires at least one (guess, truth) pair")
    abs_errors = [abs(g - t) for g, t in zip(guesses, truths)]
    signed_errors = [g - t for g, t in zip(guesses, truths)]
    return {
        "n": n,
        "mae": sum(abs_errors) / n,
        "rmse": math.sqrt(sum(e * e for e in abs_errors) / n),
        "bias": sum(signed_errors) / n,
        "over_rate": sum(1 for s in signed_errors if s > 0) / n,
    }


def format_summary_line(stats: dict[str, float]) -> str:
    bias = stats["bias"]
    bias_str = f"{'+' if bias >= 0 else '-'}${abs(bias):,.2f}"
    return (
        f"MAE: ${stats['mae']:,.2f}   RMSE: ${stats['rmse']:,.2f}   "
        f"Bias: {bias_str}   Over-prediction: {stats['over_rate']:.0%}   n={stats['n']}"
    )


def threshold_violations(
    stats: dict[str, float],
    max_mae: float | None = None,
    max_abs_bias: float | None = None,
    max_over_rate: float | None = None,
) -> list[str]:
    violations = []
    if max_mae is not None and stats["mae"] > max_mae:
        violations.append(f"mae {stats['mae']:.2f} > max_mae {max_mae:.2f}")
    if max_abs_bias is not None and abs(stats["bias"]) > max_abs_bias:
        violations.append(
            f"abs(bias) {abs(stats['bias']):.2f} > max_abs_bias {max_abs_bias:.2f}"
        )
    if max_over_rate is not None and stats["over_rate"] > max_over_rate:
        violations.append(
            f"over_rate {stats['over_rate']:.2f} > max_over_rate {max_over_rate:.2f}"
        )
    return violations
