import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from auto_deal_hunter.infra.paths import PROJECT_ROOT

load_dotenv(override=True)

DEFAULT_MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
]


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def run_python(code: str, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build and evaluate vector stores for multiple embedding models.")
    parser.add_argument("--size", type=int, default=200, help="Holdout items to score per model.")
    parser.add_argument("--k", type=int, default=5, help="Neighbors per query.")
    parser.add_argument("--max-items", default="10000", help="MCAULEY_MAX_ITEMS used for each build.")
    parser.add_argument("--holdout-size", default="500", help="EVAL_HOLDOUT_SIZE used for each build.")
    parser.add_argument("--output-dir", default="docs/eval/embeddings", help="Directory for per-model JSON metrics.")
    parser.add_argument("--store-dir", default="data/embedding_compare", help="Directory for per-model Chroma stores.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Embedding model names to compare.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    store_dir = Path(args.store_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    holdout_path = output_dir / "holdout.json"

    results = []
    for model in args.models:
        name = safe_name(model)
        env = dict(os.environ)
        env.update(
            {
                "EMBEDDING_MODEL": model,
                "PRODUCTS_VECTORSTORE_PATH": str(store_dir / name),
                "EVAL_HOLDOUT_PATH": str(holdout_path),
                "MCAULEY_MAX_ITEMS": str(args.max_items),
                "EVAL_HOLDOUT_SIZE": str(args.holdout_size),
            }
        )
        metrics_path = output_dir / f"{name}.json"
        print(f"\n=== {model} ===")
        model_literal = repr(model)
        store_literal = repr(str(store_dir / name))
        holdout_path_literal = repr(str(holdout_path))
        max_items_literal = repr(str(args.max_items))
        holdout_literal = repr(str(args.holdout_size))
        metrics_literal = repr(str(metrics_path))
        build_code = (
            "import os; "
            "import scripts.build_vector_store as s; "
            f"s.EMBEDDING_MODEL={model_literal}; "
            f"s.DB_PATH={store_literal}; "
            f"s.HOLDOUT_PATH={holdout_path_literal}; "
            f"s.MAX_ITEMS=int({max_items_literal}); "
            f"s.HOLDOUT_SIZE=int({holdout_literal}); "
            "s.main()"
        )
        eval_code = (
            "import os, sys; "
            "import scripts.eval_retrieval as s; "
            f"model={model_literal}; store={store_literal}; metrics={metrics_literal}; "
            "s.EMBEDDING_MODEL=model; "
            "os.environ['PRODUCTS_VECTORSTORE_PATH']=store; "
            f"os.environ['EVAL_HOLDOUT_PATH']={holdout_path_literal}; "
            f"sys.argv=['eval_retrieval','--size','{args.size}','--k','{args.k}',"
            "'--output-json',metrics]; "
            "s.main()"
        )
        run_python(build_code, env)
        run_python(eval_code, env)
        with metrics_path.open() as f:
            metrics = json.load(f)
        results.append({"model": model, "metrics": metrics, "path": str(metrics_path)})

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
