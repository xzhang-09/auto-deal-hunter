# Auto Deal Hunter — Autonomous Deal Discovery Agent

MCP-based autonomous deal hunting agent. LLM + MCP tools (scan, estimate, notify) collaborate to find and alert on bargain deals from RSS feeds.

## Features

- **Scan** — Fetches deals from DealNews RSS feeds (Electronics, Computers, Smart Home)
- **Estimate** — Estimates true market value via RAG (ChromaDB + sentence embeddings) and optional ensemble pricer (RAG + fine-tuned Qwen 2.5)
- **Notify** — Sends push notifications for compelling deals
- **Gradio UI** — Web interface with deal table, logs, and 3D t-SNE visualization of the product vector store

## Prerequisites

- Python 3.10+
- [HuggingFace](https://huggingface.co/) account and token
- (Optional) [Modal](https://modal.com/) account for cloud pricer deployment
- (Optional) [Weights & Biases](https://wandb.ai/) for training metrics

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env   # Windows: copy .env.example .env
```

Required: `OPENAI_API_KEY`, `HF_TOKEN`. Optional: `HF_USER`, `PUSHOVER_USER`/`PUSHOVER_TOKEN` (notifications), `USE_SPECIALIST` (fine-tuned pricer). See [Environment Variables](#environment-variables) for the full list.

## Quick Start

### 1. Build the vector store

Required before running the agent. Uses the items dataset from HuggingFace for RAG-based price estimation.

```bash
python build_vector_store.py
```

### 2. Run the agent

```bash
python deal_hunter.py
```

Opens a Gradio UI in your browser. The agent scans deals, estimates values, surfaces opportunities, and can send notifications. It auto-refreshes every 5 minutes.

## Optional: Fine-tuned Pricer

### Train the pricer (Google Colab)

1. Push pricer data to HuggingFace:
   ```bash
   python build_pricer_data.py
   ```

2. Run `train_pricer.py` on Colab (T4 for LITE_MODE, A100 for full).

3. Set `PRICER_RUN`, `PRICER_HUB_MODEL`, etc. in `.env` if using a custom run.

### Deploy the pricer (Modal)

```bash
modal deploy pricer_service.py
```

Requires `modal token` and a `huggingface-secret` (HF_TOKEN) in the Modal dashboard.

## Project Structure

```
auto-deal-hunter/
├── deal_hunter.py         # Main Gradio app entry point
├── deal_agent_framework.py # Agent orchestration and memory
├── agent_mcp.py           # MCP client integration
├── mcp_server.py          # MCP server (scan_deals, estimate_value, notify_deal)
├── build_vector_store.py  # Build ChromaDB from HuggingFace items dataset
├── build_pricer_data.py   # Prepare and push pricer training data
├── train_pricer.py        # Fine-tune Qwen 2.5 for price prediction (Colab)
├── pricer_service.py      # Modal deployment for fine-tuned pricer
├── eval_pricers.py        # Evaluate pricer models
├── agents/
│   ├── deals.py           # Deal models and RSS scraping
│   ├── items.py           # Item model and HuggingFace dataset loading
│   ├── scanner_agent.py   # Scans feeds and selects deals
│   ├── frontier_agent.py  # RAG-based price estimator
│   ├── specialist_agent.py# Fine-tuned model estimator
│   ├── ensemble_agent.py  # Combines RAG + specialist
│   └── messaging_agent.py # Push notifications
└── requirements.txt
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for LLM (MCP agent) |
| `HF_TOKEN` | HuggingFace API token |
| `HF_USER` | HuggingFace username (for datasets, models) |
| `PUSHOVER_USER` | Pushover user key (for push notifications) |
| `PUSHOVER_TOKEN` | Pushover app token |
| `LITE_MODE` | Use smaller datasets (default: `true`) |
| `PRODUCTS_VECTORSTORE_PATH` | Path to ChromaDB store |
| `USE_SPECIALIST` | Use fine-tuned pricer |
| `ITEMS_DATASET` | HuggingFace dataset (default: `ed-donner/items_lite` or `items_full`) |
| `PRICER_BASE_MODEL` | Base model for fine-tuning (default: `Qwen/Qwen2.5-3B-Instruct`) |
| `PRICER_RUN` | Pricer run name for Modal |
| `PRICER_HUB_MODEL` | HuggingFace model ID for fine-tuned pricer |

## License

MIT License. See [LICENSE](LICENSE) for details.
