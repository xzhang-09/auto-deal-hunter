import logging
import os
import queue
import threading
import time
import warnings
from collections import Counter
import gradio as gr

from auto_deal_hunter.app.orchestrator import Orchestrator
from auto_deal_hunter.core.identity_policy import display_description
from auto_deal_hunter.infra.config import ESTIMATE_MISMATCH_RATIO
from auto_deal_hunter.core.source_ids import deal_id
from auto_deal_hunter.infra.log_utils import reformat
from auto_deal_hunter.infra.paths import DEFAULT_VECTORSTORE_PATH
from auto_deal_hunter.domain.deal import Deal, Opportunity  # noqa: F401  re-exported for tests/UI helpers
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)

warnings.filterwarnings(
    "ignore",
    message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
    category=DeprecationWarning,
)


APP_CSS = """
/* The opportunities table is a click-to-act surface (👍/👎/🔔 route through .select), not a
   spreadsheet: hide Gradio's cell-selection outline and cell menu button so a click doesn't
   look like it selected a cell for editing. */
.table-wrap td.cell-selected::after { display: none !important; }
.table-wrap td.cell-selected { box-shadow: none !important; outline: none !important; }
.table-wrap .selection-button { display: none !important; }
.app-header {
    border-bottom: 1px solid #e5e7eb;
    padding: 10px 0 14px;
}
.app-title {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.2;
}
.app-subtitle {
    color: #4b5563;
    font-size: 14px;
    margin-top: 4px;
}
.setup-warning {
    border: 1px solid #f5c6cb;
    background: #fff5f5;
    color: #842029;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 10px;
    font-size: 14px;
}
.setup-warning code {
    background: #fdeaec;
    padding: 1px 5px;
    border-radius: 4px;
}
.gradio-dataframe, .gradio-dataframe:focus-within {
    border-color: #e5e7eb !important;
}
.section-caption-row {
    margin-top: -8px !important;
}
.section-caption {
    color: #4b5563;
    font-size: 13px;
    line-height: 1.45;
    margin: 0 0 10px;
    padding-top: 2px;
}
"""

REFERENCE_MAP_CAPTION = (
    '<div class="section-caption">3D t-SNE projection of embedded reference products. '
    "Nearby dots represent similar items; hover to inspect product titles.</div>"
)



class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(log_data[-18:])
    return f"""
    <div style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def below_estimate_percent(value: float, price: float) -> str:
    """How far below the (list-price-capped) estimated value the deal price sits, as a
    percentage of that value. Always <= 100% — unlike a price-based ratio, it reads as an
    intuitive "X% below fair value" rather than a confusing >100% figure."""
    if value <= 0 or price >= value:
        return "n/a"
    return f"{((value - price) / value) * 100:.1f}%"


def vector_store_ready(path=DEFAULT_VECTORSTORE_PATH) -> bool:
    return os.path.exists(os.path.join(str(path), "chroma.sqlite3"))


def dashboard_stats(opportunities, vector_store_ready: bool) -> dict:
    return {
        "opportunities": str(len(opportunities)),
        "ready": vector_store_ready,
    }


OPPORTUNITIES_NOTE = (
    "Est. Value is the model's independent fair-value estimate; Savings and Below Est. % are "
    "capped at the list price, so an estimate higher than the list price never inflates them. "
    "Click 👍 if you verified the price is genuinely below what the item sells for elsewhere, "
    "👎 if the bargain is false (normal street price, junk listing, or a bad estimate), and "
    "🔔 to push that deal to your phone. A ✅ marks your saved label. "
    "Est. Value shows ⚠️ n/a when the retrieved comparable products were a poor match "
    "(estimate far above list price) — no trustworthy estimate exists there, and Savings "
    "falls back to the seller's own list-price discount."
)


def stats_html(stats: dict) -> str:
    parts = []
    if not stats["ready"]:
        parts.append(
            '<div class="setup-warning">⚠ Vector store not built — run '
            "<code>python -m auto_deal_hunter.scripts.build_vector_store</code> before scanning.</div>"
        )
    parts.append(
        f'<div class="section-caption"><b>{stats["opportunities"]} saved deals.</b> '
        f"{OPPORTUNITIES_NOTE}</div>"
    )
    return "".join(parts)


# Column indices of the in-row action cells; keep in sync with the Dataframe headers below.
# Clicking one of these cells acts on its row directly (see handle_cell_action), so feedback
# and alerts need no separate row-selection step -- and the saved label stays visible in-row.
GOOD_COL = 7
BAD_COL = 8
ALERT_COL = 9


def table_for(opps, feedback_by_id=None):
    feedback_by_id = feedback_by_id or {}
    rows = []
    for opp in opps:
        quantity = opp.deal.quantity
        # Show the pack-level prices the user actually pays. The per-unit basis is internal --
        # used only to value a multipack against per-unit comparables -- so multiply it back by
        # the pack size for display. Quantity 1 leaves single-item prices unchanged.
        price = opp.deal.price * quantity
        list_price = opp.deal.list_price * quantity if opp.deal.list_price else None
        estimate = opp.estimate * quantity
        description = display_description(opp.deal.product_description, quantity)
        label = feedback_by_id.get(deal_id(opp.deal.url))
        # A comparables-mismatch estimate (a multiple of list price) is untrustworthy: blank
        # it rather than display a nonsense figure or fake agreement with the list price.
        # Savings stay visible -- they are list-price-capped and therefore real.
        mismatch = opp.is_comparables_mismatch(ESTIMATE_MISMATCH_RATIO)
        rows.append(
            [
                description,
                f"${price:.2f}",
                f"${list_price:.2f}" if list_price else "n/a",
                "⚠️ n/a" if mismatch else f"${estimate:.2f}",
                f"${opp.total_discount:.2f}",
                "n/a" if mismatch else below_estimate_percent(opp.effective_value, opp.deal.price),
                f"[View]({opp.deal.url})",
                "✅" if label == "good_deal" else "👍",
                "✅" if label == "bad_deal" else "👎",
                "🔔",
            ]
        )
    return rows


def refreshed_table(agent_framework):
    """Current opportunity table with saved feedback labels rendered into the action cells."""
    return table_for(agent_framework.memory, agent_framework.opportunity_store.feedback_map())


def handle_cell_action(agent_framework, row: int | None, col: int | None):
    """Route a Dataframe cell click: 👍/👎 write feedback for that row, 🔔 alerts it.

    Returns the refreshed table. Clicks on non-action cells fall through to a plain refresh
    so the select event never leaves the table stale."""
    if row is None or col is None:
        return refreshed_table(agent_framework)
    if row < 0 or row >= len(agent_framework.memory):
        return refreshed_table(agent_framework)
    if col == GOOD_COL:
        return mark_feedback_for_row(agent_framework, row, "good_deal")
    if col == BAD_COL:
        return mark_feedback_for_row(agent_framework, row, "bad_deal")
    if col == ALERT_COL:
        alert_for_row(agent_framework, row)
    return refreshed_table(agent_framework)


def mark_feedback_for_row(agent_framework, row: int | None, label: str):
    if row is None:
        return refreshed_table(agent_framework)
    if row < 0 or row >= len(agent_framework.memory):
        return refreshed_table(agent_framework)
    opportunity = agent_framework.memory[row]
    agent_framework.opportunity_store.mark_feedback(opportunity.deal.url, label)
    return refreshed_table(agent_framework)


def alert_for_row(agent_framework, row: int | None) -> None:
    if row is None:
        return
    if row < 0 or row >= len(agent_framework.memory):
        return
    from auto_deal_hunter.agents.messaging_agent import MessagingAgent

    MessagingAgent().alert(agent_framework.memory[row])


def setup_logging(log_queue):
    logger = logging.getLogger()
    for existing in list(logger.handlers):
        if isinstance(existing, QueueHandler):
            logger.removeHandler(existing)
            existing.close()
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = Orchestrator()
        return self.agent_framework

    def run(self):
        with gr.Blocks(
            title="Auto Deal Hunter",
            css=APP_CSS,
            analytics_enabled=False,
        ) as ui:
            log_data = gr.State([])

            def update_output(log_data, log_queue, result_queue):
                initial_result = refreshed_table(self.get_agent_framework())
                final_result = None
                while True:
                    current_memory = self.get_agent_framework().memory
                    current_stats = stats_html(
                        dashboard_stats(current_memory, vector_store_ready())
                    )
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, current_stats, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            current_stats = stats_html(
                                dashboard_stats(self.get_agent_framework().memory, vector_store_ready())
                            )
                            yield log_data, current_stats, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_plot():
                try:
                    documents, vectors, colors, labels = Orchestrator.get_plot_data(max_datapoints=800)
                    fig = go.Figure()
                    # Legend ordered by point count, biggest group first: readable and stable
                    # across rebuilds (vs. the arbitrary first-seen order of the raw data).
                    for label, _ in Counter(labels).most_common():
                        idx = [i for i, lbl in enumerate(labels) if lbl == label]
                        fig.add_trace(
                            go.Scatter3d(
                                x=vectors[idx, 0],
                                y=vectors[idx, 1],
                                z=vectors[idx, 2],
                                mode="markers",
                                marker=dict(size=2, color=[colors[i] for i in idx], opacity=0.7),
                                name=label,
                                text=[documents[i][:60] for i in idx],
                                hovertemplate="%{text}<extra></extra>",
                            )
                        )
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="t-SNE x",
                            yaxis_title="t-SNE y",
                            zaxis_title="t-SNE z",
                            aspectmode="manual",
                            aspectratio=dict(x=2.2, y=2.2, z=1),
                            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                        ),
                        legend=dict(itemsizing="constant", font=dict(size=10), x=0, y=1),
                        height=400,
                        margin=dict(r=5, b=1, l=5, t=2),
                    )
                    return fig
                except Exception:
                    fig = go.Figure()
                    fig.update_layout(title="Build vector store first (run build_vector_store.py)", height=400)
                    return fig

            def do_run():
                self.get_agent_framework().run()
                return refreshed_table(self.get_agent_framework())

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                    except Exception:
                        logging.exception("Agent run failed")
                        result = refreshed_table(self.get_agent_framework()) if self.agent_framework else []
                    result_queue.put(result)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                for log_data, stats, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, stats, output, final_result

            def do_select(selected_index: gr.SelectData):
                # Route the click to the in-row action columns (👍/👎/🔔); clicks on any
                # other cell just refresh the table.
                row, col = selected_index.index[0], selected_index.index[1]
                return handle_cell_action(self.get_agent_framework(), row, col)

            with gr.Row(elem_classes=["app-header"]):
                gr.Markdown(
                    '<div class="app-title">Auto Deal Hunter</div>'
                    '<div class="app-subtitle">Scans retail feeds, estimates fair value with RAG, and surfaces underpriced products.</div>'
                )
            with gr.Row():
                run_button = gr.Button("Scan now", variant="primary", size="sm", scale=0)
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    label="Opportunities",
                    headers=[
                        "Deal",
                        "Deal Price",
                        "List Price",
                        "Est. Value",
                        "Savings ($)",
                        "Below Est. %",
                        "URL",
                        "Good deal",
                        "Bad deal",
                        "Send alert",
                    ],
                    datatype=["str"] * 6 + ["markdown", "str", "str", "str"],
                    wrap=True,
                    column_widths=["28%", "8%", "8%", "8%", "8%", "8%", "8%", "8%", "8%", "8%"],
                    row_count=10,
                    col_count=10,
                    max_height=400,
                    # Read-only: cells route clicks to 👍/👎/🔔 actions via .select; letting
                    # Gradio's default editing UI open on click would suggest edits persist
                    # when the table is overwritten on every refresh.
                    interactive=False,
                )
            with gr.Row(elem_classes=["section-caption-row"]):
                status_cards = gr.HTML(
                    value=stats_html(
                        dashboard_stats(self.get_agent_framework().memory, vector_store_ready())
                    )
                )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Agent logs")
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("### Product reference map")
                    # Rendered lazily via ui.load below: the t-SNE projection is expensive
                    # (fit over up to 800 vectors) and computing it eagerly here would block
                    # every page open. Starting empty lets the UI appear immediately and the
                    # map fill in once the session loads.
                    plot = gr.Plot(show_label=False)
                    gr.Markdown(REFERENCE_MAP_CAPTION)

            def load_memory(initial_log_data):
                current_memory = self.get_agent_framework().memory
                return (
                    initial_log_data,
                    stats_html(dashboard_stats(current_memory, vector_store_ready())),
                    html_for(initial_log_data),
                    refreshed_table(self.get_agent_framework()),
                )

            ui.load(
                load_memory,
                inputs=[log_data],
                outputs=[log_data, status_cards, logs, opportunities_dataframe],
            )
            # Populate the reference map after the page loads, off the initial render path.
            ui.load(get_plot, inputs=None, outputs=[plot])
            run_button.click(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, status_cards, logs, opportunities_dataframe],
            )

            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, status_cards, logs, opportunities_dataframe],
            )

            opportunities_dataframe.select(do_select, outputs=[opportunities_dataframe])

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
