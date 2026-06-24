import logging
import os
import queue
import sys
import threading
import time
import warnings
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.deal_agent_framework import DealAgentFramework
from app.log_utils import reformat
from app.paths import DEFAULT_VECTORSTORE_PATH
from models.deals import Deal, Opportunity
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)

warnings.filterwarnings(
    "ignore",
    message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
    category=DeprecationWarning,
)


APP_CSS = """
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
    checkable = [opp for opp in opportunities if opp.deal.list_price is not None]
    overestimates = sum(1 for opp in checkable if opp.is_overestimate)
    return {
        "opportunities": str(len(opportunities)),
        "overestimates": f"{overestimates}/{len(checkable)}",
        "ready": vector_store_ready,
    }


OPPORTUNITIES_NOTE = (
    "Est. Value is the model's independent fair-value estimate; Savings and Below Est. % are "
    "capped at the list price, so an estimate higher than the list price never inflates them."
)


def stats_html(stats: dict) -> str:
    parts = []
    if not stats["ready"]:
        parts.append(
            '<div class="setup-warning">⚠ Vector store not built — run '
            "<code>python scripts/build_vector_store.py</code> before scanning.</div>"
        )
    parts.append(
        f'<div class="section-caption"><b>{stats["opportunities"]} saved deals.</b> '
        f"Est &gt; list price: {stats['overestimates']}. {OPPORTUNITIES_NOTE}</div>"
    )
    return "".join(parts)


def table_for(opps):
    return [
        [
            opp.deal.product_description,
            f"${opp.deal.price:.2f}",
            f"${opp.deal.list_price:.2f}" if opp.deal.list_price else "n/a",
            f"${opp.estimate:.2f}",
            f"${opp.discount:.2f}",
            below_estimate_percent(opp.effective_value, opp.deal.price),
            f"[View]({opp.deal.url})",
        ]
        for opp in opps
    ]


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
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        with gr.Blocks(
            title="Auto Deal Hunter",
            fill_width=True,
            css=APP_CSS,
            analytics_enabled=False,
        ) as ui:
            log_data = gr.State([])

            def update_output(log_data, log_queue, result_queue):
                initial_result = table_for(self.get_agent_framework().memory)
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
                    documents, vectors, colors, labels = DealAgentFramework.get_plot_data(max_datapoints=800)
                    fig = go.Figure()
                    for label in dict.fromkeys(labels):  # unique, first-seen order
                        idx = [i for i, l in enumerate(labels) if l == label]
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
                return table_for(self.get_agent_framework().run())

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                    except Exception:
                        logging.exception("Agent run failed")
                        result = table_for(self.get_agent_framework().memory) if self.agent_framework else []
                    result_queue.put(result)

                thread = threading.Thread(target=worker, daemon=True)
                thread.start()

                for log_data, stats, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, stats, output, final_result

            def do_select(selected_index: gr.SelectData):
                opportunities = self.get_agent_framework().memory
                if not opportunities:
                    return
                row = selected_index.index[0]
                if row < len(opportunities):
                    from agents.messaging_agent import MessagingAgent

                    MessagingAgent().alert(opportunities[row])

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
                    ],
                    datatype=["str", "str", "str", "str", "str", "str", "markdown"],
                    wrap=True,
                    column_widths=["40%", "10%", "10%", "10%", "10%", "10%", "10%"],
                    row_count=10,
                    col_count=7,
                    max_height=400,
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
                    plot = gr.Plot(value=get_plot(), show_label=False)
                    gr.Markdown(REFERENCE_MAP_CAPTION)

            def load_memory(initial_log_data):
                current_memory = self.get_agent_framework().memory
                return (
                    initial_log_data,
                    stats_html(dashboard_stats(current_memory, vector_store_ready())),
                    html_for(initial_log_data),
                    table_for(current_memory),
                )

            ui.load(
                load_memory,
                inputs=[log_data],
                outputs=[log_data, status_cards, logs, opportunities_dataframe],
            )
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

            opportunities_dataframe.select(do_select)

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
