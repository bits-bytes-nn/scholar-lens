from pathlib import Path

from langchain_core.runnables import Runnable
from langchain_core.runnables.graph import MermaidDrawMethod

from ..logger import logger


def plot_langchain_graph(app: Runnable, output_file_path: Path) -> None:
    try:
        app.get_graph().draw_mermaid_png(
            output_file_path=str(output_file_path),
            draw_method=MermaidDrawMethod.API,
        )
    except Exception as e:
        logger.error(f"Error plotting Langchain graph: {str(e)}")
