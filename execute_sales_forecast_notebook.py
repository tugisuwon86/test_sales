"""Run a notebook cell by cell without Jupyter and save outputs (text, tables, figures) into it.

Usage:
    python execute_sales_forecast_notebook.py                                  # model notebook
    python execute_sales_forecast_notebook.py sales-forecast-analysis.ipynb    # analysis notebook

Set SALES_PACKAGE_DIR to a local (non-OneDrive) copy of .annual_packages for faster imports.
"""

import base64
import contextlib
import io
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = Path(os.environ.get("SALES_PACKAGE_DIR", ROOT / ".annual_packages"))
sys.path.insert(0, str(PACKAGE_DIR))
os.chdir(ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["MPLBACKEND"] = "Agg"

import nbformat


def _figure_outputs():
    """Turn every open matplotlib figure into a PNG display output, then close it."""
    if "matplotlib.pyplot" not in sys.modules:
        return []
    plt = sys.modules["matplotlib.pyplot"]
    outputs = []
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        outputs.append(
            nbformat.v4.new_output(
                "display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": "<Figure>"}
            )
        )
    plt.close("all")
    return outputs


def run(notebook_path: Path):
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        print(f"Executing cell {count}", flush=True)
        outputs, stream = [], io.StringIO()

        def flush_text():
            if stream.getvalue():
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
                stream.seek(0)
                stream.truncate()

        def display(*objects):
            for obj in objects:
                flush_text()
                if hasattr(obj, "savefig"):
                    continue  # figures are collected after the cell
                data = {"text/plain": obj.to_string() if hasattr(obj, "to_string") else str(obj)}
                if hasattr(obj, "to_html"):
                    data["text/html"] = obj.to_html()
                outputs.append(nbformat.v4.new_output("display_data", data=data))

        namespace["display"] = display
        try:
            with contextlib.redirect_stdout(stream):
                exec(compile(cell.source, f"cell-{count}", "exec"), namespace)
            flush_text()
            cell.outputs = outputs + _figure_outputs()
            cell.execution_count = count
        except Exception as error:
            flush_text()
            cell.execution_count = count
            cell.outputs = outputs + [
                nbformat.v4.new_output(
                    "error", ename=type(error).__name__, evalue=str(error), traceback=traceback.format_exc().splitlines()
                )
            ]
            nbformat.write(notebook, notebook_path)
            raise
        nbformat.write(notebook, notebook_path)
    nbformat.validate(notebook)
    print(f"Executed {count} code cells of {notebook_path.name} successfully", flush=True)


if __name__ == "__main__":
    run(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "sales-forecast-walkforward.ipynb"))
