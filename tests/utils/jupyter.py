import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from pathlib import Path
import shutil


def run_notebook(path):
    path = Path(path)
    checkpoint_path = path.parent / "checkpoints"
    # only clean up if the directory did not exist before the test
    cleanup_checkpoints = not checkpoint_path.exists()
    with open(str(path), encoding="utf-8") as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    kernel = ExecutePreprocessor(timeout=3600, kernel_name="python3", resources={"metadata": {"path": path.parent}})

    try:
        result = kernel.preprocess(nb)
    finally:
        if cleanup_checkpoints and checkpoint_path.exists():
            # clean up if the directory was created by the test
            shutil.rmtree(checkpoint_path)

    return result
