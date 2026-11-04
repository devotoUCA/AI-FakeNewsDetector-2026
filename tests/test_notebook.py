"""El notebook de EDA se ejecuta de punta a punta sin errores."""

import pytest

from fakenews.config import get_paths

from .conftest import needs_data


@pytest.mark.slow
@needs_data
def test_notebook_executes(tmp_path):
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    path = get_paths().root / "notebooks" / "01_eda_y_preparacion.ipynb"
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(path.parent)}})
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
