"""The code under test must be *this* checkout (portability guard).

A copied project carries a copied `.venv`, and an editable install hardcodes the
path it was built at. When that path is a different checkout, `import jane`
silently loads the other copy's code — verified here: the venv resolved to
`C:\\Users\\kingh\\Downloads\\ai data analyist\\src` while the repo lived
elsewhere, so a running app mixed this bridge with another tree's engine and
none of the fixes in this checkout applied. These checks fail loudly instead of
leaving a silent mix, and they pin the rule that paths derive from the repo
root rather than a literal.
"""
import sys
from pathlib import Path

from jane.config import PROJECT_ROOT

REPO = Path(__file__).resolve().parents[1]
REPO_SRC = REPO / "src"


def test_the_imported_jane_is_this_checkout():
    import jane

    loaded = Path(jane.__file__).resolve()
    assert loaded.is_relative_to(REPO_SRC.resolve()), (
        f"jane was imported from {loaded}, outside this checkout ({REPO_SRC}).\n"
        f"interpreter: {sys.executable}\n"
        "a copied .venv keeps an editable path pointing at the old location — "
        "repoint it (see AGENTS.md) or the fixes in this tree never run."
    )


def test_repo_root_is_derived_not_hardcoded():
    """PROJECT_ROOT must follow the code (config.py derives it from __file__)."""
    assert PROJECT_ROOT.resolve() == REPO.resolve(), (
        f"PROJECT_ROOT={PROJECT_ROOT} but this checkout is {REPO}")
    assert (PROJECT_ROOT / "config.yaml").is_file()
    assert (PROJECT_ROOT / "src" / "jane").is_dir()


def test_the_bundled_assets_ship_and_are_found():
    """Sample datasets and the Streamlit backdrop are read at runtime; a
    non-editable install only has them if pyproject declares package-data
    (`[tool.setuptools.package-data]`). Paths come from the package itself, not
    from `jane.app`: that module is a Streamlit script and cannot be imported
    outside `streamlit run` (it touches st.session_state at module level).
    """
    import jane

    jane_dir = Path(jane.__file__).resolve().parent
    assert (jane_dir / "sample_data" / "retail_sales.csv").is_file()
    assert list((jane_dir / "sample_data").glob("*.csv")), "no bundled samples"
    assert (jane_dir / "assets" / "backdrop.jpg").is_file()
