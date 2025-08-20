import importlib.util
import pathlib
import pytest


def _load_substitutions_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    mod_path = root / "substitutions" / "substitutions.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Could not find substitutions.py at {mod_path}")
    spec = importlib.util.spec_from_file_location("substitutions_module", mod_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sub():
    """Return the loaded substitutions module (as 'sub').
    This works even if the 'substitutions' folder is not an installed package.
    """
    return _load_substitutions_module()
