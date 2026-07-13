import importlib.util
from pathlib import Path


def test_run_live_module_imports():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_live.py"
    spec = importlib.util.spec_from_file_location("run_live", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # must not require a live MT5 terminal
    assert hasattr(mod, "main")
