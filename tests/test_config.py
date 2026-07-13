import os
import pytest
from mt5gold.config import DataConfig, config_hash, require_env


def test_dataconfig_defaults_are_xauusd_m1():
    cfg = DataConfig()
    assert cfg.symbol == "XAUUSD"
    assert cfg.base_timeframe == "M1"


def test_config_hash_is_stable_and_sensitive():
    a = DataConfig(symbol="XAUUSD")
    b = DataConfig(symbol="XAUUSD")
    c = DataConfig(symbol="EURUSD")
    assert config_hash(a) == config_hash(b)          # deterministic
    assert config_hash(a) != config_hash(c)          # sensitive to fields
    assert len(config_hash(a)) == 12


def test_require_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MT5GOLD_TEST_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="MT5GOLD_TEST_SECRET"):
        require_env("MT5GOLD_TEST_SECRET")


def test_require_env_returns_value(monkeypatch):
    monkeypatch.setenv("MT5GOLD_TEST_SECRET", "abc")
    assert require_env("MT5GOLD_TEST_SECRET") == "abc"
