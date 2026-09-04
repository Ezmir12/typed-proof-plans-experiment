"""Config loader tests: defaults, file load, env overrides, factories."""
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tpps.config import Settings, load_config, make_kernel, make_store

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"


def test_defaults_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_config()
    assert isinstance(s, Settings)
    assert s.llm.model and s.kernel.env_hash and s.store.owner
    assert s.llm.api_key == ""  # no secret leaking in from anywhere


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


def test_example_file_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local = tmp_path / "ex.toml"
    shutil.copy(EXAMPLE, local)
    s = load_config(local)
    assert s.llm.provider == "openai-compatible"
    assert s.llm.timeout_s == 60.0
    assert s.kernel.system == "lean4+mathlib"
    assert s.store.backend == "mock"
    assert s.app.log_level == "INFO"


def test_env_overrides_file_and_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TPPS_LLM_MODEL", "test-model")
    monkeypatch.setenv("TPPS_LLM_TIMEOUT_S", "5.5")
    monkeypatch.setenv("TPPS_STORE_OWNER", "tester")
    monkeypatch.setenv("TPPS_LLM_API_KEY", "secret-xyz")
    s = load_config()
    assert s.llm.model == "test-model"
    assert s.llm.timeout_s == 5.5
    assert s.store.owner == "tester"
    assert s.llm.api_key == "secret-xyz"


def test_unknown_section_and_key_rejected(tmp_path):
    bad_section = tmp_path / "bad.toml"
    bad_section.write_text("[nope]\nx = 1\n")
    with pytest.raises(ValueError):
        load_config(bad_section)
    bad_key = tmp_path / "badkey.toml"
    bad_key.write_text("[llm]\nmodel = \"m\"\nbogus = 1\n")
    with pytest.raises(ValueError):
        load_config(bad_key)


def test_factories_follow_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TPPS_KERNEL_ENV_HASH", "pinned-123")
    monkeypatch.setenv("TPPS_STORE_OWNER", "factory-owner")
    s = load_config()
    assert make_kernel(s).env_hash == "pinned-123"
    assert make_store(s).owner == "factory-owner"
    with pytest.raises(ValueError):
        make_store(replace(Settings(), store=replace(Settings().store,
                                                     backend="real")))


def test_dotenv_picked_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "TPPS_LLM_MODEL=dotenv-model\nTPPS_STORE_OWNER=dotenv-owner\n")
    s = load_config()
    assert s.llm.model == "dotenv-model"
    assert s.store.owner == "dotenv-owner"


def test_exported_env_beats_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("TPPS_LLM_MODEL=dotenv-model\n")
    monkeypatch.setenv("TPPS_LLM_MODEL", "exported-model")
    assert load_config().llm.model == "exported-model"


def test_dotenv_beats_config_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text('[llm]\nmodel = "toml-model"\n')
    (tmp_path / ".env").write_text("TPPS_LLM_MODEL=dotenv-model\n")
    assert load_config().llm.model == "dotenv-model"


def test_dotenv_format_quirks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# a comment\n\nOTHER_VAR=ignored\n"
        'TPPS_LLM_MODEL="quoted model"\n'
        "export TPPS_STORE_OWNER=exported-owner\n"
        "MALFORMED LINE\n")
    assert "OTHER_VAR" not in os.environ  # loader never mutates the environment
    s = load_config()
    assert s.llm.model == "quoted model"
    assert s.store.owner == "exported-owner"
