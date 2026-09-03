"""Setup configuration: TOML file plus TPPS_ environment overrides.

Secrets never live in files: the config only names the environment variable
that holds the LLM API key. `config.toml` is gitignored; copy
`config.example.toml` to create it. With no file present, sane defaults apply.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import ENV_HASH, KernelInterface
from .store import MockStore

DEFAULT_API_KEY_ENV = "TPPS_LLM_API_KEY"
DEFAULT_PATH = Path("config.toml")


@dataclass(frozen=True)
class LlmSettings:
    provider: str = "openai-compatible"
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_s: float = 60.0
    max_retries: int = 2

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass(frozen=True)
class KernelSettings:
    system: str = "lean4+mathlib"
    env_hash: str = ENV_HASH
    endpoint: str = ""  # empty = in-process mock backend


@dataclass(frozen=True)
class StoreSettings:
    owner: str = "tpps-prototype"
    backend: str = "mock"


@dataclass(frozen=True)
class AppSettings:
    log_level: str = "INFO"


@dataclass(frozen=True)
class Settings:
    llm: LlmSettings = field(default_factory=LlmSettings)
    kernel: KernelSettings = field(default_factory=KernelSettings)
    store: StoreSettings = field(default_factory=StoreSettings)
    app: AppSettings = field(default_factory=AppSettings)


_SECTIONS = {"llm": LlmSettings, "kernel": KernelSettings,
             "store": StoreSettings, "app": AppSettings}

_ENV_OVERRIDES = {
    "llm": {"provider": "TPPS_LLM_PROVIDER", "model": "TPPS_LLM_MODEL",
            "base_url": "TPPS_LLM_BASE_URL", "api_key_env": "TPPS_LLM_API_KEY_ENV",
            "timeout_s": "TPPS_LLM_TIMEOUT_S", "max_retries": "TPPS_LLM_MAX_RETRIES"},
    "kernel": {"system": "TPPS_KERNEL_SYSTEM", "env_hash": "TPPS_KERNEL_ENV_HASH",
               "endpoint": "TPPS_KERNEL_ENDPOINT"},
    "store": {"owner": "TPPS_STORE_OWNER", "backend": "TPPS_STORE_BACKEND"},
    "app": {"log_level": "TPPS_LOG_LEVEL"},
}


def _coerce(cls, values: dict, where: str):
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**{k: v for k, v in values.items() if k in known})


def load_config(path: str | Path | None = None) -> Settings:
    """Load settings; missing file falls back to defaults (env still applies)."""
    resolved = Path(path) if path is not None else DEFAULT_PATH
    data: dict = {}
    if resolved.is_file():
        with open(resolved, "rb") as f:
            data = tomllib.load(f)
    elif path is not None:
        raise FileNotFoundError(f"config file not found: {resolved}")
    unknown_sections = set(data) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(f"unknown config sections {sorted(unknown_sections)}")

    kwargs: dict = {}
    for section, cls in _SECTIONS.items():
        values = dict(data.get(section, {}))
        if not isinstance(values, dict):
            raise ValueError(f"section [{section}] must be a table")
        for key, env_var in _ENV_OVERRIDES[section].items():
            if env_var in os.environ:
                raw = os.environ[env_var]
                field_type = cls.__dataclass_fields__[key].type
                values[key] = (float(raw) if field_type == "float"
                               else int(raw) if field_type == "int" else raw)
        kwargs[section] = _coerce(cls, values, f"[{section}]")
    return Settings(**kwargs)


def make_kernel(settings: Settings) -> KernelInterface:
    """Kernel handle pinned to the configured environment hash."""
    return KernelInterface(env_hash=settings.kernel.env_hash)


def make_store(settings: Settings) -> MockStore:
    """Store handle for the configured owner/backend."""
    if settings.store.backend != "mock":
        raise ValueError(f"unknown store backend {settings.store.backend!r}")
    return MockStore(owner=settings.store.owner)
