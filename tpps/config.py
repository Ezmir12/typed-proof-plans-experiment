"""Setup configuration: layered overrides with documented precedence.

Precedence: exported environment variables > local dotenv file >
config file > built-in defaults. Secrets never live in files: only the
name of the variable holding the LLM API key is configured, never the key.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from tpps.adapters.kernel import ENV_HASH, KernelInterface
from tpps.adapters.store import MockStore

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
    # Resolved at load from the named env var (dotenv counts); never from a file.
    api_key: str = ""


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
    extractor: str = "llm"  # llm | demo | hooks (explicit hooks win regardless)
    extract_retries: int = 2


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
    "app": {"log_level": "TPPS_LOG_LEVEL", "extractor": "TPPS_EXTRACTOR",
            "extract_retries": "TPPS_EXTRACT_RETRIES"},
}


def _coerce(cls, values: dict, where: str):
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    return cls(**{k: v for k, v in values.items() if k in known})


def _coerce_value(raw: str, field_type) -> object:
    if field_type == "float":
        return float(raw)
    if field_type == "int":
        return int(raw)
    return raw


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser. Only TPPS_* keys are honored; the process
    environment is never mutated."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key.startswith("TPPS_"):
            values[key] = val
    return values


def load_config(path: str | Path | None = None) -> Settings:
    """Load settings. Precedence: exported env vars > dotenv file >
    config file > defaults. The dotenv sidecar is looked up next to the
    given path (or the working directory when no path is given). Missing
    files fall back silently."""
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
    dotenv_path = (resolved.parent if path is not None else Path(".")) / ".env"
    dotenv = _parse_dotenv(dotenv_path) if dotenv_path.is_file() else {}

    kwargs: dict = {}
    for section, cls in _SECTIONS.items():
        values = dict(data.get(section, {}))
        if not isinstance(values, dict):
            raise ValueError(f"section [{section}] must be a table")
        if section == "llm" and "api_key" in values:
            raise ValueError("[llm]: api_key must come from the environment, "
                             "never the config file")
        for key, env_var in _ENV_OVERRIDES[section].items():
            raw = os.environ.get(env_var, dotenv.get(env_var))
            if raw is not None:
                field_type = cls.__dataclass_fields__[key].type
                values[key] = _coerce_value(raw, field_type)
        kwargs[section] = _coerce(cls, values, f"[{section}]")
    settings = Settings(**kwargs)
    key_name = settings.llm.api_key_env
    key = os.environ.get(key_name, dotenv.get(key_name, ""))
    return replace(settings, llm=replace(settings.llm, api_key=key))


def make_kernel(settings: Settings) -> KernelInterface:
    """Kernel handle pinned to the configured environment hash."""
    return KernelInterface(env_hash=settings.kernel.env_hash)


def make_store(settings: Settings) -> MockStore:
    """Store handle for the configured owner/backend."""
    if settings.store.backend != "mock":
        raise ValueError(f"unknown store backend {settings.store.backend!r}")
    return MockStore(owner=settings.store.owner)
