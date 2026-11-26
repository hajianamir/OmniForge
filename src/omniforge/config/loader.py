"""OmegaConf/Hydra-style layered YAML configuration composer for OmniForge."""

from __future__ import annotations

import copy
import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import EnvSettingsSource

from omniforge.config.settings import OmniForgeSettings
from omniforge.core.exceptions import ConfigurationError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)
_PLATFORM_FIELDS = ("environment", "debug", "random_seed", "default_tenant_id")


def load_packaged_domain_config(slug: str) -> dict[str, Any] | None:
    """Read a built-in domain default, including from an installed wheel."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
        raise ConfigurationError("Invalid domain name for a packaged configuration.")
    resource = files(__package__).joinpath("domain_defaults", f"{slug}.yaml")
    if not resource.is_file():
        return None
    content = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ConfigurationError(f"The packaged configuration for '{slug}' must be a YAML mapping.")
    return dict(content)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries without mutating inputs.

    Args:
        base: Lower-precedence dictionary.
        override: Higher-precedence dictionary.

    Returns:
        Newly merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_dotlist_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse Hydra-style ``section.key=value`` CLI overrides into a nested dictionary.

    Args:
        overrides: List of strings like ``["automl.max_trials=50", "debug=true"]``.

    Returns:
        Nested dictionary representation.
    """
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ConfigurationError(
                f"Invalid configuration override '{item}'. Expected 'key=value' syntax.",
                suggestions=["Use dot-notation overrides such as 'automl.max_trials=50'."],
            )
        raw_key, raw_val = item.split("=", 1)
        keys = [part.strip() for part in raw_key.split(".") if part.strip()]
        if not keys:
            raise ConfigurationError(f"Empty key in configuration override '{item}'.")
        value = yaml.safe_load(raw_val)
        cursor = parsed
        for part in keys[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[keys[-1]] = value
    return parsed


def _resolve_env_interpolation(obj: Any) -> Any:
    """Resolve ``${ENV_VAR:default}`` placeholders inside YAML configs."""
    if isinstance(obj, dict):
        return {k: _resolve_env_interpolation(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_interpolation(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        expr = obj[2:-1]
        if ":" in expr:
            env_name, default_val = expr.split(":", 1)
            return os.getenv(env_name.strip(), default_val.strip())
        return os.getenv(expr.strip(), obj)
    return obj


def _normalize_platform_layer(layer: dict[str, Any]) -> dict[str, Any]:
    """Map the YAML platform section to settings field names within its own layer."""
    layer = dict(layer)
    platform = layer.pop("platform", {})
    if isinstance(platform, dict):
        for field in _PLATFORM_FIELDS:
            if field in platform:
                layer.setdefault(field, platform[field])
    return layer


class ConfigLoader:
    """Composes OmniForge configuration from base YAML, domain YAML, custom overlays, and dotlist overrides."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        if config_dir is not None:
            self.config_dir = Path(config_dir)
        else:
            # Locate configs/ directory relative to package root or cwd
            pkg_root = Path(__file__).resolve().parents[3]
            candidate = pkg_root / "configs"
            self.config_dir = candidate if candidate.exists() else Path.cwd() / "configs"

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """Load and interpolate a single YAML configuration file."""
        file_path = Path(path)
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                suggestions=["Verify config_dir or pass an explicit YAML file path."],
            )
        try:
            content = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
            if not isinstance(content, dict):
                raise ConfigurationError(
                    f"YAML root in {file_path} must be a mapping, got {type(content).__name__}."
                )
            return dict(_resolve_env_interpolation(content))
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to parse YAML configuration file {file_path}.",
                cause=exc,
            ) from exc

    def compose(
        self,
        *,
        base_file: str | Path | None = None,
        domain: str | None = None,
        extra_files: list[str | Path] | None = None,
        overrides: list[str] | None = None,
    ) -> OmniForgeSettings:
        """Compose settings with CLI overrides above environment variables and YAML layers.

        Args:
            base_file: Optional path to base YAML (defaults to ``configs/default.yaml``).
            domain: Optional domain name (configured YAML overrides packaged defaults).
            extra_files: Optional list of additional overlay YAML paths.
            overrides: Optional Hydra-style ``key.subkey=value`` overrides.

        Returns:
            Validated :class:`OmniForgeSettings`.
        """
        merged: dict[str, Any] = {}
        default_yaml = Path(base_file) if base_file else (self.config_dir / "default.yaml")
        if default_yaml.exists():
            merged = deep_merge(merged, _normalize_platform_layer(self.load_yaml(default_yaml)))

        domain_cfg: dict[str, Any] = {}
        if domain:
            slug = domain.lower()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug):
                raise ConfigurationError("Domain names must use letters, digits, underscores, or hyphens.")
            domain_root = (self.config_dir / "domains").resolve()
            domain_path = (domain_root / f"{slug}.yaml").resolve()
            if not domain_path.is_relative_to(domain_root):
                raise ConfigurationError("Domain configuration must be inside the configured domains directory.")
            if domain_path.exists():
                domain_cfg = self.load_yaml(domain_path)
            else:
                packaged = load_packaged_domain_config(slug)
                if packaged is None:
                    raise ConfigurationError(
                        f"No configuration is available for domain '{slug}'.",
                        suggestions=["Run 'omniforge domains list' to view available domains."],
                    )
                domain_cfg = _resolve_env_interpolation(packaged)

        if extra_files:
            for extra in extra_files:
                merged = deep_merge(merged, _normalize_platform_layer(self.load_yaml(extra)))

        if domain_cfg:
            merged["domain_overrides"] = domain_cfg

        cli = _normalize_platform_layer(parse_dotlist_overrides(overrides or []))
        try:
            env = EnvSettingsSource(OmniForgeSettings)()
            settings = OmniForgeSettings.model_validate(deep_merge(deep_merge(merged, env), cli))
            logger.info(
                "configuration_composed",
                environment=settings.environment,
                domain=domain or "generic",
            )
            return settings
        except Exception as exc:
            raise ConfigurationError(
                "Composed configuration failed schema validation.",
                cause=exc,
                suggestions=["Review YAML types and dotlist override syntax."],
            ) from exc


def load_config(
    *,
    domain: str | None = None,
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
) -> OmniForgeSettings:
    """Convenience function to compose and return :class:`OmniForgeSettings`."""
    loader = ConfigLoader()
    return loader.compose(base_file=config_file, domain=domain, overrides=overrides)


load_settings = load_config
