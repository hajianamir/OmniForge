"""Verify the OmniForge distribution from a directory outside its checkout."""

from __future__ import annotations

from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import assert_type

import omniforge
from omniforge.config import ConfigLoader
from omniforge.config.settings import OmniForgeSettings
from omniforge.domains import BaseDomainPlugin, default_domain_registry


def main() -> None:
    installed = distribution("omniforge")
    package = Path(omniforge.__file__).resolve().parent
    assert package == Path(str(installed.locate_file("omniforge"))).resolve(), "Run this check against an installed wheel."
    assert package.joinpath("py.typed").is_file(), "The installed distribution is missing its PEP 561 marker."
    assets = sorted(package.joinpath("config", "domain_defaults").glob("*.yaml"))
    assert len(assets) == 18, f"Expected 18 built-in defaults, found {len(assets)}."
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for asset in assets:
            plugin = assert_type(default_domain_registry.get(asset.stem), BaseDomainPlugin)
            config = plugin.load_config(root / "domains")
            assert config["domain"]["name"] == asset.stem, asset.name
            settings = assert_type(ConfigLoader(root).compose(domain=asset.stem), OmniForgeSettings)
            assert settings.domain_overrides == config, asset.name

        override = root / "domains"
        override.mkdir()
        (override / "energy.yaml").write_text("domain:\n  name: local_energy\n", encoding="utf-8")
        plugin = default_domain_registry.get("energy")
        assert plugin.load_config(override) == {"domain": {"name": "local_energy"}}
        assert ConfigLoader(root).compose(domain="energy").domain_overrides == {"domain": {"name": "local_energy"}}
    print(f"Installed SDK verified: {len(assets)} domain defaults, local overrides, and py.typed.")


if __name__ == "__main__":
    main()
