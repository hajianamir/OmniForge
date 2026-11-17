"""OmniForge Industry Domain Plugins (18 enterprise verticals + extensible SDK)."""

from omniforge.domains import (
    agriculture,
    bioinformatics,
    climate,
    cybersecurity,
    education,
    energy,
    finance,
    government,
    healthcare,
    hr,
    manufacturing,
    materials,
    media,
    real_estate,
    retail,
    sports,
    telecom,
    transportation,
)
from omniforge.domains.plugin_sdk import (
    BaseDomainPlugin,
    DomainMetadata,
    DomainPluginRegistry,
    default_domain_registry,
)

_BUILTIN_PLUGINS: list[BaseDomainPlugin] = [
    finance.plugin,
    energy.plugin,
    healthcare.plugin,
    manufacturing.plugin,
    retail.plugin,
    transportation.plugin,
    agriculture.plugin,
    climate.plugin,
    telecom.plugin,
    hr.plugin,
    education.plugin,
    real_estate.plugin,
    sports.plugin,
    cybersecurity.plugin,
    government.plugin,
    bioinformatics.plugin,
    materials.plugin,
    media.plugin,
]

for _p in _BUILTIN_PLUGINS:
    default_domain_registry.register(_p)

__all__ = [
    "BaseDomainPlugin",
    "DomainMetadata",
    "DomainPluginRegistry",
    "default_domain_registry",
]
