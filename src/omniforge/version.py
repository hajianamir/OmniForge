"""Version and build metadata for OmniForge (Enterprise Analytics & AI Workbench)."""

from __future__ import annotations

__version__ = "2.0.0"
__edition__ = "Enterprise Analytics & AI Workbench"
__platform_name__ = "OmniForge"
__short_name__ = "OmniForge"
__api_version__ = "v2"
__schema_version__ = "2.0"


def get_version_info() -> dict[str, str]:
    """Return structured version and platform metadata.

    Returns:
        Dictionary containing version, edition, platform name, API version,
        and schema version.
    """
    return {
        "name": __platform_name__,
        "short_name": __short_name__,
        "version": __version__,
        "edition": __edition__,
        "api_version": __api_version__,
        "schema_version": __schema_version__,
    }
