"""Probe the configured in-container API or UI listener."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    ports = sys.argv[1:] or [os.getenv("OMNIFORGE_API_PORT", "8000"), os.getenv("OMNIFORGE_UI_PORT", "8501")]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in ports:
        try:
            with opener.open(f"http://127.0.0.1:{port}/api/v1/health", timeout=4) as response:
                if response.status == 200:
                    return 0
        except (urllib.error.URLError, TimeoutError):
            continue
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
