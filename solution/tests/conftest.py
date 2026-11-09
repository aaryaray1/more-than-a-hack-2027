import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if path not in sys.path:
        sys.path.insert(0, path)

from games import asset_paths  # noqa: E402


def _have_data():
    paths = list(asset_paths().values()) + list(asset_paths(source="broadcast").values())
    return all(os.path.isfile(p) for p in paths if not p.endswith(".mp4"))


needs_data = pytest.mark.skipif(not _have_data(), reason="match data not unpacked into data/")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: streams the full ATD (1-2 min)")
