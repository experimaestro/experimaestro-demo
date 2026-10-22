"""Minimal launcher config for CI: everything runs locally.

Pointed at via ``run-experiment --xpm-config-dir tests/xpm-config`` so the demo
does not depend on a user-level ``~/.config/experimaestro/launchers.py``. Any
requirement resolves to a local DirectLauncher (localhost).
"""

from experimaestro.connectors.local import LocalConnector
from experimaestro.launchers.direct import DirectLauncher


def find_launcher(requirements, tags: set[str] = set()):
    return DirectLauncher(LocalConnector.instance())
