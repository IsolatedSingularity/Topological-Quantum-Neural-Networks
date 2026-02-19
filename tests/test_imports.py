"""
Smoke tests: verify all visualization modules import cleanly.

These are intentionally lightweight -- they confirm the dependency chain is
intact without running any expensive plotting or animation logic.
"""

import sys
import os
import importlib

import pytest

# Ensure source directories are on the path
REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'Code', 'Static Visualization'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Code', 'Image Classification'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'Code', 'Cobordism Viewer'))


# Use Agg backend so tests don't open windows
import matplotlib
matplotlib.use('Agg')


MODULES = [
    'tqnn_helpers',
    'static_visualizations',
    'animated_visualizations',
    'tqnn_sandbox',
    'interactive_tqnn_classifier',
    'cobordism_evolution_viewer',
]


@pytest.mark.parametrize('module_name', MODULES)
def test_module_imports(module_name: str) -> None:
    """Each visualization module should import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
