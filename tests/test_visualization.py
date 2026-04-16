"""
Smoke tests for visualization functions.

These tests use the Agg backend (no display) and verify that plot functions
execute without raising exceptions and produce valid figure objects.
"""

import os
import tempfile
import matplotlib
matplotlib.use("Agg")

import pytest
import matplotlib.pyplot as plt

from tqnn.visualization.static import (
    plot_braiding_pattern,
    plot_large_braiding_pattern,
    plot_topological_charge_flow,
    plot_large_topological_charge_flow,
    plot_logical_gate_structure,
    plot_large_logical_gate_structure,
)


@pytest.fixture
def tmpDir(tmp_path):
    """Provide a temporary directory for plot output."""
    return str(tmp_path)


class TestStaticPlots:
    @pytest.mark.parametrize("plotFunc", [
        plot_braiding_pattern,
        plot_large_braiding_pattern,
        plot_topological_charge_flow,
        plot_large_topological_charge_flow,
        plot_logical_gate_structure,
        plot_large_logical_gate_structure,
    ])
    def test_plot_creates_file(self, tmpDir: str, plotFunc) -> None:
        savePath = os.path.join(tmpDir, f"{plotFunc.__name__}.png")
        plotFunc(savePath)
        assert os.path.isfile(savePath)
        assert os.path.getsize(savePath) > 0
        plt.close("all")
