from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from risk import empirical_risk
from risk.native import native_risk


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("optional C++20 compiler unavailable")
    output = tmp_path_factory.mktemp("native") / "libquantumrisk.so"
    source = Path(__file__).parents[1] / "native/risk.cpp"
    subprocess.run([compiler, "-std=c++20", "-O2", "-shared", "-fPIC", str(source), "-o", str(output)], check=True, capture_output=True)
    return output


@pytest.mark.parametrize("alpha", [.5, .9, .95, .99])
def test_native_matches_python_reference(library, alpha):
    losses = np.random.default_rng(17).normal(size=1000)
    actual = native_risk(losses, alpha, library)
    expected = empirical_risk(losses, alpha)
    assert actual["var"] == pytest.approx(expected["var"])
    assert actual["cvar"] == pytest.approx(expected["cvar"])


def test_native_fractional_atom_and_invalid_input(library):
    assert native_risk([0, 1, 2, 3], .5, library)["cvar"] == 2.5
    with pytest.raises(ValueError):
        native_risk([np.inf], .99, library)
