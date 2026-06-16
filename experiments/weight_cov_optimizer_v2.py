"""Back-compat shim.

The implementation now lives at the repo root in ``spectral_filter.py`` (the
clean, canonical entry point for external users). This module re-exports it so
the existing experiment scripts that do ``from weight_cov_optimizer_v2 import
WeightCovarianceFilterV2`` keep working unchanged.

New code should prefer:  ``from spectral_filter import SpectralGradientFilter``
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spectral_filter import SpectralGradientFilter, WeightCovarianceFilterV2  # noqa: E402,F401
