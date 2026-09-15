"""Computer vision pipeline (focus, scan, edge, flakes, calibration)."""

from talos.cv.focus_metric import (  # noqa: F401
    METRICS,
    brenner,
    laplacian_variance,
    tenengrad,
)
