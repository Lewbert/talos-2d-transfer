"""px↔stage calibration: phase-correlation shift estimation and 2×2
jacobian fitting from commanded stage moves."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# The CANONICAL calibration reference: µm per SENSOR pixel at the 4K
# sensor width. Every stored value (wizard, Labscope import, the
# pixel-pitch fallback) is expressed per 4K-sensor pixel; consumers
# working on other frame widths (the 1080p live view) scale by
# SENSOR_WIDTH_PX / frame_width — a 1080p pixel covers 2× the µm.
SENSOR_WIDTH_PX = 3840
#: The same sensor's height. The field of view is um_per_px × these two,
#: which is why the FOV does not change when the frame size does.
SENSOR_HEIGHT_PX = 2160


@dataclass
class JacobianSample:
    """One measurement: commanded stage move (µm) + measured pixel shift."""
    dx_um: float
    dy_um: float
    dx_px: float
    dy_px: float


@dataclass
class JacobianFit:
    jacobian: list[list[float]]      # [[jxx, jxy], [jyx, jyy]] px per µm
    residual_px: float
    um_per_px_x: float               # inverse of the dominant diagonal terms
    um_per_px_y: float
    samples: list[JacobianSample] = field(default_factory=list)


def phase_correlation_shift(img_before: np.ndarray,
                            img_after: np.ndarray,
                            roi: tuple[int, int, int, int] | None = None
                            ) -> tuple[float, float]:
    """Sub-pixel shift (dx, dy) in pixels via cv2.phaseCorrelate.

    A positive dx means the scene moved RIGHT in the after image, i.e.
    the stage moved LEFT. Callers account for the sign convention.
    """
    a = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)
    if roi is not None:
        x, y, w, h = roi
        a = a[y:y + h, x:x + w]
        b = b[y:y + h, x:x + w]
    a = np.float32(a)
    b = np.float32(b)
    # Hann window to suppress FFT edge artifacts.
    win = np.hanning(b.shape[0])[:, None] * np.hanning(b.shape[1])[None, :]
    (dx, dy), response = cv2.phaseCorrelate(a * win, b * win)
    return float(dx), float(dy)


def estimate_jacobian(moves: list[tuple[float, float]],
                      shifts: list[tuple[float, float]]) -> JacobianFit:
    """Least-squares fit of [px] = J @ [um].

    ``moves``: commanded stage moves (dx_um, dy_um).
    ``shifts``: measured pixel shifts with the OPPOSITE sign convention
    (scene moves +dx px when the stage moves -x µm). The fit absorbs the
    sign so callers can pass raw phase-correlation outputs.
    """
    if len(moves) < 2:
        raise ValueError("Need at least 2 moves for a jacobian fit")
    A = np.array(moves, dtype=float)
    B = np.array(shifts, dtype=float)
    jacobian, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    predicted = A @ jacobian
    residual = float(np.sqrt(np.mean((B - predicted) ** 2)))
    j = jacobian.tolist()
    # µm/px from the inverse jacobian (diagonal when near-axis-aligned).
    jinv = np.linalg.pinv(np.array(j))
    um_per_px_x = float(abs(jinv[0, 0])) if abs(jinv[0, 0]) > 1e-9 else 0.0
    um_per_px_y = float(abs(jinv[1, 1])) if abs(jinv[1, 1]) > 1e-9 else 0.0
    samples = [JacobianSample(m[0], m[1], s[0], s[1])
               for m, s in zip(moves, shifts)]
    return JacobianFit(jacobian=j, residual_px=residual,
                       um_per_px_x=um_per_px_x, um_per_px_y=um_per_px_y,
                       samples=samples)


def run_px_um_wizard(stage, camera, move_um: float = 50.0,
                     n_moves: int = 4) -> JacobianFit:
    """Wizard steps: capture → move +Δ → capture → phase-correlate, along
    +X, +Y, −X, −Y. Requires a feature-rich region in view. The caller is
    responsible for safety (bounded moves, abort handling).
    """
    from talos.hal.base import StageSpeed

    moves: list[tuple[float, float]] = []
    shifts: list[tuple[float, float]] = []
    sequence = [(move_um, 0.0), (0.0, move_um), (-move_um, 0.0), (0.0, -move_um)]
    frame = camera.fetch(timeout_ms=3000.0)
    if frame is None:
        raise RuntimeError("No camera frame for calibration")
    roi = (frame.shape[1] // 4, frame.shape[0] // 4,
           frame.shape[1] // 2, frame.shape[0] // 2)
    for i, (dx, dy) in enumerate(sequence[:n_moves]):
        stage.move_rel_um(dx, dy, speed=StageSpeed.SLOW)
        stage.wait_idle(timeout_s=60.0)
        after = camera.fetch(timeout_ms=3000.0)
        if after is None:
            raise RuntimeError(f"No camera frame after move {i + 1}")
        shift = phase_correlation_shift(frame, after, roi)
        shifts.append(shift)
        moves.append((dx, dy))
    fit = estimate_jacobian(moves, shifts)
    # The measurement ran on LIVE frames; the stored value is canonical
    # (per 4K-sensor pixel) — a 1080p live frame halves the µm/px.
    sensor_scale = frame.shape[1] / float(SENSOR_WIDTH_PX)
    fit.um_per_px_x *= sensor_scale
    fit.um_per_px_y *= sensor_scale
    return fit
