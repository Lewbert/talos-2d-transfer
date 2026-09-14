"""Offline decode study for SmartCamApi raw buffers vs the ZEN color ground
truth. Reads %APPDATA%\\TALOS\\benchmark\\*.npy + ZEN-snap.czi and writes
PNGs + conclusions.json into %APPDATA%\\TALOS\\benchmark\\decode_study\\.

Usage:
    python tools/smartcam_decode_study.py            # full study
    python tools/smartcam_decode_study.py --input path.npy --size WxH
                                                     # idx2-style raw re-capture
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from talos.hal.devices.camera.smartcam_decode import (
    bayer_pattern_stats,
    decode_yuv420,
    decode_yuv420_planar,
    demosaic,
    detect_bayer_pattern,
    score_against_reference,
    trim_data_extent,
)
from talos.paths import get_appdata_dir

BENCH = Path(get_appdata_dir()) / "benchmark"
OUT = BENCH / "decode_study"

BAYER_HYPOTHESES = ("rggb", "bggr", "grbg", "gbrg")


def load_reference() -> np.ndarray | None:
    import czifile
    path = BENCH / "ZEN-snap.czi"
    if not path.is_file():
        return None
    arr = np.squeeze(czifile.imread(str(path)))
    return arr if arr.ndim == 3 else arr[0]


def save_png(img: np.ndarray, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(path)


def live_study(reference: np.ndarray | None) -> dict:
    """NV12 vs planar decode of the two live buffers + color-matrix fit."""
    results = {}
    for name in ("raw_frame.npy", "raw_post_zen.npy"):
        path = BENCH / name
        if not path.is_file():
            continue
        raw = trim_data_extent(np.load(path))
        yuv = decode_yuv420(raw)
        planar = decode_yuv420_planar(raw)
        save_png(yuv, f"{Path(name).stem}_yuv420.png")
        save_png(planar, f"{Path(name).stem}_i420.png")
        entry = {
            "yuv420_rgb_means": [round(float(yuv[:, :, i].mean()), 1) for i in range(3)],
            "i420_rgb_means": [round(float(planar[:, :, i].mean()), 1) for i in range(3)],
        }
        if reference is not None:
            entry["yuv420_score"] = score_against_reference(yuv, reference)
            entry["i420_score"] = score_against_reference(planar, reference)
        results[name] = entry

    # 3x4 affine color-matrix fit (raw NV12 -> ZEN reference) on the post-ZEN
    # buffer; applied to BOTH buffers to test generalization across exposure.
    matrix = None
    if reference is not None and (BENCH / "raw_post_zen.npy").is_file():
        src = decode_yuv420(trim_data_extent(np.load(BENCH / "raw_post_zen.npy"))).astype(np.float64)
        tgt = reference.astype(np.float64)
        x = np.hstack([src.reshape(-1, 3), np.ones((src.size // 3, 1))])
        m, *_ = np.linalg.lstsq(x, tgt.reshape(-1, 3), rcond=None)
        matrix = m.tolist()
        err = float(np.abs(np.clip(x @ m, 0, 255) - tgt.reshape(-1, 3)).mean())

        def apply(raw_path: Path, out_name: str) -> None:
            rgb = decode_yuv420(trim_data_extent(np.load(raw_path))).astype(np.float64)
            x2 = np.hstack([rgb.reshape(-1, 3), np.ones((rgb.size // 3, 1))])
            out = np.clip(x2 @ m, 0, 255).reshape(rgb.shape).astype(np.uint8)
            save_png(out, out_name)

        apply(BENCH / "raw_post_zen.npy", "post_zen_matrix.png")
        apply(BENCH / "raw_frame.npy", "pre_zen_matrix.png")
        results["color_matrix"] = {"matrix": matrix, "fit_mean_abs_err": round(err, 2)}
    return results


def idx2_study(raw: np.ndarray, size: tuple[int, int], reference: np.ndarray | None,
               tag: str = "idx2") -> dict:
    """Hypothesis gallery for the full-res raw buffer."""
    w, h = size
    n = w * h
    if raw.size < n:
        raw = np.pad(raw, (0, n - raw.size))
    mono = raw[:n].reshape(h, w)
    stats = bayer_pattern_stats(mono)
    pattern = detect_bayer_pattern(mono)
    print(f"  {tag}: shape={mono.shape} bayer-detect={pattern} "
          f"ratio_h={stats['ratio_h']:.2f} ratio_v={stats['ratio_v']:.2f}")

    gallery: list[tuple[str, np.ndarray]] = [("gray8", cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB))]
    for p in BAYER_HYPOTHESES:
        gallery.append((f"bayer8_{p}", demosaic(mono, p)))
    # 16-bit reinterpretations, downscaled to 8-bit
    for order in ("le", "be"):
        if raw.size >= 2 * n:
            u16 = raw[:2 * n].view(("<u2" if order == "le" else ">u2")).reshape(h, w)
            gallery.append((f"u16{order}", cv2.cvtColor((u16 >> 8).astype(np.uint8),
                                                        cv2.COLOR_GRAY2RGB)))
    # planar-16: low byte plane | high byte plane
    if raw.size >= 2 * n:
        lo = raw[0:2 * n:2].reshape(h, w)
        hi = raw[1:2 * n:2].reshape(h, w)
        gallery.append(("planar16_lo", cv2.cvtColor(lo, cv2.COLOR_GRAY2RGB)))
        gallery.append(("planar16_hi", cv2.cvtColor(hi, cv2.COLOR_GRAY2RGB)))
        gallery.append(("planar16_hi_lo", cv2.cvtColor(np.maximum(hi, lo),
                                                       cv2.COLOR_GRAY2RGB)))
    # row deinterlace: two fields
    even = mono[0::2].repeat(2, axis=0)[:h]
    odd = mono[1::2].repeat(2, axis=0)[:h]
    gallery.append(("field_even", cv2.cvtColor(even, cv2.COLOR_GRAY2RGB)))
    gallery.append(("field_odd", cv2.cvtColor(odd, cv2.COLOR_GRAY2RGB)))
    gallery.append(("field_diff", cv2.cvtColor(
        np.abs(even.astype(int) - odd.astype(int)).clip(0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB)))

    scores = {}
    for name, img in gallery:
        save_png(img, f"{tag}_{name}.png")
        scores[name] = score_against_reference(img, reference) if reference is not None else None
    ranked = sorted(
        [(name, s["luma_ncc"]) for name, s in scores.items() if s],
        key=lambda kv: kv[1], reverse=True)
    print(f"  {tag}: top matches (luma NCC): " +
          ", ".join(f"{name}={v:.3f}" for name, v in ranked[:5]))
    return {"size": [w, h], "bayer_stats": {k: round(v, 3) for k, v in stats.items()},
            "detected_pattern": pattern, "scores": scores, "ranked": ranked}


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartCamApi decode study")
    parser.add_argument("--input", help="single raw .npy to run the idx2 gallery on")
    parser.add_argument("--size", default="5472x3600", help="WxH for --input")
    args = parser.parse_args()

    reference = load_reference()
    conclusions: dict = {"reference": "ZEN-snap.czi" if reference is not None else None}

    if args.input:
        w, h = (int(v) for v in args.size.split("x"))
        raw = trim_data_extent(np.load(args.input))
        conclusions["gallery"] = idx2_study(raw, (w, h), reference,
                                            Path(args.input).stem)
    else:
        conclusions["live"] = live_study(reference)
        for name in ("zen_dll_idx2.npy",):
            path = BENCH / name
            if path.is_file():
                raw = trim_data_extent(np.load(path))
                conclusions["idx2"] = idx2_study(raw, (5472, 3600), reference)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "conclusions.json").write_text(json.dumps(conclusions, indent=2))
    print(f"wrote {OUT / 'conclusions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
