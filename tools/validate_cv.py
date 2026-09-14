"""CV validation on recorded images: flake detection + annotated output.

Usage:
    python tools/validate_cv.py --image <path> --um-per-px 0.2
    python tools/validate_cv.py --dataset <dir> --um-per-px 0.2
    python tools/validate_cv.py --dataset <dir> --edge-only   # wafer-edge fit

Writes <name>_annotated.png and report.json into the output dir (default:
the image's directory under a _talos_cv subfolder).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from talos.cv.edge import detect_wafer_edge  # noqa: E402
from talos.cv.flakes import ClassicFlakeDetector, FlakeConfig  # noqa: E402
from talos.models import ObjectiveCalibration  # noqa: E402

_IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")

# Axiocam 208 pixel pitch 2.0 µm → µm/px = 2.0 / magnification.
_UM_PER_PX_BY_MAG = {"5x": 0.40, "10x": 0.20, "20x": 0.10, "50x": 0.04, "100x": 0.02}

# Magnification-appropriate minimum flake area (µm²) — at higher
# magnification flakes are physically smaller.
_MIN_AREA_BY_MAG = {"5x": 30.0, "10x": 30.0, "20x": 10.0, "50x": 4.0, "100x": 2.0}


def _mag_from_name(name: str) -> str | None:
    import re

    match = re.search(r"(\d{2,3})x", name, re.IGNORECASE)
    return match.group(0).lower() if match else None


def resolve_um_per_px(path: Path, override: float | None) -> float:
    if override is not None:
        return override
    mag = _mag_from_name(path.name)
    if mag and mag in _UM_PER_PX_BY_MAG:
        return _UM_PER_PX_BY_MAG[mag]
    return 0.2  # 10x default


def validate_image(path: Path, um_per_px: float, cfg: FlakeConfig,
                   out_dir: Path, edge_only: bool = False) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"image": path.name, "error": "unreadable"}
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"image": path.name, "shape": list(img.shape)}
    annotated = img.copy()

    if not edge_only:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        calib = ObjectiveCalibration(objective_id=0,
                                     um_per_px_x=um_per_px, um_per_px_y=um_per_px)
        flakes = ClassicFlakeDetector().find(rgb, calib, cfg)
        report["n_flakes"] = len(flakes)
        report["flakes"] = [
            {"area_um2": round(f.area_um2, 2), "x_px": round(f.x_px, 1),
             "y_px": round(f.y_px, 1), "score": round(f.score, 1),
             "bbox": list(f.bbox)}
            for f in flakes]
        for f in flakes:
            x, y, w, h = f.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(annotated, f"{f.area_um2:.0f}um2", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    edge = detect_wafer_edge(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    report["edge"] = {"method": edge.method, "confidence": round(edge.confidence, 3),
                      "center_px": edge.center_px, "size_px": edge.size_px}
    if edge.method == "rectangle" and edge.corners:
        import numpy as np

        pts = np.array(edge.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], True, (255, 160, 0), 3)

    out_img = out_dir / (path.stem + "_annotated.png")
    cv2.imwrite(str(out_img), annotated)
    report["annotated"] = str(out_img)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CV on recorded images")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--um-per-px", type=float, default=None,
                        help="override µm/px (default: auto from magnification in filename)")
    parser.add_argument("--min-area-um2", type=float, default=None,
                        help="override min flake area µm² (default: mag-appropriate)")
    parser.add_argument("--max-area-um2", type=float, default=100000.0)
    parser.add_argument("--edge-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    images: list[Path] = []
    if args.image:
        images = [args.image]
    elif args.dataset:
        images = sorted(p for p in args.dataset.iterdir()
                        if p.suffix.lower() in _IMAGE_EXTS)
    else:
        parser.error("provide --image or --dataset")

    out_dir = args.out or (images[0].parent / "_talos_cv")
    reports = []
    for path in images:
        um_per_px = resolve_um_per_px(path, args.um_per_px)
        mag = _mag_from_name(path.name)
        min_area = (args.min_area_um2 if args.min_area_um2 is not None
                    else _MIN_AREA_BY_MAG.get(mag, 30.0))
        cfg = FlakeConfig(min_area_um2=min_area, max_area_um2=args.max_area_um2)
        print(f"validating {path.name} ({um_per_px} µm/px, min {min_area} µm²)…")
        reports.append(validate_image(path, um_per_px, cfg, out_dir,
                                      edge_only=args.edge_only))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8")
    for r in reports:
        if "error" in r:
            print(f"  {r['image']}: {r['error']}")
        else:
            print(f"  {r['image']}: {r.get('n_flakes', '-')} flakes, "
                  f"edge={r['edge']['method']} conf={r['edge']['confidence']:.2f}")
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
