"""
Robust pointcloud generation entrypoint.

This module exposes a single function:

    generate_pointcloud(frames_dir: str) -> str

Behavior:
- Lazy-loads heavy libraries (torch, mapanything, plyfile, numpy) inside the function so
  importing this module is lightweight.
- Attempts to run the MapAnything model if available and the environment supports it.
- If any import or runtime error occurs (missing deps, ABI mismatch, OOM, etc.), writes a
  clear debug file and falls back to writing a small placeholder pointcloud (.ply).
- Always returns the absolute path to the generated .ply file.

Notes:
- The function intentionally accepts only `frames_dir` to keep the interface simple.
- If you want progress reporting, add an external mechanism (socket events, status file)
  instead of coupling this function to the web stack.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional, Tuple


def _write_ply_ascii(output_path: str, points, colors=None) -> None:
    """
    Write an ASCII PLY file.

    Args:
        output_path: target file path.
        points: iterable or numpy array-like of shape (N,3).
        colors: optional iterable or ndarray of shape (N,3) with 0-255 ints.
    """
    # Avoid requiring numpy at module import time. Convert using duck-typing.
    pts = list(points)
    n = len(pts)
    if n == 0:
        raise ValueError("No points available to write PLY.")

    # Normalize colors
    if colors is None:
        cols = [(255, 255, 255)] * n
    else:
        cols = list(colors)
        if len(cols) != n:
            # Try to broadcast or truncate
            if len(cols) == 1:
                cols = cols * n
            else:
                cols = (cols[:n] + [(255, 255, 255)] * n)[:n]

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(output_path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        for (x, y, z), (r, g, b) in zip(pts, cols):
            fh.write(
                f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}\n"
            )


def _create_placeholder_pointcloud(output_path: str, frames_count: int = 1) -> int:
    """
    Create a small placeholder PLY to ensure downstream logic receives a file.

    Returns:
        number of points written
    """
    # Make a small grid sized by frames_count (keeps file small)
    grid_size = max(2, min(16, int(max(2, frames_count**0.5) * 2)))
    pts = []
    cols = []
    xs = [(-1.0 + 2.0 * i / (grid_size - 1)) for i in range(grid_size)]
    ys = [(-1.0 + 2.0 * j / (grid_size - 1)) for j in range(grid_size)]
    zs = [0.0, 1.0]  # two layers to add depth
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                pts.append((x, y, z))
                r = int(255 * (i / max(1, len(xs) - 1)))
                g = int(255 * (j / max(1, len(ys) - 1)))
                b = int(255 * (k / max(1, len(zs) - 1)))
                cols.append((r, g, b))
    _write_ply_ascii(output_path, pts, cols)
    return len(pts)


def generate_pointcloud(frames_dir: str) -> str:
    """
    Generate a pointcloud .ply from images in `frames_dir`.

    Args:
        frames_dir: directory path containing frame images.

    Returns:
        absolute path to the generated .ply file.

    Behavior:
    - Attempts to run MapAnything model if available.
    - On any failure, writes a fallback PLY and a debug file in the results directory.
    """
    if not isinstance(frames_dir, str):
        raise TypeError("frames_dir must be a string path")

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")

    results_dir = os.path.join(frames_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_ply = os.path.join(results_dir, "pointcloud.ply")
    debug_txt = os.path.join(results_dir, "pointcloud_generation_debug.txt")

    # Try to import heavy deps lazily and run model inference.
    try:
        # Local imports to avoid import-time side effects
        import numpy as _np  # type: ignore
        import torch  # type: ignore

        # Import mapanything from either installed package or local vendored libs.
        # The import itself may trigger further imports (cv2, etc.) and can fail when
        # the environment has ABI mismatches. We'll catch any exception below.
        from mapanything.models import MapAnything  # type: ignore
        from mapanything.utils.image import load_images  # type: ignore

        # Prepare device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model (may download weights on first run)
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)

        # Load images as views (mapanything utility does preprocessing)
        views = load_images(frames_dir)

        # Configure AMP/precision defensively
        use_amp = True if device == "cuda" else False
        amp_kwargs = {}
        if use_amp:
            amp_kwargs["use_amp"] = True
            # Use bf16 only on CUDA devices where supported; fall back if needed
            amp_kwargs["amp_dtype"] = "bf16"

        # Run inference - try full argset and fall back to simpler call if needed
        try:
            predictions = model.infer(
                views,
                memory_efficient_inference=False,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
                **amp_kwargs,
            )
        except TypeError:
            # model.infer may not accept optional amp args in older/newer versions
            predictions = model.infer(
                views,
                memory_efficient_inference=False,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )

        # Collect points + colors
        all_points = []
        all_colors = []
        total_points = 0

        for pred in predictions:
            # Expected keys: pts3d, img_no_norm (colors), mask
            pts3d = pred.get("pts3d", None)
            img = pred.get("img_no_norm", None)
            mask = pred.get("mask", None)

            if pts3d is None or img is None or mask is None:
                continue

            # Convert tensors to numpy if necessary
            if hasattr(pts3d, "detach"):
                pts3d = pts3d.detach().cpu().numpy()
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()

            # Normalize dims: support (H,W,3) or (B,H,W,3)
            if pts3d.ndim == 3:
                pts3d = pts3d[None, ...]
            if img.ndim == 3:
                img = img[None, ...]
            # mask can be (H,W), (H,W,1), (B,H,W), (B,H,W,1)
            if mask.ndim == 2:
                mask = mask[None, ..., None]
            elif mask.ndim == 3:
                # If last dim is channel, move to consistent shape
                if mask.shape[-1] == 1:
                    mask = mask[None, ...]
                else:
                    mask = mask[..., None]

            B = int(pts3d.shape[0])
            for b in range(B):
                p = pts3d[b].reshape(-1, 3)
                c = img[b].reshape(-1, 3)
                m = mask[b].reshape(-1)
                # mask may be float confidence map or boolean; treat >0 as valid
                valid = m > 0
                if not _np.any(valid):
                    continue
                p_sel = p[valid]
                c_sel = c[valid]
                # convert colors 0..1 -> 0..255 if floats
                if _np.issubdtype(c_sel.dtype, _np.floating):
                    c_sel = _np.clip(c_sel * 255.0, 0, 255).astype(_np.uint8)
                else:
                    c_sel = _np.clip(c_sel, 0, 255).astype(_np.uint8)
                all_points.append(p_sel)
                all_colors.append(c_sel)
                total_points += p_sel.shape[0]

        if not all_points:
            raise RuntimeError("Model produced no valid points")

        points = _np.vstack(all_points)
        colors = _np.vstack(all_colors)

        # Optionally downsample if extremely large (protective)
        max_points = 5_000_000
        if points.shape[0] > max_points:
            idx = _np.random.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        # Try using plyfile if available for compact binary PLY; otherwise write ASCII
        try:
            from plyfile import PlyData, PlyElement  # type: ignore

            vertex_dtype = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
            vertex_data = _np.empty(points.shape[0], dtype=vertex_dtype)
            vertex_data["x"] = points[:, 0].astype(_np.float32)
            vertex_data["y"] = points[:, 1].astype(_np.float32)
            vertex_data["z"] = points[:, 2].astype(_np.float32)
            vertex_data["red"] = colors[:, 0]
            vertex_data["green"] = colors[:, 1]
            vertex_data["blue"] = colors[:, 2]
            el = PlyElement.describe(vertex_data, "vertex")
            PlyData([el], text=False).write(output_ply)
        except Exception:
            # Fallback to plain ASCII writer (no dependency)
            _write_ply_ascii(output_ply, points.tolist(), colors.tolist())

        # Optionally write a small stats file
        try:
            with open(os.path.join(results_dir, "pointcloud_stats.txt"), "w") as st:
                st.write(f"frames_used: {len(os.listdir(frames_dir))}\n")
                st.write(f"points_written: {int(points.shape[0])}\n")
        except Exception:
            pass

        return os.path.abspath(output_ply)

    except Exception as exc:
        # Record debug info and create fallback PLY
        try:
            with open(debug_txt, "w") as fh:
                fh.write("Pointcloud generation failed with exception:\n")
                fh.write(traceback.format_exc())
        except Exception:
            # ignore debug file write errors
            pass

        # Fallback: create a placeholder PLY
        try:
            # Try to estimate frames count
            try:
                frames_count = len(
                    [
                        f
                        for f in os.listdir(frames_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))
                    ]
                )
            except Exception:
                frames_count = 1
            _create_placeholder_pointcloud(output_ply, frames_count)
            # Write a minimal stats file noting fallback
            try:
                with open(os.path.join(results_dir, "pointcloud_stats.txt"), "w") as st:
                    st.write("fallback: true\n")
                    st.write(f"frames_found: {frames_count}\n")
            except Exception:
                pass
            return os.path.abspath(output_ply)
        except Exception as final_exc:
            # If even fallback writing fails, re-raise the original exception for visibility
            raise RuntimeError("Failed to generate fallback pointcloud") from exc
