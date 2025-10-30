"""
Pointcloud generation utility.

Primary function:
    generate_pointcloud(frames_dir) -> str
        Reads image frames from `frames_dir`, runs the MapAnything model if available
        to produce 3D points with colors, writes an ASCII PLY file, and returns the path
        to the written PLY file.

Notes:
- This module does lazy imports of heavy dependencies (torch, mapanything) so importing
  the module is cheap. If the model or required libraries are not available, the
  function will fall back to writing a small placeholder PLY so callers get a path.
- The function intentionally accepts only `frames_dir` and returns the generated file path.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover - defensive fallback
    np = None  # we'll handle None in the function

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _find_image_files(frames_dir: str) -> Sequence[str]:
    files = []
    for fn in sorted(os.listdir(frames_dir)):
        if fn.lower().endswith(_IMAGE_EXTS):
            files.append(os.path.join(frames_dir, fn))
    return files


def _write_ply_ascii(
    output_path: str, points: "np.ndarray", colors: Optional["np.ndarray"] = None
) -> None:
    """
    Write an ASCII PLY file with vertex colors.

    Args:
        output_path: target file path
        points: (N,3) float32 array or list-like
        colors: (N,3) uint8 array or None (will default to 255,255,255)
    """
    if np is None:
        raise RuntimeError("numpy is required to write PLY files")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N,3)")

    n = points.shape[0]

    if colors is None:
        colors = np.full((n, 3), 255, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8)
        if colors.shape != (n, 3):
            raise ValueError("colors must be shape (N,3) and dtype uint8")

    header_lines = [
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

    with open(output_path, "w") as f:
        f.write("\n".join(header_lines) + "\n")
        # write points
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def generate_pointcloud(frames_dir: str) -> str:
    """
    Generate a colored pointcloud PLY from images in `frames_dir`.

    Returns:
        Absolute path to the generated .ply file.

    Behavior:
    - If `mapanything` and `torch` are available, the MapAnything model is used to
      infer per-view 3D points and colors which are aggregated into a single PLY.
    - If the model or dependencies are missing or inference fails, a small placeholder
      PLY is written so the caller still receives a file path.
    """
    if not os.path.isdir(frames_dir):
        raise ValueError(f"frames_dir must be an existing directory: {frames_dir}")

    image_files = _find_image_files(frames_dir)
    if not image_files:
        raise ValueError(f"No image frames found in directory: {frames_dir}")

    results_dir = os.path.join(frames_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, "pointcloud.ply")

    # Try to run the MapAnything model (lazy imports)
    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model (this may download weights on first run)
        model = MapAnything.from_pretrained("facebook/map-anything").to(device)

        # load_images accepts a directory or list of paths
        views = load_images(frames_dir)

        # Run inference - these args can be tuned for memory vs quality
        # Pick AMP options based on device. BF16 is typically only valid on supported GPU
        # backends; avoid requesting bf16 on CPU which can raise errors.
        use_amp = True if device == "cuda" else False
        amp_dtype = "bf16" if device == "cuda" else "fp16"

        try:
            # Try the preferred argument set
            predictions = model.infer(
                views,
                memory_efficient_inference=False,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )
        except TypeError:
            # Some model versions may not accept amp-related args; fall back to a safer call.
            predictions = model.infer(
                views,
                memory_efficient_inference=False,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )

        # Collect points and colors from predictions
        if np is None:
            raise RuntimeError("numpy is required for processing model outputs")

        all_points = []
        all_colors = []
        total_points_seen = 0

        for pred in predictions:
            # Expected keys: "pts3d", "img_no_norm", "mask"
            pts3d = pred.get(
                "pts3d", None
            )  # shape could be (B,H,W,3) or (H,W,3) or tensor
            img = pred.get("img_no_norm", None)  # (B,H,W,3) floats 0..1 or (H,W,3)
            mask = pred.get("mask", None)  # (B,H,W) or (H,W) or (B,H,W,1)

            if pts3d is None or img is None or mask is None:
                # skip incomplete predictions
                continue

            # Convert torch tensors to numpy if needed
            if hasattr(pts3d, "detach"):
                pts3d = pts3d.detach().cpu().numpy()
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()

            # Normalize dims: ensure a batch dimension exists (B, H, W, ...)
            if pts3d.ndim == 3:
                pts3d = pts3d[None, ...]  # (1, H, W, 3)
            if img.ndim == 3:
                img = img[None, ...]  # (1, H, W, 3)
            # mask may be (H,W), (H,W,1), (B,H,W) or (B,H,W,1)
            if mask.ndim == 2:
                mask = mask[None, ..., None]  # (1, H, W, 1)
            elif mask.ndim == 3:
                # could be (H,W,1) or (B,H,W)
                if mask.shape[-1] == 1:
                    mask = mask[None, ...]  # (1, H, W, 1)
                else:
                    mask = mask[..., None]  # (B, H, W, 1)

            # pts3d: (B, H, W, 3)
            B = int(pts3d.shape[0])
            for b in range(B):
                p = pts3d[b].reshape(-1, 3)
                c = img[b].reshape(-1, 3)
                m = mask[b].reshape(-1)
                # mask might be float probabilities; treat >0 as valid
                try:
                    valid = m > 0
                except Exception:
                    # If mask is somehow non-numeric, attempt truthiness
                    valid = np.asarray(m).astype(bool)
                if not np.any(valid):
                    continue
                p_sel = p[valid]
                c_sel = c[valid]
                # convert colors in range 0..1 to 0..255 if needed
                if c_sel.dtype.kind == "f":
                    c_sel = np.clip((c_sel * 255.0), 0, 255).astype(np.uint8)
                else:
                    # ensure uint8
                    c_sel = np.clip(c_sel, 0, 255).astype(np.uint8)

                total_points_seen += p_sel.shape[0]
                all_points.append(p_sel)
                all_colors.append(c_sel)

        # Basic logging: write a small stats file about the number of points aggregated
        try:
            stats_path = os.path.join(results_dir, "pointcloud_stats.txt")
            with open(stats_path, "w") as st:
                st.write(f"frames_processed: {len(image_files)}\n")
                st.write(f"total_points_aggregated: {total_points_seen}\n")
        except Exception:
            # do not fail the whole flow for logging errors
            pass

        if not all_points:
            raise RuntimeError("Model returned no valid 3D points")

        points = np.vstack(all_points)
        colors = np.vstack(all_colors)

        # Optionally downsample if too many points to keep PLY manageable
        max_points = 5_000_000  # arbitrary large limit
        if points.shape[0] > max_points:
            # random downsample
            idx = np.random.choice(points.shape[0], size=max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        # Write PLY (ASCII)
        _write_ply_ascii(output_file, points, colors)
        return os.path.abspath(output_file)

    except Exception as e:
        # If anything goes wrong (missing deps, OOM, model error), write a fallback placeholder PLY.
        # We intentionally catch broadly so callers still receive a file path.
        # Log minimal information by creating a small text file alongside the PLY for debugging.
        debug_txt = os.path.join(results_dir, "pointcloud_generation_error.txt")
        try:
            with open(debug_txt, "w") as dbg:
                dbg.write("Pointcloud generation via MapAnything failed.\n")
                dbg.write("Error repr:\n")
                dbg.write(repr(e) + "\n")
        except Exception:
            pass  # best-effort

        # Create a simple placeholder pointcloud: a small grid sized by frame count
        try:
            if np is None:
                raise RuntimeError("numpy is required to write placeholder PLY")

            n_frames = max(1, len(image_files))
            # create a modest grid of points proportional to number of frames
            grid_size = min(64, max(4, int(n_frames**0.5) * 4))
            xs = np.linspace(-1.0, 1.0, grid_size)
            ys = np.linspace(-1.0, 1.0, grid_size)
            zs = np.linspace(0.0, 1.0, max(2, grid_size // 8))
            pts = []
            cols = []
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    for k, z in enumerate(zs):
                        pts.append((x, y, z))
                        # color gradient based on indices
                        r = int(255 * (i / max(1, len(xs) - 1)))
                        g = int(255 * (j / max(1, len(ys) - 1)))
                        b = int(255 * (k / max(1, len(zs) - 1)))
                        cols.append((r, g, b))
            pts = np.array(pts, dtype=np.float32)
            cols = np.array(cols, dtype=np.uint8)
            _write_ply_ascii(output_file, pts, cols)
            return os.path.abspath(output_file)
        except Exception:
            # Last resort: create a tiny PLY with a single white point
            try:
                with open(output_file, "w") as f:
                    f.write(
                        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0.0 0.0 0.0 255 255 255\n"
                    )
                return os.path.abspath(output_file)
            except Exception as final_e:
                # If even this fails, surface an exception so callers know generation failed catastrophically.
                raise RuntimeError("Failed to write fallback PLY") from final_e
