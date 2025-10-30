import os
import time

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)


def generate_pointcloud(frames_dir, progress_callback=None):
    # Load and preprocess images from a folder or list of paths
    images = frames_dir
    views = load_images(images)
    predictions = model.infer(
        views,  # Input views
        memory_efficient_inference=False,  # Trades off speed for more views (up to 2000 views on 140 GB)
        use_amp=True,  # Use mixed precision inference (recommended)
        amp_dtype="bf16",  # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
        apply_mask=True,  # Apply masking to dense geometry outputs
        mask_edges=True,  # Remove edge artifacts by using normals and depth
        apply_confidence_mask=False,  # Filter low-confidence regions
        confidence_percentile=10,
    )

    # Access results for each view - Complete list of metric outputs
    for i, pred in enumerate(predictions):
        # Geometry outputs
        pts3d = pred["pts3d"]  # 3D points in world coordinates (B, H, W, 3)
        pts3d_cam = pred["pts3d_cam"]  # 3D points in camera coordinates (B, H, W, 3)
        depth_z = pred["depth_z"]  # Z-depth in camera frame (B, H, W, 1)
        depth_along_ray = pred[
            "depth_along_ray"
        ]  # Depth along ray in camera frame (B, H, W, 1)

        # Camera outputs
        ray_directions = pred[
            "ray_directions"
        ]  # Ray directions in camera frame (B, H, W, 3)
        intrinsics = pred["intrinsics"]  # Recovered pinhole camera intrinsics (B, 3, 3)
        camera_poses = pred[
            "camera_poses"
        ]  # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
        cam_trans = pred[
            "cam_trans"
        ]  # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
        cam_quats = pred[
            "cam_quats"
        ]  # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

        # Quality and masking
        confidence = pred["conf"]  # Per-pixel confidence scores (B, H, W)
        mask = pred["mask"]  # Combined validity mask (B, H, W, 1)
        non_ambiguous_mask = pred[
            "non_ambiguous_mask"
        ]  # Non-ambiguous regions (B, H, W)
        non_ambiguous_mask_logits = pred[
            "non_ambiguous_mask_logits"
        ]  # Mask logits (B, H, W)

        # Scaling
        metric_scaling_factor = pred[
            "metric_scaling_factor"
        ]  # Applied metric scaling (B,)

        # Original input
        img_no_norm = pred[
            "img_no_norm"
        ]  # Denormalized input images for visualization (B, H, W, 3)


#     # Count frames
#     frame_files = [f for f in os.listdir(frames_dir) if f.endswith((".jpg", ".png"))]
#     num_frames = len(frame_files)

#     if num_frames == 0:
#         raise ValueError("No frames found in directory")

#     # Simulate processing with progress updates
#     # In real implementation, this would be your actual pointcloud generation logic
#     output_file = os.path.join(frames_dir, "pointcloud.poi")

#     # Simulate processing steps
#     steps = 10
#     for i in range(steps):
#         time.sleep(0.5)  # Simulate processing time

#         # Report progress
#         if progress_callback:
#             progress = int(((i + 1) / steps) * 100)
#             progress_callback(progress)

#     # Create a placeholder POI file
#     # In real implementation, this would be your actual pointcloud data
#     with open(output_file, "w") as f:
#         f.write(f"# Placeholder Pointcloud File\n")
#         f.write(f"# Generated from {num_frames} frames\n")
#         f.write(f"# Replace this with actual pointcloud data\n")
#         f.write(f"# Format: X Y Z R G B\n")
#         f.write(f"\n")
#         f.write(f"# Example point data:\n")
#         for i in range(100):
#             f.write(f"{i * 0.1} {i * 0.2} {i * 0.3} 255 128 64\n")

#     if progress_callback:
#         progress_callback(100)

#     return output_file


# # Alternative implementation example if you're using a different pointcloud library:
# """
# def generate_pointcloud_with_library(frames_dir, progress_callback=None):
#     '''
#     Example integration with a hypothetical pointcloud library
#     '''
#     import your_pointcloud_library as pcl

#     frame_files = sorted([
#         os.path.join(frames_dir, f)
#         for f in os.listdir(frames_dir)
#         if f.endswith(('.jpg', '.png'))
#     ])

#     # Initialize your pointcloud generator
#     generator = pcl.PointcloudGenerator()

#     # Process frames
#     for idx, frame_path in enumerate(frame_files):
#         generator.add_frame(frame_path)

#         if progress_callback:
#             progress = int(((idx + 1) / len(frame_files)) * 100)
#             progress_callback(progress)

#     # Generate final pointcloud
#     output_file = os.path.join(frames_dir, 'pointcloud.ply')
#     generator.export(output_file)

#     return output_file
# """
