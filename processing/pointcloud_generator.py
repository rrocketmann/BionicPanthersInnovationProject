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


def generate_pointcloud(
    frames_dir, progress_callback=None
):  # Load and preprocess images from a folder or list of paths
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

    output_file = os.path.join(frames_dir, "pointcloud.ply")  # Define output file path

    # Process predictions to generate pointcloud data
    with open(output_file, "w") as f:
        f.write(f"# Pointcloud File\n")
        f.write(f"# Generated from provided frames\n")
        f.write(f"# Replace this with actual pointcloud data\n")
        f.write(f"# Format: X Y Z R G B\n")
        for i, pred in enumerate(predictions):
            # Geometry outputs
            pts3d = pred["pts3d"]  # 3D points in world coordinates (B, H, W, 3)
            # Here you would transform your pts3d and other data into the desired pointcloud format
            # Example of iterating if pts3d were usable.
            for point in pts3d.reshape(-1, 3):  # Flatten and iterate over 3D points
                f.write(
                    f"{point[0]} {point[1]} {point[2]} 255 255 255\n"
                )  # Placeholder color white

    return output_file  # Return the path to the generated .ply file


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
