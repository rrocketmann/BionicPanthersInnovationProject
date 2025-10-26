from doctest import OutputChecker
import os
import time
import numpy as np
import torch
from mapanything import MapAnything
from typing import List, Optional


def generate_pointcloud(frames_dir, progress_callback=None):
    def save_pointcloud_to_ply(
        output_path: str, points: np.ndarray, colors: np.ndarray
    ) -> None:
        """
        Save merged points & colors to a PLY file (binary little-endian).
        """
        if points.shape[0] == 0:
            raise ValueError("No points to save to PLY.")

        vertices = np.zeros(
            len(points),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )
        vertices["x"] = points[:, 0]
        vertices["y"] = points[:, 1]
        vertices["z"] = points[:, 2]
        vertices["red"] = colors[:, 0]
        vertices["green"] = colors[:, 1]
        vertices["blue"] = colors[:, 2]

        ply_element = PlyElement.describe(vertices, "vertex")
        PlyData([ply_element]).write(output_path)

    def main(argv: Optional[List[str]] = None) -> int:
        parser = argparse.ArgumentParser(description="MapAnything 3D Reconstruction")

        parser.add_argument(
            "--output", type=str, default="output.ply", help="Output PLY file path"
        )
        parser.add_argument(
            "--images-dir",
            type=str,
            default="images",
            help="Directory containing images (default: images)",
        )
        parser.add_argument(
            "--video",
            type=str,
            default=None,
            help="Path to an input video. If provided, frames will be extracted and used instead of --images-dir.",
        )
        parser.add_argument(
            "--max-images",
            type=int,
            default=5,
            help="Maximum number of images/frames to use for reconstruction when extracting from a video (default: 5).",
        )
        parser.add_argument(
            "--keep-frames",
            action="store_true",
            help="If set while using --video, keep the extracted frames (do not delete the temporary directory).",
        )

        args = parser.parse_args(argv)
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        if device == "cpu":
            print("Warning: CPU inference is slow. GPU recommended for production use.")

            # Load model
            print("\nLoading model...")
            model = MapAnything.from_pretrained("facebook/map-anything").to(device)
            print("Model loaded successfully")

            # Load and preprocess images
            print("\nLoading and preprocessing images...")
            # `load_images` accepts either a directory or list of image paths; pass list for clarity.
            views = load_images(frames_dir)
            print(f"Images loaded: {len(views)} views")

            # Run inference
            print("\nRunning inference (this may take several minutes)...")
            predictions = model.infer(
                views,
                memory_efficient_inference=False,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
                confidence_percentile=10,
            )
            print("Inference completed")

            # Process and merge point clouds
            print("\nMerging point clouds...")
            all_points = []
            all_colors = []

            for i, pred in enumerate(predictions):
                # Extract 3D points in world coordinates
                pts3d = pred["pts3d"][0].cpu().numpy()
                points_flat = pts3d.reshape(-1, 3)

                # Get image colors
                img = pred["img_no_norm"][0].cpu().numpy()
                colors = (img.reshape(-1, 3) * 255).astype(np.uint8)

                # Apply mask
                mask = pred["mask"][0].cpu().numpy().reshape(-1)
                valid_points = points_flat[mask > 0.5]
                valid_colors = colors[mask > 0.5]

                all_points.append(valid_points)
                all_colors.append(valid_colors)
                print(
                    f"  View {i + 1}/{len(predictions)}: {len(valid_points):,} points"
                )

            # Merge all views
            if len(all_points) == 0:
                print("No valid points recovered from predictions.")
                return 5

            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)
            print(f"\nTotal merged points: {len(merged_points):,}")

            # Save to PLY file
            save_pointcloud_to_ply(args.output, merged_points, merged_colors)
            print(f"\nOutput saved: {args.output}")
            print("Reconstruction complete")

            # If we used a temporary dir and user didn't ask to keep frames, remove it
            # if temp_dir and not args.keep_frames:
            #     try:
            #         shutil.rmtree(temp_dir)
            #         if temp_dir:
            #             print(f"Cleaned up temporary frames dir: {temp_dir}")
            #     except Exception:
            #         # Non-fatal; warn but continue
            #         print(f"Warning: failed to remove temporary frames dir: {temp_dir}")

            return 0


#     """
#     PLACEHOLDER FUNCTION: Generate pointcloud from frames

#     This is a placeholder that simulates the pointcloud generation process.
#     Replace this with your actual pointcloud generation implementation.

#     Args:
#         frames_dir: Directory containing extracted frames
#         progress_callback: Optional callback function(progress_percentage)

#     Returns:
#         Path to generated pointcloud file (.poi or .ply)
#     """
#     # Count frames
#     frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
#     num_frames = len(frame_files)

#     if num_frames == 0:
#         raise ValueError("No frames found in directory")

#     # Simulate processing with progress updates
#     # In real implementation, this would be your actual pointcloud generation logic
#     output_file = os.path.join(frames_dir, 'pointcloud.poi')

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
#     with open(output_file, 'w') as f:
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
