"""
Video processing utilities with lazy OpenCV import and clearer error messages.

This module provides `VideoProcessor` which extracts frames from videos and
returns metadata. OpenCV (`cv2`) is imported lazily inside functions so merely
importing this module will not attempt to load heavy native extension modules.

If OpenCV fails to import (common on systems with incompatible NumPy/OpenCV
binary builds), the raised ImportError will include actionable guidance.
"""

from typing import Optional, Dict
import os


def _import_cv2():
    """
    Lazily import OpenCV and raise a helpful error if it fails.

    Returns:
        The imported cv2 module.

    Raises:
        ImportError: With guidance on how to resolve common installation problems.
    """
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - runtime import issues
        # Construct a helpful message. Keep it explicit and actionable.
        msg = (
            "Failed to import OpenCV (cv2). This is required for video processing.\n\n"
            "Common causes:\n"
            " - Incompatible binary wheels (e.g. OpenCV built against a different NumPy ABI).\n"
            " - Missing OpenCV installation in the active Python environment.\n\n"
            "Suggested fixes (choose the one appropriate for your environment):\n"
            " 1) Reinstall OpenCV and NumPy in the same environment:\n"
            "      pip install --upgrade --force-reinstall opencv-python numpy\n\n"
            " 2) If you are on a headless server (no GUI), use the headless package:\n"
            "      pip install --upgrade --force-reinstall opencv-python-headless numpy\n\n"
            " 3) If you use conda, prefer conda-forge builds which are ABI-consistent:\n"
            "      conda install -c conda-forge opencv numpy\n\n"
            " 4) If you recently upgraded NumPy to v2.x and a binary extension was compiled\n"
            "    against NumPy 1.x, consider downgrading to a NumPy 1.x series or rebuilding\n"
            "    the extension. Example (pip):\n"
            "      pip install 'numpy<2'\n\n"
            "Original import error: {err}\n"
        ).format(err=repr(exc))
        raise ImportError(msg) from exc


class VideoProcessor:
    """Handles video frame extraction with progress tracking."""

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        target_fps: int = 30,
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Extract frames from a video file at approximately `target_fps`.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save extracted frames.
            target_fps: Target frames per second to extract (integer > 0).
            progress_callback: Optional callable that accepts a single int argument
                               representing percentage progress (0-100).

        Returns:
            The number of frames saved.

        Raises:
            FileNotFoundError: If `video_path` does not exist.
            ValueError: If the video could not be opened or FPS could not be determined.
            ImportError: If OpenCV cannot be imported.
        """
        cv2 = _import_cv2()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Provide more context in the error; sometimes an inability to open may
            # mean a missing codec or corrupted file.
            raise ValueError(
                f"Could not open video file: {video_path}. "
                "This may indicate a corrupted file or missing codec support in your OpenCV build."
            )

        # Get properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if not video_fps or video_fps <= 0:
            cap.release()
            raise ValueError(
                f"Could not determine video FPS for file: {video_path}. "
                "Ensure the file is a valid video and your OpenCV build supports the file's codecs."
            )

        # Determine frame interval (sample frames to approximate target_fps)
        try:
            target_fps_i = max(1, int(target_fps))
        except Exception:
            target_fps_i = 30
        frame_interval = max(1, int(round(video_fps / float(target_fps_i))))

        frame_count = 0
        saved_count = 0
        last_progress = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(
                    output_dir, f"frame_{saved_count:06d}.jpg"
                )
                # Save with reasonable JPEG quality default; cv2 will choose encoding from extension
                success = cv2.imwrite(frame_filename, frame)
                if not success:
                    # If writing fails, record and continue; don't raise to allow partial output.
                    # Optionally the caller can inspect the output_dir to see what was written.
                    pass
                saved_count += 1

            frame_count += 1

            # Progress callback uses total_frames to estimate progress; if unknown, skip percent calc.
            if progress_callback and total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                if progress != last_progress:
                    try:
                        progress_callback(progress)
                    except Exception:
                        # avoid breaking processing due to callback errors
                        pass
                    last_progress = progress

        cap.release()

        # Finalize progress
        if progress_callback:
            try:
                progress_callback(100)
            except Exception:
                pass

        return saved_count

    def get_video_info(self, video_path: str) -> Dict[str, float]:
        """
        Return basic video metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with keys: 'fps', 'total_frames', 'width', 'height', 'duration'.

        Raises:
            ValueError: If the file cannot be opened or required properties are missing.
            ImportError: If OpenCV cannot be imported.
        """
        cv2 = _import_cv2()

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(
                f"Could not open video file to read metadata: {video_path}. "
                "Check the file and that your OpenCV has necessary codec support."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = None
        if fps and fps > 0:
            duration = total_frames / fps
        else:
            duration = 0.0

        cap.release()

        return {
            "fps": float(fps or 0.0),
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "duration": float(duration),
        }
