import cv2
import os


class VideoProcessor:
    """Handles video frame extraction with progress tracking"""

    def extract_frames(self, video_path, output_dir, target_fps=30, progress_callback=None):
        """
        Extract frames from video at specified FPS

        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames
            target_fps: Target frames per second to extract
            progress_callback: Optional callback function(progress_percentage)

        Returns:
            Number of frames extracted
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps == 0:
            raise ValueError("Could not determine video FPS")

        # Calculate frame interval to achieve target FPS
        frame_interval = max(1, int(video_fps / target_fps))

        frame_count = 0
        saved_count = 0
        last_progress = -1

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save frame if it matches our interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f'frame_{saved_count:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

            # Report progress
            if progress_callback and total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                if progress != last_progress:
                    progress_callback(progress)
                    last_progress = progress

        cap.release()

        # Final progress callback
        if progress_callback:
            progress_callback(100)

        return saved_count

    def get_video_info(self, video_path):
        """
        Get video metadata

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }

        cap.release()
        return info
