"""
Video segmentation using CLIP-guided SAM2.

Workflow:
1. Extract first frame
2. Run CLIP dense prediction to find objects
3. Extract prompt points for each class
4. Use SAM2 video predictor to track across all frames
5. Generate segmented video output
"""

import numpy as np
import torch
import cv2
import subprocess
import tempfile
import shutil
import os
import colorsys
from typing import List, Dict
from sam2.build_sam import build_sam2_video_predictor


def generate_distinct_colors(n):
    """
    Generate visually distinct colors using a curated palette.

    Args:
        n: Number of colors needed

    Returns:
        List of RGB tuples in range [0, 255] (for OpenCV)
    """
    # Curated palette of visually distinct colors (RGB in 0-255 range for OpenCV)
    base_palette = [
        (31, 119, 180),   # Blue
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),   # Yellow
        (23, 190, 207),   # Cyan
        (255, 166, 0),    # Gold
        (51, 204, 51),    # Lime
        (204, 0, 204),    # Magenta
        (0, 128, 128),    # Teal
        (128, 0, 0),      # Maroon
        (0, 0, 128),      # Navy
    ]

    if n <= len(base_palette):
        return [list(c) for c in base_palette[:n]]

    # If we need more colors, generate additional ones using HSV
    colors = [list(c) for c in base_palette]
    for i in range(n - len(base_palette)):
        hue = (i * 0.618033988749895) % 1.0  # Golden ratio for spacing
        saturation = 0.6 + (i % 3) * 0.15
        value = 0.7 + (i % 2) * 0.2

        # Convert HSV to RGB (0-255)
        rgb_01 = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_255 = [int(c * 255) for c in rgb_01]
        colors.append(rgb_255)

    return colors


class CLIPGuidedVideoSegmentor:
    """
    Video segmentation using CLIP guidance + SAM2 tracking.

    Strategy:
    - Frame 0: CLIP finds what objects are present and where
    - SAM2: Tracks those objects across all frames
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/sam2_hiera_large.pt",
        model_cfg: str = "sam2_hiera_l.yaml",
        device: str = None
    ):
        """
        Initialize video segmentor.

        Args:
            checkpoint_path: Path to SAM2 checkpoint
            model_cfg: SAM2 model configuration
            device: Computation device
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg

        print(f"[Video Segmentor] Loading SAM2 video predictor (with CPU offloading to save GPU memory)...")

        # Use smaller model to fit in limited GPU memory
        # and enable CPU offloading for video frames
        self.predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint_path,
            device=self.device
        )
        print(f"[Video Segmentor] Ready!")

    def segment_video(
        self,
        video_path: str,
        prompts: List[Dict],
        output_path: str = None,
        visualize: bool = True,
        fps: int = None
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Segment video using CLIP-extracted prompts.

        Args:
            video_path: Path to input video
            prompts: List of prompt dictionaries from CLIP analysis
                     Each should have: 'point' (x,y), 'class_idx', 'class_name'
            output_path: Path to save output video (optional)
            visualize: Whether to create visualization overlay
            fps: Output FPS (uses input FPS if None)

        Returns:
            video_segments: Dict[frame_idx -> Dict[obj_id -> mask]]
        """
        print(f"\n[Video Segmentation] Processing {video_path}")
        print(f"  Prompts: {len(prompts)} objects to track")

        # Initialize inference state with CPU offloading to save GPU memory
        print(f"[Video Segmentation] Loading video...")
        inference_state = self.predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True,  # Keep video frames on CPU to save GPU memory
            offload_state_to_cpu=True   # Keep intermediate states on CPU
        )
        num_frames = inference_state["num_frames"]
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]

        print(f"  Video: {num_frames} frames, {video_width}x{video_height}")

        # Add prompts for each object on frame 0
        print(f"[Video Segmentation] Adding prompts on frame 0...")
        frame_idx = 0

        for prompt_info in prompts:
            obj_id = prompt_info['class_idx']  # Use class_idx as object ID
            point = prompt_info['point']
            class_name = prompt_info['class_name']

            # Convert point to numpy array
            points = np.array([[point[0], point[1]]], dtype=np.float32)
            labels = np.array([1], np.int32)  # 1 = foreground

            # Add prompt to SAM2
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

            print(f"  Added object {obj_id} ({class_name}) at {point}")

        # Propagate across video
        print(f"[Video Segmentation] Propagating masks across {num_frames} frames...")
        video_segments = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            if (out_frame_idx + 1) % 30 == 0:
                print(f"  Processed {out_frame_idx + 1}/{num_frames} frames...")

        print(f"[Video Segmentation] Segmentation complete!")

        # Generate output video if requested
        if output_path and visualize:
            self._save_video(
                video_path=video_path,
                video_segments=video_segments,
                prompts=prompts,
                output_path=output_path,
                fps=fps
            )

        return video_segments

    def _save_video(
        self,
        video_path: str,
        video_segments: Dict,
        prompts: List[Dict],
        output_path: str,
        fps: int = None
    ):
        """
        Save segmented video with visualization overlay using ffmpeg.

        Uses H.264 codec with faststart for universal compatibility.

        Args:
            video_path: Original video path
            video_segments: Segmentation results
            prompts: Prompt information for class labels
            output_path: Output video path
            fps: Output FPS
        """
        print(f"[Video Output] Generating visualization...")

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = fps or input_fps

        # Create class_idx -> class_name mapping
        class_names = {p['class_idx']: p['class_name'] for p in prompts}

        # Create colors for each class using distinct colors
        distinct_colors = generate_distinct_colors(len(class_names))
        # Map class indices to colors
        class_idx_list = sorted(class_names.keys())
        class_colors = {
            class_idx: distinct_colors[i]
            for i, class_idx in enumerate(class_idx_list)
        }

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp(prefix='video_seg_')
        try:
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Create overlay
                overlay = frame.copy()

                # Draw masks for this frame
                if frame_idx in video_segments:
                    for obj_id, mask in video_segments[frame_idx].items():
                        if obj_id in class_colors:
                            color = class_colors[obj_id]
                            class_name = class_names.get(obj_id, f"Object {obj_id}")

                            # Apply colored mask
                            mask_bool = mask[0] > 0  # (H, W)
                            overlay[mask_bool] = (
                                overlay[mask_bool] * 0.5 +
                                np.array(color) * 0.5
                            ).astype(np.uint8)

                            # Draw label
                            coords = np.argwhere(mask_bool)
                            if len(coords) > 0:
                                centroid_y, centroid_x = coords.mean(axis=0).astype(int)
                                cv2.putText(
                                    overlay,
                                    class_name,
                                    (centroid_x, centroid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 255),
                                    2
                                )

                # Save frame as PNG
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_path, overlay)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    print(f"  Rendered {frame_idx} frames...")

            cap.release()
            total_frames = frame_idx

            # Use ffmpeg to encode video with H.264 and faststart
            print(f"[Video Output] Encoding video with ffmpeg (H.264 + faststart)...")

            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',  # H.264 codec (universal compatibility)
                '-pix_fmt', 'yuv420p',  # Standard pixel format
                '-preset', 'medium',  # Encoding speed/quality tradeoff
                '-crf', '23',  # Quality (18-28 range, 23 is default)
                '-movflags', '+faststart',  # Move moov atom to start for streaming
                output_path
            ]

            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {result.stderr}")

            print(f"[Video Output] Saved to {output_path}")
            print(f"  Frames: {total_frames}, FPS: {fps}")

            # Verify output is playable
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_cap.release()
                print(f"[Video Output] ✓ Video verified playable")
            else:
                print(f"[Video Output] ⚠ Warning: Could not verify output video")

        finally:
            # Clean up temporary frames
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"[Video Output] Cleaned up temporary frames")
