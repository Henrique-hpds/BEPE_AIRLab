"""Stream the original Fast SAM demo views into one synchronized video."""

from pathlib import Path
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np


LIGHT_BLUE = (0.65, 0.74, 0.86)


def valid_mesh(person):
    vertices = person["pred_vertices"]
    return (vertices is not None and len(vertices) > 0
            and np.isfinite(vertices).all() and np.isfinite(person["pred_cam_t"]).all()
            and np.isfinite(float(person["focal_length"])) and float(person["focal_length"]) > 0)


def load_backend():
    """Reuse the original demo's MHR renderer and skeleton visualizer."""
    from mocap.utils.renderer import Renderer
    from sam_3d_body.metadata.mhr70 import pose_info
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

    skeleton = SkeletonVisualizer(line_width=2, radius=3)
    skeleton.set_pose_meta(pose_info)
    return SimpleNamespace(Renderer=Renderer, skeleton=skeleton)


def label(image, text, *, bottom=False):
    image = image.copy()
    height, width = image.shape[:2]
    y = height - 8 if bottom else 22
    cv2.rectangle(image, (0, y - 19), (width, y + 5), (245, 245, 245), -1)
    cv2.putText(image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    return image


class VideoVisualizations:
    """Write original + four demo views side by side, without reading an NPZ."""

    def __init__(self, output_dir, faces, *, fps, height=480, codec="mp4v"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = load_backend()
        self.mhr_faces = np.asarray(faces, dtype=np.int32)
        self.mhr_renderer = self.backend.Renderer(focal_length=1.0, faces=self.mhr_faces)
        self.fps = fps
        self.height = height
        self.codec = codec
        self.writers = {}
        self.closed = False

    def _resize(self, image):
        height, width = image.shape[:2]
        target_width = max(2, round(width * self.height / height) // 2 * 2)
        return cv2.resize(image, (target_width, self.height), interpolation=cv2.INTER_AREA)

    def _render(self, renderer, vertices, translation, background, *, side=False):
        image = renderer(
            vertices, translation, background.copy(), side_view=side,
            mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1),
        )
        return np.clip(image * 255, 0, 255).astype(np.uint8)

    def _mhr_images(self, frame, outputs):
        skeleton, hands = frame.copy(), frame.copy()
        vertices_list, faces_list, valid = [], [], []
        vertex_offset = 0
        for person in sorted(outputs, key=lambda item: float(item["pred_cam_t"][2]), reverse=True):
            points = np.asarray(person["pred_keypoints_2d"], dtype=np.float64)
            points = np.where(np.isfinite(points), points, np.nan)
            points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            skeleton = self.backend.skeleton.draw_skeleton(skeleton, points)
            for key, target, color in (
                ("bbox", hands, (0, 255, 0)),
                ("lhand_bbox", hands, (255, 0, 0)),
                ("rhand_bbox", hands, (0, 0, 255)),
            ):
                if key in person:
                    box = np.asarray(person[key])
                    if np.isfinite(box).all():
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(target, (x1, y1), (x2, y2), color, 2)

            if not valid_mesh(person):
                continue
            vertices = person["pred_vertices"]
            translation = person["pred_cam_t"]
            vertices_list.append(np.asarray(vertices) + translation)
            faces_list.append(self.mhr_faces + vertex_offset)
            vertex_offset += len(vertices)
            valid.append(person)

        white = np.full_like(frame, 255)
        overlay, side = frame.copy(), white.copy()
        if valid:
            closest = np.concatenate(vertices_list[-2:], axis=0)
            translation = (closest.max(axis=0) + closest.min(axis=0)) / 2
            vertices = np.concatenate(vertices_list, axis=0) - translation
            self.mhr_renderer.faces = np.concatenate(faces_list, axis=0)
            self.mhr_renderer.focal_length = float(valid[-1]["focal_length"])
            overlay = self._render(self.mhr_renderer, vertices, translation, frame)
            side = self._render(self.mhr_renderer, vertices, translation, white, side=True)
        return {
            "original": frame, "hand_boxes": hands,
            "keypoints": skeleton, "mhr_overlay": overlay,
            "mhr_side": side,
        }

    def _video_frame(self, filename, image):
        height, width = image.shape[:2]
        if filename not in self.writers:
            with tempfile.NamedTemporaryFile(
                dir=self.output_dir, prefix=".render-", suffix=".mp4", delete=False,
            ) as file:
                temporary_path = Path(file.name)
            writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*self.codec),
                                     self.fps, (width, height))
            if not writer.isOpened():
                writer.release()
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(f"Could not open visualization video writer ({self.codec}): {filename}")
            self.writers[filename] = (writer, temporary_path, (width, height))
        writer, _path, size = self.writers[filename]
        if size != (width, height):
            raise ValueError("Visualization frame dimensions changed within the video")
        writer.write(image)

    def write(self, frame_index, frame, outputs):
        """Consume this frame's inference results once; keep skipped frames in videos."""
        if self.closed:
            raise RuntimeError("Visualization writer is closed")
        images = self._mhr_images(frame, outputs)
        panels = [
            label(self._resize(images[key]), title) for key, title in (
                ("original", "Original"), ("hand_boxes", "Corpo / maos"),
                ("keypoints", "Esqueleto 2D"), ("mhr_overlay", "Malha MHR"),
                ("mhr_side", "MHR lateral"),
            )
        ]
        status = f"frame={frame_index} t={frame_index / self.fps:.3f}s"
        if len(outputs) != 1:
            status += f" | Ignorado pelo SMPL: {len(outputs)} pessoas"
        self._video_frame("visualization.mp4", label(np.concatenate(panels, axis=1), status, bottom=True))

    def close(self, *, commit=True):
        if self.closed:
            return
        self.closed = True
        cleanup_error = None
        try:
            for writer, _temporary_path, _size in self.writers.values():
                try:
                    writer.release()
                except Exception as error:
                    cleanup_error = cleanup_error or error
            try:
                self.mhr_renderer.delete()
            except Exception as error:
                cleanup_error = cleanup_error or error
            if commit and cleanup_error is None:
                for filename, (_writer, temporary_path, _size) in self.writers.items():
                    temporary_path.replace(self.output_dir / filename)
        finally:
            for _writer, temporary_path, _size in self.writers.values():
                temporary_path.unlink(missing_ok=True)
        if cleanup_error is not None:
            raise cleanup_error
