"""Streaming visualization of live Fast SAM 3D Body and SMPL frame results."""

import json
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


def load_backend(smpl_model_path):
    """Reuse the repository's skeleton, mesh and SMPL debug renderers."""
    from debug_smpl_stream import prepare_render_records
    from mocap.utils.renderer import Renderer
    from mocap.utils.smpl_render_utils import (
        _project_joints, draw_skeleton, load_smpl_model, smpl_vertices_joints_from_pose,
    )
    from sam_3d_body.metadata.mhr70 import pose_info
    from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

    skeleton = SkeletonVisualizer(line_width=2, radius=3)
    skeleton.set_pose_meta(pose_info)
    smpl_model, smpl_faces, device, num_betas = load_smpl_model(smpl_model_path)

    def smpl_geometry(record):
        render_record = prepare_render_records([record])[0]
        return smpl_vertices_joints_from_pose(
            render_record["smpl_pose"], smpl_model, device, num_betas,
            body_quat=render_record["body_quat"],
        )

    return SimpleNamespace(
        Renderer=Renderer, skeleton=skeleton, smpl_faces=smpl_faces,
        smpl_geometry=smpl_geometry, project_joints=_project_joints,
        draw_skeleton=draw_skeleton,
    )


def label(image, text, *, bottom=False):
    image = image.copy()
    height, width = image.shape[:2]
    y = height - 8 if bottom else 22
    cv2.rectangle(image, (0, y - 19), (width, y + 5), (245, 245, 245), -1)
    cv2.putText(image, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    return image


class VideoVisualizations:
    """Write synchronized videos, frame images and PLYs without reading an NPZ."""

    def __init__(self, output_dir, faces, *, smpl_model_path, fps, save_videos=False,
                 save_images=False, save_meshes=False, export_every=1,
                 height=480, codec="mp4v"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = load_backend(smpl_model_path)
        self.mhr_faces = np.asarray(faces, dtype=np.int32)
        self.mhr_renderer = self.backend.Renderer(focal_length=1.0, faces=self.mhr_faces)
        self.smpl_width = max(2, int(height * 2 / 3) // 2 * 2)
        self.smpl_height = height
        self.smpl_focal = float(height)
        self.smpl_translation = np.array([0.0, 0.0, 2.5], dtype=np.float32)
        self.smpl_renderer = self.backend.Renderer(
            focal_length=self.smpl_focal, faces=self.backend.smpl_faces,
        )
        self.fps = fps
        self.save_videos = save_videos
        self.save_images = save_images
        self.save_meshes = save_meshes
        self.export_every = export_every
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

    def _mhr_images(self, frame, outputs, include_front):
        skeleton, boxes, hands = frame.copy(), frame.copy(), frame.copy()
        vertices_list, faces_list, valid = [], [], []
        vertex_offset = 0
        for person in sorted(outputs, key=lambda item: float(item["pred_cam_t"][2]), reverse=True):
            points = np.asarray(person["pred_keypoints_2d"], dtype=np.float64)
            points = np.where(np.isfinite(points), points, np.nan)
            points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            skeleton = self.backend.skeleton.draw_skeleton(skeleton, points)
            for key, target, color in (
                ("bbox", boxes, (0, 255, 0)),
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
        overlay, front, side = frame.copy(), white.copy(), white.copy()
        if valid:
            closest = np.concatenate(vertices_list[-2:], axis=0)
            translation = (closest.max(axis=0) + closest.min(axis=0)) / 2
            vertices = np.concatenate(vertices_list, axis=0) - translation
            self.mhr_renderer.faces = np.concatenate(faces_list, axis=0)
            self.mhr_renderer.focal_length = float(valid[-1]["focal_length"])
            overlay = self._render(self.mhr_renderer, vertices, translation, frame)
            side = self._render(self.mhr_renderer, vertices, translation, white, side=True)
            if include_front:
                front = self._render(self.mhr_renderer, vertices, translation, white)
        images = {
            "original": frame, "body_boxes": boxes, "hand_boxes": hands,
            "keypoints": skeleton, "mhr_overlay": overlay,
            "mhr_front": front, "mhr_side": side,
        }
        return images, valid

    def _smpl_images(self, record, detection_count):
        background = np.full((self.smpl_height, self.smpl_width, 3), 255, dtype=np.uint8)
        if record is None:
            text = "Sem pessoa" if detection_count == 0 else "Multiplas pessoas"
            blank = label(background, text, bottom=True)
            return blank, blank.copy(), None

        vertices, joints = self.backend.smpl_geometry(record)
        front = self._render(self.smpl_renderer, vertices, self.smpl_translation, background)
        side = self._render(self.smpl_renderer, vertices, self.smpl_translation, background, side=True)
        for image, is_side in ((front, False), (side, True)):
            projected, mask = self.backend.project_joints(
                joints, self.smpl_translation, self.smpl_focal,
                self.smpl_width, self.smpl_height, side_view=is_side,
            )
            self.backend.draw_skeleton(image, projected, mask)
        return front, side, vertices

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

    def write(self, frame_index, frame, outputs, smpl_record):
        """Consume this frame's inference results once; keep skipped frames in videos."""
        if self.closed:
            raise RuntimeError("Visualization writer is closed")
        export_frame = frame_index % self.export_every == 0
        if not self.save_videos and not export_frame:
            return

        valid_people = [person for person in outputs if valid_mesh(person)]
        smpl_vertices = None
        if self.save_videos or self.save_images:
            images, valid_people = self._mhr_images(
                frame, outputs, include_front=self.save_images and export_frame,
            )
            smpl_front, smpl_side, smpl_vertices = self._smpl_images(smpl_record, len(outputs))
            images["smpl_front"] = smpl_front
            images["smpl_side"] = smpl_side
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
            images["concat_all"] = label(np.concatenate(panels, axis=1), status, bottom=True)
            images["smpl_all"] = label(np.concatenate([
                label(smpl_front, "SMPL frontal"), label(smpl_side, "SMPL lateral"),
            ], axis=1), status, bottom=True)
        elif smpl_record is not None:
            smpl_vertices, _joints = self.backend.smpl_geometry(smpl_record)

        if self.save_videos:
            self._video_frame("visualization.mp4", images["concat_all"])
            self._video_frame("smpl_visualization.mp4", images["smpl_all"])
        if self.save_images and export_frame:
            directory = self.output_dir / "frames" / f"frame_{frame_index:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            for name, image in images.items():
                if not cv2.imwrite(str(directory / f"{name}.jpg"), image):
                    raise RuntimeError(f"Could not save visualization image: {directory / name}")
        if self.save_meshes and export_frame:
            directory = self.output_dir / "meshes"
            directory.mkdir(parents=True, exist_ok=True)
            for person_index, person in enumerate(valid_people):
                exporter = self.backend.Renderer(float(person["focal_length"]), self.mhr_faces)
                mesh = exporter.vertices_to_trimesh(
                    person["pred_vertices"], person["pred_cam_t"], LIGHT_BLUE,
                )
                mesh.export(str(directory / f"frame_{frame_index:06d}_mhr_{person_index:03d}.ply"))
            if smpl_vertices is not None:
                mesh = self.smpl_renderer.vertices_to_trimesh(
                    smpl_vertices, np.zeros(3, dtype=np.float32), LIGHT_BLUE,
                )
                mesh.export(str(directory / f"frame_{frame_index:06d}_smpl.ply"))
        if export_frame and (self.save_images or self.save_meshes):
            directory = (self.output_dir / "frames" / f"frame_{frame_index:06d}"
                         if self.save_images else self.output_dir / "meshes")
            metadata = {
                "frame_index": frame_index, "time_seconds": frame_index / self.fps,
                "smpl_valid": smpl_record is not None, "smpl_shape": "neutral (betas=0)",
                "people": [{
                    "focal_length": float(person["focal_length"]),
                    "bbox": np.asarray(person["bbox"]).tolist(),
                } for person in valid_people],
            }
            path = directory / f"frame_{frame_index:06d}_camera.json"
            path.write_text(json.dumps(metadata, indent=2) + "\n")

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
            for renderer in (self.mhr_renderer, self.smpl_renderer):
                try:
                    renderer.delete()
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
