#!/usr/bin/env python3
"""Extract Fast SAM 3D Body SMPL arrays and optional live visualizations from videos.

Run with --help for usage. Model dependencies are imported only for extraction.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
FASTSAM_DIR = ROOT / "Fast-SAM-3D-Body"
VIDEO_EXTENSIONS = {
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
    ".mts", ".m2ts", ".webm", ".wmv",
}
LOGGER = logging.getLogger("extract_poses")


@contextmanager
def model_output(verbose=False):
    """Silence upstream print diagnostics while leaving stderr and exceptions visible."""
    if verbose:
        yield
    else:
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            try:
                yield
            finally:
                # Libraries imported while stdout is redirected can retain the
                # sink in their handlers. Rebind those handlers before closing it.
                loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
                for logger in loggers:
                    if isinstance(logger, logging.Logger):
                        for handler in logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and handler.stream is sink:
                                handler.setStream(sys.stderr)


class UniqueWarnings(logging.Filter):
    """Show each model warning once, without suppressing repeated errors."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        if record.levelno != logging.WARNING:
            return True
        message = record.getMessage()
        if message in self.seen:
            return False
        self.seen.add(message)
        return True


@contextmanager
def model_logging(verbose=False):
    """Route YOLO logs through tqdm and deduplicate warnings in quiet mode."""
    from tqdm.contrib.logging import logging_redirect_tqdm

    logger = logging.getLogger("ultralytics")
    previous_level, previous_propagate = logger.level, logger.propagate
    warning_filter = UniqueWarnings()
    if not verbose:
        logger.setLevel(logging.WARNING)
        logger.addFilter(warning_filter)
    logger.propagate = False
    try:
        with logging_redirect_tqdm(loggers=[logger]):
            yield
    finally:
        logger.removeFilter(warning_filter)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def discover_videos(input_path):
    """Return videos and their base directory, preserving folder structure."""
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return [input_path], input_path.parent
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    videos = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise ValueError(f"No supported videos found in: {input_path}")
    return videos, input_path


def load_estimator(args):
    """Use the same estimator setup as run_publisher.py, with unknown intrinsics."""
    required = [
        FASTSAM_DIR / "checkpoints/sam-3d-body-dinov3/model.ckpt",
        FASTSAM_DIR / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
        args.yolo_model,
        args.smpl_model_path,
        args.nn_model_dir / "best_model.pth",
        args.nn_model_dir / "sample_idx.npy",
        args.mhr2smpl_mapping_path,
    ]
    if args.mhr_mesh_path is not None:
        required.append(args.mhr_mesh_path)
    if args.smoother_dir is not None:
        required.extend([
            args.smoother_dir / "smoother_best.pth",
            args.smoother_dir / "smoother_config.json",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model assets:\n  " + "\n  ".join(missing))

    # build_fov_estimator reads these settings at import time. Match the publisher.
    os.environ.update({
        "FOV_MODEL": "s", "FOV_LEVEL": "0", "FOV_SIZE": "512", "FOV_FAST": "1",
    })
    if str(FASTSAM_DIR) not in sys.path:
        sys.path.insert(0, str(FASTSAM_DIR))

    import torch
    from mocap.core.setup_estimator import build_default_estimator

    if not torch.cuda.is_available():
        raise RuntimeError("Fast SAM 3D Body's inference pipeline requires CUDA.")

    # The original setup uses relative SAM/MHR checkpoint paths.
    previous_directory = Path.cwd()
    try:
        os.chdir(FASTSAM_DIR)
        return build_default_estimator(
            image_size=args.image_size,
            yolo_model_path=str(args.yolo_model),
        )
    finally:
        os.chdir(previous_directory)


def load_converter(args):
    from mocap.core.multiview_mhr2smpl import MultiViewFusionRunner

    # A fresh converter prevents temporal smoothing from crossing video boundaries.
    return MultiViewFusionRunner(
        smpl_model_path=str(args.smpl_model_path),
        model_dir=str(args.nn_model_dir),
        mapping_path=str(args.mhr2smpl_mapping_path),
        mhr_mesh_path=str(args.mhr_mesh_path) if args.mhr_mesh_path else None,
        smoother_dir=str(args.smoother_dir) if args.smoother_dir else None,
    )


def camera_reference(gravity):
    import numpy as np
    from mocap.core.gravity_alignment import build_camera_to_world_rotation

    if gravity is None:
        return np.eye(3, dtype=np.float64)
    gravity = np.asarray(gravity, dtype=np.float64)
    gravity /= np.linalg.norm(gravity)
    adjustment = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
    return adjustment @ build_camera_to_world_rotation(gravity)


def convert_pose(output, converter, rotation):
    """Repeat the publisher's conversion and protocol transforms unchanged."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mocap.utils.pose_protocol import prepare_publish_pose

    vertices = np.asarray(output["pred_vertices"], dtype=np.float32)
    translation = np.asarray(output["pred_cam_t"], dtype=np.float32)
    global_rotation = np.asarray(output["global_rot"], dtype=np.float64).reshape(3)
    body_rotation = Rotation.from_euler("ZYX", global_rotation)
    x180 = Rotation.from_euler("x", 180.0, degrees=True)
    body_quat_xyzw = (x180 * body_rotation).as_quat().astype(np.float64)
    smpl_pose, canonical_joints, _betas, _weights = converter.infer(
        [(vertices, translation)]
    )
    return prepare_publish_pose(body_quat_xyzw, canonical_joints, smpl_pose, rotation)


def create_visualizer(output_dir, estimator, fps, options):
    from pose_visualization import VideoVisualizations

    return VideoVisualizations(output_dir, estimator.faces, fps=fps, **options)


def extract_video(video_path, output_dir, estimator, converter, *, gravity=None,
                  bbox_threshold=0.5, verbose=False, progress_position=0,
                  visualization=None):
    """Process every decoded frame sequentially; save only single-person poses."""
    import cv2
    import numpy as np
    from tqdm.auto import tqdm

    rotation = camera_reference(gravity)
    capture = cv2.VideoCapture(str(video_path))
    timestamps, quaternions, joints, poses = [], [], [], []
    frames_read = no_person = multiple_people = 0
    start_time = time.time()
    visualizer = None
    visualization_error = None
    finished = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has invalid FPS ({fps}): {video_path}")

        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
        with tqdm(total=total, desc=video_path.name, unit="frame", dynamic_ncols=True,
                  position=progress_position, leave=progress_position == 0) as progress, \
                model_output(verbose):
            if visualization is not None:
                try:
                    visualizer = create_visualizer(output_dir, estimator, fps, visualization)
                except Exception as error:
                    visualization_error = error
                    LOGGER.error("Could not initialize visualizations: %s", error, exc_info=verbose)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_index = frames_read
                frames_read += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                outputs = estimator.process_one_image(
                    frame_rgb,
                    cam_int=None,
                    hand_box_source="yolo_pose",
                    bbox_thr=bbox_threshold,
                )
                smpl_record = None
                if len(outputs) != 1:
                    no_person += len(outputs) == 0
                    multiple_people += len(outputs) > 1
                else:
                    body_quat, smpl_joints, smpl_pose = convert_pose(outputs[0], converter, rotation)
                    # Match VideoFileSource: Unix start time + the frame's media timestamp.
                    timestamps.append(start_time + frame_index / fps)
                    quaternions.append(body_quat)
                    joints.append(smpl_joints)
                    poses.append(smpl_pose)
                    smpl_record = {
                        "frame_index": frame_index,
                        "body_quat": body_quat,
                        "smpl_joints": smpl_joints,
                        "smpl_pose": smpl_pose,
                    }
                if visualizer is not None and visualization_error is None:
                    try:
                        visualizer.write(frame_index, frame, outputs, smpl_record)
                    except Exception as error:
                        visualization_error = error
                        LOGGER.error("Visualization failed at frame %d: %s; continuing pose extraction",
                                     frame_index, error, exc_info=verbose)
                progress.set_postfix(poses=len(poses), ignorados=no_person + multiple_people,
                                     refresh=False)
                progress.update(1)
            finished = True
    finally:
        capture.release()
        if visualizer is not None:
            try:
                visualizer.close(commit=finished and visualization_error is None)
            except Exception as error:
                if visualization_error is None:
                    visualization_error = error

    if frames_read == 0:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_video": str(video_path),
        "fps": fps,
        "frames_read": frames_read,
        "poses_saved": len(poses),
        "frames_without_person": no_person,
        "frames_with_multiple_people": multiple_people,
        "timestamp_origin": start_time,
        "timestamp_units": "Unix seconds (origin + frame_index / fps)",
        "camera_intrinsics": "estimated per frame with MoGe2",
        "orientation_reference": "camera" if gravity is None else "gravity_aligned",
        "gravity": gravity,
        "R_world_cam": rotation.tolist(),
        "visualization": {
            "status": ("disabled" if visualization is None else
                       "failed" if visualization_error is not None else "complete"),
            "options": visualization,
            "error": str(visualization_error) if visualization_error is not None else None,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Publish the archive only after it is complete, so a failed run can be retried.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as file:
            temporary_path = Path(file.name)
            np.savez(
                file,
                timestamps=np.asarray(timestamps, dtype=np.float64),
                body_quats=np.asarray(quaternions, dtype=np.float64).reshape(-1, 4),
                smpl_joints=np.asarray(joints, dtype=np.float64).reshape(-1, 24, 3),
                smpl_poses=np.asarray(poses, dtype=np.float64).reshape(-1, 21, 3),
            )
        temporary_path.replace(output_dir / "smpl_data.npz")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if not poses:
        LOGGER.warning("No single-person poses found in %s; saved empty arrays", video_path)
    if visualization_error is not None:
        raise RuntimeError(
            f"SMPL data saved to {output_dir / 'smpl_data.npz'}, but visualizations failed: "
            f"{visualization_error}"
        ) from visualization_error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Video file or folder (searched recursively)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output",
                        help="Output folder (default: <repository>/output)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing archives")
    parser.add_argument("--verbose", action="store_true", help="Show original model diagnostics and tracebacks")
    parser.add_argument("--visualize", action="store_true",
                        help="Save live MHR panels and SMPL front/side videos")
    parser.add_argument("--save-images", action="store_true",
                        help="Save original, boxes, skeleton, mesh views and SMPL frame images")
    parser.add_argument("--save-meshes", action="store_true",
                        help="Save live MHR and SMPL meshes as PLY files")
    parser.add_argument("--visualize-all", action="store_true",
                        help="Enable visualization videos, frame images and PLY exports")
    parser.add_argument("--export-every", type=int, default=1,
                        help="Export images/PLY every N frames; videos keep all frames (default: 1)")
    parser.add_argument("--viz-height", type=int, default=480,
                        help="Even visualization video panel height in pixels (default: 480)")
    parser.add_argument("--video-codec", default="mp4v",
                        help="Four-character visualization video codec (default: mp4v)")
    parser.add_argument("--smpl-model-path", type=Path,
                        default=FASTSAM_DIR / "mhr2smpl/data/SMPL_NEUTRAL.pkl")
    parser.add_argument("--nn-model-dir", type=Path,
                        default=FASTSAM_DIR / "mhr2smpl/experiments/multiview_n30000_e500")
    parser.add_argument("--mhr2smpl-mapping-path", type=Path,
                        default=FASTSAM_DIR / "mhr2smpl/data/mhr2smpl_mapping.npz")
    parser.add_argument("--mhr-mesh-path", type=Path,
                        help="MHR PLY mesh, required for mappings using triangle_ids")
    parser.add_argument("--smoother-dir", type=Path, help="Optional original SMPL smoother")
    parser.add_argument("--yolo-model", type=Path,
                        default=FASTSAM_DIR / "checkpoints/yolo/yolo11m-pose.pt")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    parser.add_argument("--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
                        help="Optional gravity vector in camera coordinates; otherwise use camera reference")
    args = parser.parse_args(argv)
    if args.visualize_all:
        args.visualize = args.save_images = args.save_meshes = True
    if args.export_every < 1:
        parser.error("--export-every must be at least 1")
    if args.viz_height < 64 or args.viz_height % 2:
        parser.error("--viz-height must be an even integer of at least 64")
    if len(args.video_codec) != 4 or not args.video_codec.isascii():
        parser.error("--video-codec must contain four ASCII characters")
    if not 0 <= args.bbox_threshold <= 1:
        parser.error("--bbox-threshold must be between 0 and 1")
    if args.gravity is not None and (
        not all(math.isfinite(value) for value in args.gravity)
        or math.hypot(*args.gravity) < 1e-6
    ):
        parser.error("--gravity must be a finite, nonzero vector")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    try:
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        videos, base_dir = discover_videos(args.input)
        jobs = []
        for video in videos:
            output_dir = args.output_dir / video.relative_to(base_dir)
            if (output_dir / "smpl_data.npz").exists() and not args.overwrite:
                LOGGER.info("Skipping existing output: %s (use --overwrite to replace)", output_dir)
            else:
                jobs.append((video, output_dir))
        if not jobs:
            return 0
        if args.gravity is None:
            LOGGER.info("No gravity supplied: orientations use the camera reference, without gravity alignment")
        visualization = None
        if args.visualize or args.save_images or args.save_meshes:
            visualization = {
                "smpl_model_path": str(args.smpl_model_path),
                "save_videos": args.visualize,
                "save_images": args.save_images,
                "save_meshes": args.save_meshes,
                "export_every": args.export_every,
                "height": args.viz_height,
                "codec": args.video_codec,
            }
        failures = 0
        with logging_redirect_tqdm():
            with tqdm(total=1, desc="Carregando modelos", unit="etapa", leave=False,
                      dynamic_ncols=True) as loading, model_output(args.verbose):
                estimator = load_estimator(args)
                loading.update(1)
            with model_logging(args.verbose), \
                    tqdm(jobs, desc="Vídeos", unit="vídeo", dynamic_ncols=True,
                      disable=len(jobs) == 1, position=0) as videos_progress:
                for video, output_dir in videos_progress:
                    try:
                        with model_output(args.verbose):
                            converter = load_converter(args)
                        extract_video(video, output_dir, estimator, converter,
                                      gravity=args.gravity, bbox_threshold=args.bbox_threshold,
                                      verbose=args.verbose,
                                      progress_position=1 if len(jobs) > 1 else 0,
                                      visualization=visualization)
                    except Exception as error:
                        LOGGER.error("Failed to process %s: %s", video, error,
                                     exc_info=args.verbose)
                        failures += 1
        return 1 if failures else 0
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
