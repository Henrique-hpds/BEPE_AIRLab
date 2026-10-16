"""Reconstruct standard SMPL parameters in OpenCV camera coordinates (metres)."""

import numpy as np
from scipy.spatial.transform import Rotation


def smpl_geometry(model, device, parameters):
    """Evaluate standard SMPL inputs without centering or changing coordinates."""
    import torch

    inputs = {
        name: torch.as_tensor(np.asarray(parameters[name]).copy(), dtype=torch.float32,
                              device=device).reshape(1, -1)
        for name in ("body_pose", "global_orient", "betas", "transl")
    }
    with torch.no_grad():
        output = model(**inputs)
    return (output.vertices[0].detach().cpu().numpy().copy(),
            output.joints[0, :24].detach().cpu().numpy().copy())


def fit_rigid_smpl(canonical_vertices, canonical_pelvis, camera_vertices):
    """Fit corresponding vertices, accounting for SMPL's rotation about its pelvis.

    SMPL evaluates R @ (v - pelvis) + pelvis + transl, rather than R @ v + transl.
    Return its axis-angle global_orient and additive transl, without fitting scale.
    """
    source = np.asarray(canonical_vertices, dtype=np.float64)
    target = np.asarray(camera_vertices, dtype=np.float64)
    if (source.shape != target.shape or source.ndim != 2
            or source.shape[1] != 3 or len(source) < 3):
        raise ValueError("SMPL alignment requires matching (V, 3) vertex arrays")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("SMPL alignment vertices must be finite")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, singular_values, vt = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    if singular_values[0] <= 0 or singular_values[1] <= singular_values[0] * 1e-10:
        raise ValueError("SMPL alignment needs non-collinear corresponding vertices")
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    pelvis = np.asarray(canonical_pelvis, dtype=np.float64).reshape(3)
    translation = target_center - rotation @ (source_center - pelvis) - pelvis
    return Rotation.from_matrix(rotation).as_rotvec(), translation


def camera_smpl_parameters(converter, vertices, camera_translation, pose, betas):
    """Preserve shape and recover placement from the converter's vertex mapping.

    The upstream converter returns only root-relative joints. Reuse its loaded
    SMPL model, expanded barycentric mapping and sample indices to align a posed
    SMPL mesh to the MHR prediction directly in camera space. No protocol/frame
    rotations or gravity alignment are applied to these standard parameters.
    """
    body_pose = np.zeros((23, 3), dtype=np.float64)
    body_pose[:21] = np.asarray(pose).reshape(21, 3)
    parameters = {
        "body_pose": body_pose.reshape(69),
        "global_orient": np.zeros(3, dtype=np.float64),
        "betas": np.asarray(betas, dtype=np.float64).reshape(10),
        "transl": np.zeros(3, dtype=np.float64),
    }
    canonical_vertices, canonical_joints = smpl_geometry(
        converter._smpl, converter.device, parameters,
    )
    # Mapping indices have already been expanded from triangles by the converter.
    mapped_vertices = (
        np.asarray(vertices)[converter._mhr_vert_ids]
        * converter._baryc[:, :, None]
    ).sum(axis=1) + np.asarray(camera_translation)
    sample = converter._sample_idx
    parameters["global_orient"], parameters["transl"] = fit_rigid_smpl(
        canonical_vertices[sample], canonical_joints[0], mapped_vertices[sample],
    )
    return parameters
