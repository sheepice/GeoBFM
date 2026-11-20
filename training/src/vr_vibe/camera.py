from da2_vibe.paths import repo_root
"""Explicit RGB pinhole contract shared by training, export and hardware preprocessing."""

import hashlib
import json
import math
from dataclasses import replace
import numpy as np


def resolve_camera(camera):
    """Resolve the shared file and reject a stale per-experiment override."""
    from pathlib import Path

    camera = dict(camera)
    if "contract_file" not in camera:
        return camera
    path = Path(camera["contract_file"])
    if not path.is_absolute():
        path = repo_root() / "configs" / path
    shared = json.loads(path.read_text())
    protected = (
        "width",
        "height",
        "fovy",
        "fovx",
        "fx",
        "fy",
        "cx",
        "cy",
        "parent_body",
        "pos",
        "quat",
    )
    for key in protected:
        if key not in camera or key not in shared:
            continue
        if isinstance(shared[key], str):
            same = camera[key] == shared[key]
        else:
            same = np.allclose(camera[key], shared[key], rtol=0, atol=1e-10)
        if not same:
            raise ValueError(f"Camera override differs from shared contract: {key}")
    return dict(shared, **camera)


def configure_camera(sensor, camera):
    camera = resolve_camera(camera)
    width, height = int(camera["width"]), int(camera["height"])
    if (width, height) != (112, 112):
        raise ValueError("Keep the reviewed 112x112 RGB input")
    # Legacy configurations retain their existing nominal optics.
    fovy = float(camera.get("fovy", sensor.fovy))
    if not 0 < fovy < 180:
        raise ValueError("Invalid vertical FOV")
    fovx = math.degrees(2 * math.atan(math.tan(math.radians(fovy / 2)) * width / height))
    if "fovx" in camera and not math.isclose(fovx, float(camera["fovx"]), abs_tol=1e-8):
        raise ValueError("FOV and aspect ratio are inconsistent with square pixels")
    mount = {k: tuple(camera[k]) for k in ("pos", "quat") if k in camera}
    if "parent_body" in camera:
        mount["parent_body"] = camera["parent_body"]
    return replace(sensor, width=width, height=height, fovy=fovy, data_types=("rgb",), **mount)


def camera_contract(sensor):
    fy = sensor.height / (2 * math.tan(math.radians(sensor.fovy / 2)))
    return dict(
        version=1,
        model="Intel RealSense D435i RGB",
        width=sensor.width,
        height=sensor.height,
        fovy=float(sensor.fovy),
        fovx=math.degrees(2 * math.atan(sensor.width / (2 * fy))),
        fx=fy,
        fy=fy,
        cx=(sensor.width - 1) / 2,
        cy=(sensor.height - 1) / 2,
        parent_body=sensor.parent_body,
        pos=list(sensor.pos),
        quat=list(sensor.quat),
        quaternion_order="wxyz",
        convention="opengl",
        optical_axes="+X right,+Y up,-Z forward",
        image_processing="project target pixel-center rays through measured color intrinsics and distortion; bilinear sample; no stretch or padding",
        mount_calibration="nominal mount; not a measured hardware calibration",
    )


def contract_sha256(contract):
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def assert_sim_camera(raw, contract):
    camera = raw.scene["head_cam"]
    model = raw.sim._mj_model
    idx = camera.camera_idx
    np.testing.assert_allclose(model.cam_fovy[idx], contract["fovy"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(model.cam_pos[idx], contract["pos"], atol=1e-7, rtol=0)
    a = np.asarray(model.cam_quat[idx])
    b = np.asarray(contract["quat"])
    if abs(float((a / np.linalg.norm(a)) @ (b / np.linalg.norm(b)))) < 1 - 1e-8:
        raise ValueError("Compiled camera orientation differs from contract")
    return dict(
        contract,
        compiled_fovy=float(model.cam_fovy[idx]),
        compiled_pos=model.cam_pos[idx].tolist(),
        compiled_quat=model.cam_quat[idx].tolist(),
    )


def target_rays(contract):
    y, x = np.mgrid[: contract["height"], : contract["width"]]
    return np.stack(
        (
            (x - contract["cx"]) / contract["fx"],
            (y - contract["cy"]) / contract["fy"],
            np.ones_like(x),
        ),
        -1,
    )


def sampling_map(source_intrinsics, contract, projector=None):
    """Map the target crop to source RGB pixel centers; never ignore lens distortion.

    projector maps a 3D camera point to source pixels, e.g. the RealSense SDK's
    rs2_project_point_to_pixel with measured color intrinsics.
    """
    rays = target_rays(contract)
    coeffs = np.asarray(source_intrinsics.get("coeffs", []), dtype=float)
    if projector is None:
        if np.any(coeffs != 0):
            raise ValueError("Distorted color intrinsics require an SDK projection function")
        uv = rays[..., :2] * [source_intrinsics["fx"], source_intrinsics["fy"]]
        uv += [source_intrinsics["ppx"], source_intrinsics["ppy"]]
    else:
        uv = np.asarray(
            [projector(point.tolist()) for point in rays.reshape(-1, 3)], dtype=float
        ).reshape(*rays.shape[:2], 2)
    if not np.isfinite(uv).all():
        raise ValueError("Nonfinite source pixel mapping")
    width, height = int(source_intrinsics["width"]), int(source_intrinsics["height"])
    if (
        uv[..., 0].min() < 0
        or uv[..., 1].min() < 0
        or uv[..., 0].max() > width - 1
        or uv[..., 1].max() > height - 1
    ):
        raise ValueError("Target FOV exceeds source coverage; do not pad or stretch")
    return uv


def realsense_sampling_map(intrinsics, contract):
    import pyrealsense2 as rs

    source = dict(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        ppx=intrinsics.ppx,
        ppy=intrinsics.ppy,
        coeffs=list(intrinsics.coeffs),
    )
    return sampling_map(source, contract, lambda p: rs.rs2_project_point_to_pixel(intrinsics, p))


def resample_rgb(image, uv):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected RGB uint8")
    x, y = uv[..., 0], uv[..., 1]
    if x.min() < 0 or y.min() < 0 or x.max() > image.shape[1] - 1 or y.max() > image.shape[0] - 1:
        raise ValueError("Sampling coordinates exceed image")
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    ax = (x - x0)[..., None]
    ay = (y - y0)[..., None]
    value = (image[y0, x0] * (1 - ax) + image[y0, x1] * ax) * (1 - ay) + (
        image[y1, x0] * (1 - ax) + image[y1, x1] * ax
    ) * ay
    return np.rint(value).clip(0, 255).astype(np.uint8)
