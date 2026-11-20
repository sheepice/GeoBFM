"""One pinhole/mount contract for DA2 training, simulation, export and RGB capture."""
import math
from dataclasses import replace


def configure_camera(sensor, camera):
    width, height = int(camera['width']), int(camera['height'])
    if (width, height) != (112, 112):
        raise ValueError('DA2 expects 112x112 pixels')
    if camera.get('fov_policy') == 'preserve_previous_112x64_horizontal_fov':
        # Historical checkpoints retain their original optics for reproducibility.
        fovy = math.degrees(2 * math.atan(math.tan(math.radians(sensor.fovy / 2)) * 112 / 64))
    else:
        fovy = float(camera['fovy'])
        fovx = math.degrees(2 * math.atan(math.tan(math.radians(fovy / 2)) * width / height))
        if not 0 < fovy < 180 or not math.isclose(fovx, camera['fovx'], abs_tol=1e-8):
            raise ValueError('Camera horizontal and vertical FOV do not match its aspect ratio')
    mount = {key: tuple(camera[key]) for key in ('pos', 'quat') if key in camera}
    if 'parent_body' in camera:
        mount['parent_body'] = camera['parent_body']
    return replace(sensor, width=width, height=height, fovy=fovy, data_types=('rgb',), **mount)


def camera_contract(sensor):
    focal = sensor.height / (2 * math.tan(math.radians(sensor.fovy / 2)))
    return dict(width=sensor.width, height=sensor.height, fovy=float(sensor.fovy),
        fovx=math.degrees(2 * math.atan(sensor.width / (2 * focal))),
        parent_body=sensor.parent_body, pos=list(sensor.pos), quat=list(sensor.quat),
        quaternion_order='wxyz', convention='opengl',
        optical_axes='+X right, +Y up, -Z forward',
        fx=focal, fy=focal, cx=(sensor.width - 1) / 2, cy=(sensor.height - 1) / 2,
        image_processing='project target pixel-center rays through measured color intrinsics and distortion; bilinear sample; no stretch or padding')


def assert_sim_camera(raw, contract):
    """Check compiled MuJoCo optics and mount, not merely a config dictionary."""
    import numpy as np
    camera = raw.scene['head_cam']
    model = raw.sim._mj_model
    idx = camera.camera_idx
    np.testing.assert_allclose(model.cam_fovy[idx], contract['fovy'], atol=1e-6)
    np.testing.assert_allclose(model.cam_pos[idx], contract['pos'], atol=1e-7)
    a, b = np.asarray(model.cam_quat[idx]), np.asarray(contract['quat'])
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    assert abs(float(a @ b)) > 1 - 1e-8
    return contract
