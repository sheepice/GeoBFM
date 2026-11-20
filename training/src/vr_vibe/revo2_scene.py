"""Official Revo2 URDF import and a physical table-to-stairs scene."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from vibe_grail_joint.deploy.observations import JOINT_NAMES


def import_hand(directory, side):
    directory = Path(directory).resolve()
    source = directory / "urdf" / f"revo2_{side}_hand.urdf"
    root = ET.parse(source).getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", mesh.get("filename").replace("package://revo2_description", str(directory)))
    compiler = ET.SubElement(ET.SubElement(root, "mujoco"), "compiler")
    compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false", balanceinertia="true")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    independent = []
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.get("name")
        hinge = spec.joint(name)
        hinge.armature = 0.0001
        hinge.damping[0] = 0.015
        mimic = joint.find("mimic")
        if mimic is not None:
            equality = spec.add_equality()
            equality.name = name + "_coupling"
            equality.type = mujoco.mjtEq.mjEQ_JOINT
            equality.name1, equality.name2 = name, mimic.get("joint")
            equality.data[:5] = [float(mimic.get("offset", "0")), float(mimic.get("multiplier", "1")), 0, 0, 0]
            equality.solref[:] = [0.005, 1]
        else:
            independent.append(name)
            actuator = spec.add_actuator(name=name + "_drive", target=name,
                trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = 5.0
            actuator.biasprm[:3] = [0, -5.0, -0.08]
            effort = float(joint.find("limit").get("effort"))
            actuator.forcelimited = True
            actuator.forcerange[:] = [-effort, effort]
            actuator.ctrllimited = True
            actuator.ctrlrange[:] = hinge.range
    for index, geom in enumerate(spec.geoms):
        geom.name = f"{side}_hand_geom_{index}"
        if geom.contype or geom.conaffinity:
            geom.group = 3
            # The URDF contains overlapping adjacent pad shells. Disable internal
            # hand mesh contacts, retaining contacts with robot, object and scene.
            geom.contype, geom.conaffinity = 2, 1
            geom.condim = 4
            geom.friction[:] = [1.2, 0.01, 0.001]
            geom.solref[:] = [0.006, 1]
        else:
            geom.group = 2
            geom.rgba[:] = [.18, .23, .28, 1] if "touch" in geom.meshname else [.8, .83, .86, 1]
    return spec, independent


def make_scene(settings, hand_directory=None, *, stairs_height=.20, stairs_start=2.0,
               table=True, object_position=(.35, -.30, .875), object_mass=.06,
               table_center=(.43, -.54, .79)):
    import mjlab  # initialize the asset registry before mocke
    from mjlab.entity import Entity
    from mocke.sonic.profile import robot_cfg

    cfg = robot_cfg()
    spec = Entity(cfg).spec
    spec.meshdir = str(Path(spec.modelfiledir) / spec.meshdir)
    for key in list(spec.keys):
        spec.delete(key)
    default = np.zeros(29)
    for i, name in enumerate(JOINT_NAMES):
        for pattern, value in cfg.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                default[i] = value
    hand_drives = {}
    if hand_directory:
        for side in ("left", "right"):
            wrist = spec.body(f"{side}_wrist_yaw_link")
            for geom in list(wrist.geoms):
                if "hand" in geom.name or "rubber_hand" in geom.meshname:
                    spec.delete(geom)
            hand, joints = import_hand(hand_directory, side)
            # Hand +Z points along the wrist +X; thumb points toward wrist +Z.
            matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            if side == "left":
                matrix = matrix @ np.diag([-1, -1, 1])
            quat = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
            frame = wrist.add_frame(name=f"{side}_revo_mount", pos=[.0415, 0, 0], quat=quat)
            prefix = f"revo_{side}_"
            spec.attach(hand, prefix=prefix, frame=frame)
            hand_drives[side] = [prefix + name + "_drive" for name in joints]
    spec.option.timestep = .0025
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 30
    spec.option.ls_iterations = 20
    spec.stat.extent = 5
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient[:] = [.45, .45, .45]
    spec.visual.headlight.diffuse[:] = [.65, .65, .65]
    spec.visual.map.znear = .005
    spec.worldbody.add_light(name="key", pos=[1, -3, 6], dir=[0, .3, -1],
        diffuse=[.8, .8, .8], castshadow=True)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[25, 25, .1], rgba=[.32, .35, .39, 1], friction=[.8, .005, .0001])
    for i in range(6):
        height = (i + 1) * stairs_height
        if height > 0:
            spec.worldbody.add_geom(name=f"step_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[stairs_start + .16 + .32 * i, 0, height / 2],
                size=[.16, 1.4, height / 2], rgba=[.60, .43, .25, 1], friction=[.8, .005, .0001])
    if stairs_height > 0:
        spec.worldbody.add_geom(name="landing", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[stairs_start + 1.92 + 1.8, 0, 3 * stairs_height],
            size=[1.8, 1.4, 3 * stairs_height], rgba=[.60, .43, .25, 1])
    if table:
        spec.worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=table_center, size=[.32, .30, .025], rgba=[.26, .38, .43, 1],
            friction=[.8, .005, .0001])
        for x in (table_center[0]-.25, table_center[0]+.25):
            for y in (table_center[1]-.24, table_center[1]+.24):
                spec.worldbody.add_geom(name=f"leg_{x}_{y}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, y, .385], size=[.025, .025, .385], rgba=[.12, .15, .19, 1])
        obj = spec.worldbody.add_body(name="payload", pos=object_position)
        obj.add_freejoint(name="payload_free")
        obj.add_geom(name="payload_collision", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[.015, .06, 0], mass=object_mass*.75, rgba=[1.0, .39, .06, 1],
            friction=[1.2, .01, .001], condim=4, solref=[.006, 1])
        obj.add_geom(name="payload_cap", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[.025, .008, 0], pos=[0, 0, .065], mass=object_mass*.25,
            rgba=[.12, .20, .24, 1], friction=[1.2, .01, .001], condim=4, solref=[.006, 1])
    camera = settings["camera"]
    spec.body("torso_link").add_camera(name="head_cam", pos=camera["pos"], quat=camera["quat"], fovy=camera["fovy"])
    model = spec.compile()
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    aids = np.array([model.actuator(n).id for n in JOINT_NAMES])
    scale = .25 * model.actuator_forcerange[aids, 1] / model.actuator_gainprm[aids, 0]
    return spec, model, dict(default=default, qadr=qadr, vadr=vadr, aids=aids, scale=scale,
        hand_drives={side: [model.actuator(n).id for n in names] for side, names in hand_drives.items()})
