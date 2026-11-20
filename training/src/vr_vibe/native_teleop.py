"""Named-joint MuJoCo runtime for the existing DA2 TELEOP checkpoint."""
import numpy as np
import torch
from tensordict import TensorDict

from da2_vibe.paths import load_checkpoint
from vibe_grail_joint.deploy.observations import JOINT_NAMES, ObservationBuilder, RobotState, Reference
from vibe_grail_joint.da2_locomotion.encoder import FrozenVisualEncoder
from vibe_grail_joint.da2_locomotion.history import VisualHistory
from vibe_grail_joint.velocity.planner import PlannerPool
from .contracts import inv, mul, points_from_bodies, teleop_input
from .fk import RobotFK


class NativeTeleop:
    def __init__(self, settings, checkpoint, model, mapping, device="cuda:0"):
        from .da2_task import agent_config
        from .da2_model import DA2TeleopActor
        self.settings, self.device, self.mapping = settings, device, mapping
        self.builder = ObservationBuilder(dict(joint_names=JOINT_NAMES, default_joint_pos=mapping["default"]))
        seed = np.r_[0, 0, .76, 1, 0, 0, 0, mapping["default"]]
        self.pool = PlannerPool(settings["planner"], seed, [0., .4], seed=261115)
        self.channels = self.pool.channels()
        self.fk = RobotFK(device)
        self.encoder = FrozenVisualEncoder(settings["visual"], device)
        self.visual = VisualHistory(1, 2, 64, 96, device)
        self.step = 0
        self.previous_action = np.zeros(29)
        fixture = dict(policy=torch.zeros(1, 930, device=device), tokenizer=torch.zeros(1, 640, device=device),
            augmentation=torch.zeros(1, 6, device=device), velocity_command=torch.zeros(1, 8, device=device),
            q_proprio=torch.zeros(1, 64, device=device), teleop_reference=torch.zeros(1, 267, device=device),
            reference_mode=torch.ones(1, 1, device=device),
            kv_tokens=TensorDict(dict(img_tokens=self.visual.snapshot()), batch_size=[1]))
        obs = TensorDict(fixture, batch_size=[1])
        cfg = agent_config(settings["bank_root"] + "/rank_0", settings, 20)
        acfg = dict(cfg["actor"])
        acfg.pop("class_name")
        acfg["cache_tokens"] = False
        self.actor = DA2TeleopActor(obs=obs, obs_groups=cfg["obs_groups"], obs_set="actor", output_dim=29, **acfg).to(device).eval()
        state = load_checkpoint(checkpoint, map_location=device, weights_only=False)
        if state["infos"]["vr_vibe"]["version"] != 2:
            raise ValueError("Expected DA2 TELEOP checkpoint version 2")
        saved_camera = state["infos"]["da2_locomotion"]["camera"]
        for name in ("pos", "quat", "fovy", "width", "height"):
            if saved_camera[name] != settings["camera"][name]:
                raise ValueError("Camera differs from trained checkpoint")
        self.actor.load_state_dict(state["actor_state_dict"], strict=True)
        self.iteration = int(state["iter"])
        self.actor.requires_grad_(False)

    def reference(self, speed):
        stream = 0 if speed == 0 else 1
        index = stream * self.pool.capacity + self.pool.cursor
        ids = index + np.arange(10) * 5
        c = self.channels
        return Reference(c["joint_pos"][ids], c["joint_vel"][ids], c["body_quat_w"][ids, 0],
            c["body_lin_vel_w"][index, 0], c["body_ang_vel_w"][index, 0])

    @torch.inference_mode()
    def default_targets(self, speed=0.):
        joints = torch.as_tensor(self.reference(speed).joint_pos[:1], device=self.device)
        p, q = self.fk(joints)
        points, orientations = points_from_bodies(p, q, self.fk.body_names,
            joints.new_zeros(1, 3), joints.new_tensor([[1., 0, 0, 0]]))
        return points[0].cpu().numpy(), orientations[0].cpu().numpy()

    @torch.inference_mode()
    def act(self, data, rgb, speed, upper_joints=None, points=None, orientations=None):
        m = self.mapping
        ref = self.reference(speed)
        state = RobotState(data.qpos[m["qadr"]].copy(), data.qvel[m["vadr"]].copy(),
            data.qpos[3:7].copy(), data.qvel[3:6].copy(), data.time)
        values = self.builder.build(state, ref, np.zeros((21, 192)), np.zeros(192), speed)
        values = {k: v for k, v in values.items() if k not in ("kv_tokens__img_tokens", "q_cls", "q_task_cmd")}
        values = {k: torch.as_tensor(v, device=self.device) for k, v in values.items()}
        if self.step % self.settings["visual"]["update_interval_steps"] == 0:
            self.visual.push(torch.tensor([0], device=self.device),
                self.encoder(torch.as_tensor(rgb.copy()[None], device=self.device)), self.step)
        values["kv_tokens"] = TensorDict(dict(img_tokens=self.visual.snapshot()), batch_size=[1])
        current = torch.as_tensor(ref.joint_pos[0].copy()[None], device=self.device)
        if upper_joints is not None:
            current[:, 12:] = torch.as_tensor(upper_joints, device=self.device)
        body_p, body_q = self.fk(current)
        zero, identity = current.new_zeros(1, 3), current.new_tensor([[1., 0, 0, 0]])
        target_p, target_q = points_from_bodies(body_p, body_q, self.fk.body_names, zero, identity)
        if points is not None:
            target_p = current.new_tensor(points[None])
            target_q = current.new_tensor(orientations[None])
        relative = mul(inv(current.new_tensor(state.quat[None])), current.new_tensor(ref.root_quat[:1]))
        values["teleop_reference"] = teleop_input(current.new_tensor(ref.joint_pos[None, :, :12]),
            current.new_tensor(ref.joint_vel[None, :, :12]), target_p, target_q, relative)
        values["reference_mode"] = current.new_ones(1, 1)
        obs = TensorDict(values, batch_size=[1])
        action = self.actor(obs, stochastic_output=False)[0].cpu().numpy()
        if not np.isfinite(action).all():
            raise RuntimeError("Nonfinite checkpoint action")
        self.builder.accept_action(action)
        self.step += 1
        changed, _ = self.pool.advance()
        if changed:
            self.channels = self.pool.channels()
        return m["default"] + m["scale"] * action
