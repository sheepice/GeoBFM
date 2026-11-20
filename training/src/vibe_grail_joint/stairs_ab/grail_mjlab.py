"""Historical GRAIL actor transfer to the common fixed-stair evaluation plant.

This is explicitly a transfer test, not a reproduction of the native GRAIL
training environment or an equal-budget algorithm comparison.
"""
import hashlib
import json
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul, matrix_from_quat


def command_observation(q, dq, reference_quat, root_quat):
    relative = quat_mul(quat_conjugate(root_quat[:, None].expand_as(reference_quat)), reference_quat)
    rotation = matrix_from_quat(relative)[..., :2].flatten(-2)
    return torch.cat((q, dq, rotation), -1).flatten(1)


def object_observation(root_pos, root_quat, object_pos, object_quat):
    local = quat_apply_inverse(root_quat, object_pos-root_pos)
    rotation = matrix_from_quat(quat_mul(quat_conjugate(root_quat), object_quat))[..., :2].flatten(-2)
    identity = torch.eye(3, device=root_pos.device)[:, :2].flatten()
    return torch.cat((local, rotation, root_pos.new_zeros(len(root_pos), 30),
                      identity.repeat(10).expand(len(root_pos), -1)), -1)


def projected_yaw_quat(quat):
    basis_x = torch.zeros(*quat.shape[:-1], 3, device=quat.device); basis_x[..., 0] = 1
    basis_z = torch.zeros_like(basis_x); basis_z[..., 2] = 1
    x, z = quat_apply(quat, basis_x), quat_apply(quat, basis_z)
    fallback = torch.where(x[..., 2:3] < 0, z[..., :2], -z[..., :2])
    xy = torch.where(x[..., :2].norm(dim=-1, keepdim=True) > .1, x[..., :2], fallback)
    yaw = torch.atan2(xy[..., 1], xy[..., 0])
    result = torch.zeros_like(quat); result[..., 0] = (yaw/2).cos(); result[..., 3] = (yaw/2).sin()
    return result


class GrailMjlabActor:
    def __init__(self, settings, checkpoint, raw):
        self.raw = raw; self.c = raw.command_manager.get_term('motion')
        self.robot = raw.scene['robot']; self.device = raw.device
        baseline = settings['evaluation_baseline']
        package = Path(baseline['export_directory'])
        self.contract = json.loads((package/'contract.json').read_text())
        digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if digest != self.contract['source_sha256'] or digest != baseline['checkpoint_sha256']:
            raise ValueError('GRAIL checkpoint identity mismatch')
        self.actor = torch.jit.load(str(package/'actor.ts'), map_location=self.device).eval()
        names = list(self.robot.joint_names)
        self.obs_ids = [names.index(n) for n in self.contract['action_joint_names']]
        self.ref_ids = [names.index(n) for n in self.contract['command_joint_names']]
        self.act_ids = [names.index(n) for n in self.contract['action_joint_names']]
        native_names = self.contract['joint_names']
        self.defaults = torch.tensor([self.contract['default_joint_pos'][native_names.index(n)]
                                     for n in self.contract['action_joint_names']], device=self.device)
        self.scales = torch.tensor(self.contract['action_scale'], device=self.device)
        if any(self.contract['position_offset']): raise ValueError('Nonzero native joint offset requires explicit handling')
        self.action_term = raw.action_manager.get_term('joint_pos')
        self.history = None
        self.actions = torch.zeros(raw.num_envs, self.contract['action_history'], 29, device=self.device)
        x = torch.linspace(-.75, .75, 11, device=self.device)
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        self.grid = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)
        self.audit = dict(checkpoint_sha256=digest, source_native_robot=self.contract['native_robot'],
            evaluation_robot='common ViBe/DA2 G1 plant; historical mode-15 plant NOT retained',
            reference='same terrain-blind flat SONIC planner as the other methods',
            environment_input='privileged 11x11 height scan and static object pose',
            static_object_frame='first riser x, tile centerline y, base ground z; identity rotation',
            action_history=self.contract['action_history'], proprio_history=10,
            transfer_limitations=['native plant differs', 'native paired stairs reference replaced by flat planner',
                                 'procedural stairs use an explicitly chosen static object frame'],
            native_action_scale=self.contract['action_scale'], joint_mapping=self.act_ids,
            native_observation_dimensions=self.contract['observation_dims'])

    def __call__(self, obs, stochastic_output=False):
        if stochastic_output: raise ValueError('Fixed evaluation requires deterministic mean actions')
        r, c = self.robot, self.c
        values = (r.data.root_com_ang_vel_b, r.data.projected_gravity_b,
                  r.data.joint_pos[:, self.obs_ids], r.data.joint_vel[:, self.obs_ids])
        if self.history is None:
            self.history = [v[:, None].repeat(1, 10, 1) for v in values]
        else:
            self.history = [torch.cat((v[:, None], h[:, :-1]), 1) for v,h in zip(values,self.history)]
        root, quat = r.data.root_link_pos_w, r.data.root_link_quat_w
        object_pos = c.tile_origins[c.tile].clone()
        object_pos[:, 0] += torch.tensor([t['edges'][0] for t in c.cfg.random_tiles], device=self.device)[c.tile]
        object_pos[:, 2] += torch.tensor([t['base'] for t in c.cfg.random_tiles], device=self.device)[c.tile]
        identity = torch.zeros_like(quat); identity[:, 0] = 1
        policy = torch.cat((*[h.flatten(1) for h in self.history], self.actions.flatten(1),
                            object_observation(root, quat, object_pos, identity)), -1)
        frames = c.future_frames(10, skip=5)
        command = command_observation(c.motion.joint_pos[frames][:, :, self.ref_ids],
            c.motion.joint_vel[frames][:, :, self.ref_ids], c.motion.body_quat_w[frames, 0], quat)
        yaw = projected_yaw_quat(quat)
        query = quat_apply(yaw[:, None].expand(-1,121,-1), self.grid[None].expand(len(root),-1,-1)) + root[:, None]
        query[..., 2] = c.ground_height(query[..., :2])
        terrain = quat_apply_inverse(quat[:, None].expand(-1,121,-1), query-root[:,None])[...,2].clamp(-5,5)
        native = self.actor(policy, command, terrain)[0]
        self.actions = torch.cat((native[:, None], self.actions[:, :-1]), 1)
        targets = torch.empty_like(native)
        targets[:, self.act_ids] = self.defaults + native*self.scales
        # Invert the common action transform. Preserve native absolute targets,
        # while explicitly retaining the common benchmark plant and PD gains.
        return (targets-self.action_term._offset)/self.action_term._scale


def load_grail(settings, checkpoint, raw):
    actor = GrailMjlabActor(settings, checkpoint, raw)
    return actor, actor.contract['checkpoint_label']
