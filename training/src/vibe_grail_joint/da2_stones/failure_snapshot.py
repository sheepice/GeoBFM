"""Capture only failing worlds before a nonfinite reward terminates a run."""
import os
from pathlib import Path
import torch


def nonfinite_world_masks(guidance):
    """Check before episode resets; never substitute invalid physical states."""
    c = guidance.c
    values = dict(guidance.rewards)
    values.update(root_position=c.robot.data.root_link_pos_w,
        body_position=c.robot_body_pos_w, body_quaternion=c.robot_body_quat_w,
        body_linear_velocity=c.robot_body_lin_vel_w, body_angular_velocity=c.robot_body_ang_vel_w,
        joint_position=c.robot_joint_pos, joint_velocity=c.robot_joint_vel)
    values.update({'quality/' + name: value for name, value in guidance.quality.step_data.items()})
    masks = {name: ~torch.isfinite(value).reshape(value.shape[0], -1).all(-1)
             for name, value in values.items()}
    masks['active_contact_substep'] = guidance.quality.nonfinite_contact
    return masks


def assert_finite(guidance):
    masks = nonfinite_world_masks(guidance)
    if torch.stack(list(masks.values())).any():
        detail = save_nonfinite_snapshot(guidance, masks)
        raise RuntimeError('Nonfinite stone reward or physical state: ' + detail)


def save_nonfinite_snapshot(guidance, masks=None):
    env, c = guidance.env, guidance.c
    masks = nonfinite_world_masks(guidance) if masks is None else masks
    bad_terms = {name: value.nonzero().flatten() for name, value in masks.items() if value.any()}
    bad = torch.unique(torch.cat(list(bad_terms.values())))
    ids = bad[:16]
    def selected(value):
        return value[ids].detach().cpu().clone()
    contacts = {}
    for side in ('left','right'):
        for prefix in ('foothold_', 'quality_shin_'):
            d = env.scene[prefix+side].data
            contacts[prefix+side] = {name:selected(getattr(d,name)) for name in ('found','force','normal','pos')}
    snapshot = dict(version=2, rank=int(os.environ.get('RANK', 0)), step=int(env.common_step_counter),
        failing_world_count=len(bad), sampled_world_ids=ids.cpu(),
        bad_terms={name:value.cpu() for name,value in bad_terms.items()},
        rewards={name:selected(value) for name,value in guidance.rewards.items()},
        quality={name:selected(value) for name,value in guidance.quality.step_data.items()},
        contacts=contacts, tile=selected(c.tile), speed=selected(c.speed),
        root_position=selected(c.robot.data.root_link_pos_w),
        body_position=selected(c.robot_body_pos_w), body_quaternion=selected(c.robot_body_quat_w),
        body_linear_velocity=selected(c.robot_body_lin_vel_w), body_angular_velocity=selected(c.robot_body_ang_vel_w),
        joint_position=selected(c.robot_joint_pos),joint_velocity=selected(c.robot_joint_vel))
    if hasattr(env, '_diagnostic_last_actions'):
        snapshot['actions'] = selected(env._diagnostic_last_actions)
    directory = getattr(env, 'failure_snapshot_dir', None)
    if directory is None:
        return f'nonfinite terms={list(bad_terms)} worlds={bad[:16].tolist()} (no snapshot directory configured)'
    path = Path(directory) / f'nonfinite_rank{snapshot["rank"]}_step{snapshot["step"]}.pt'
    torch.save(snapshot, str(path)+'.tmp')
    os.replace(str(path)+'.tmp',path)
    return f'nonfinite terms={list(bad_terms)} worlds={bad[:16].tolist()} snapshot={path}'
