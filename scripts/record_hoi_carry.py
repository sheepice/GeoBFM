"""Run a first-frame HOI policy rollout and film a continuous physical trial."""

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import matrix_from_quat

from da2_vibe.paths import load_settings
from hoi_carry_actor import CarryActor, ReferenceCarryActor
from hoi_carry_task import config, interaction
from score_hoi_tracking import UpperBodyTracking
from vr_vibe.da2_task import agent_config


def render(model, trace, dest, iteration):
    poses = trace["qpos"]
    model.vis.global_.offwidth = 896
    model.vis.global_.offheight = 560
    model.vis.headlight.ambient[:] = 0.5
    model.vis.headlight.diffuse[:] = 0.7
    d = mujoco.MjData(model)
    pelvis = model.body("robot/pelvis").id
    box = model.body("box/object").id
    centers = []
    for q in poses:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        centers.append([d.xpos[pelvis, :2].copy(), d.xpos[box, :2].copy()])
    centers = np.asarray(centers)
    mid = len(centers) // 2
    view = centers[mid, 1] - centers[mid, 0]
    view /= max(np.linalg.norm(view), 1e-6)
    side = np.array([-view[1], view[0]])
    direction = view + 0.85 * side
    cam = mujoco.MjvCamera()
    cam.azimuth = np.degrees(np.arctan2(-direction[1], -direction[0]))
    cam.elevation = -13
    cam.distance = 2.85
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    with (
        mujoco.Renderer(model, height=560, width=896) as renderer,
        imageio.get_writer(
            dest,
            fps=50,
            codec="libx264",
            ffmpeg_params=["-threads", "2", "-crf", "18", "-movflags", "+faststart"],
        ) as writer,
    ):
        for i, q in enumerate(poses):
            d.qpos[:] = q
            mujoco.mj_forward(model, d)
            cam.lookat[:] = [*(0.85 * centers[i].mean(0) + 0.15 * centers.mean((0, 1))), 0.65]
            renderer.update_scene(d, camera=cam, scene_option=options)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 895, 35), fill="#24323b")
            draw.text((15, 8), f"G1   /   HOI {iteration}   /   4 kg box", fill="white", font_size=18)
            draw.text((775, 8), f"{i / 50:.2f} s", fill="white", font_size=18)
            writer.append_data(np.asarray(img))


@torch.inference_mode()
def main(args):
    if args.replay_env is not None and not 0 <= args.replay_env < args.num_envs:
        raise ValueError("--replay-env must identify an evaluated environment")
    if args.feature_heatmap and (args.replay_env is None or args.visual_input != "normal"):
        raise ValueError("--feature-heatmap requires --replay-env with live RGB")
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    settings = load_settings(str(_bootstrap.ROOT / "configs/tasks/teleop.json"))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    version = int(state["actor_state_dict"].get("hoi_reference_version", 0))
    if version not in (0, 1, 2, 3, 4, 5) or (args.grasp_feedback and version == 3):
        raise ValueError("Checkpoint incompatible with the requested control mode")
    native_vibe = "native_hoi_version" in state["actor_state_dict"]
    if native_vibe and (args.visual_input != "normal" or args.feature_heatmap):
        raise ValueError("The paired ViBe evaluation uses live RGB without DA2 feature capture")
    original = version != 0
    visual_pickup = args.visual_pickup or version == 5
    feedback = original and (args.grasp_feedback or version in (2, 4, 5))
    smooth = version == 4 or args.audit_smooth_grasp or visual_pickup
    if smooth and version not in (2, 4, 5):
        raise ValueError("The contact and smoothness audit requires a compensation checkpoint")
    cfg = config(
        args.bundle,
        settings,
        args.num_envs,
        args.seed,
        evaluate=True,
        reference_tracking=original,
        grasp_feedback=feedback,
        full_body=version == 3,
        smooth_grasp=smooth,
        visual_pickup=visual_pickup,
    )
    if native_vibe:
        from hoi_native_vibe import NativeCarryActor, native_agent, native_observations

        native_observations(cfg)
    cfg.commands["motion"].pickup_input = args.visual_input
    cfg.commands["motion"].box_offset_m = tuple(args.box_offset_m)
    if args.perturb_start:
        cfg.commands["motion"].pose_range = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}
        cfg.commands["motion"].joint_position_range = (-0.012, 0.012)
    records = []

    def sample(env):
        c = env.command_manager.get_term("motion")
        b = c.box.data
        bottom = b.root_link_pos_w[:, 2] + (matrix_from_quat(b.root_link_quat_w)[:, 2] @ c.vertices.T).amin(
            -1
        )
        hand = env.scene["palms_box"].data.force.norm(dim=-1)
        ground = env.scene["box_floor"].data.force.norm(dim=-1).amax(dim=-1)
        quality = []
        if smooth:
            a = env.action_manager
            curvature = (
                ((a.action - 2 * a.prev_action + a.prev_prev_action) * a.get_term("joint_pos").scale)
                .abs()
                .amax(-1)
            )
            error = (c.robot_joint_pos - c.joint_pos).abs().amax(-1)
            quality = [
                c.nonhand_peak_n[:, None],
                error[:, None],
                (curvature * (env.episode_length_buf > 2))[:, None],
            ]
        return torch.cat(
            [
                env.sim.data.qpos[:],
                b.root_link_pos_w,
                bottom[:, None],
                hand,
                ground[:, None],
                c.lift[:, None],
                c.carried[:, None],
                c.success[:, None],
                c.stable[:, None],
                *quality,
                env.termination_manager.dones[:, None],
            ],
            dim=1,
        ).clone()

    def observe_reward(env):
        rew = interaction(env)
        records.append(sample(env))
        return rew

    cfg.rewards["interaction"].func = observe_reward
    raw = ManagerBasedRlEnv(cfg, "cuda:0")
    capture = None
    try:
        if args.feature_heatmap:
            from hoi_feature_capture import FeatureCapture

            capture = FeatureCapture(raw.da2_visual, args.replay_env)
        env = RslRlVecEnvWrapper(raw, clip_actions=None)
        c = raw.command_manager.get_term("motion")
        obs, _ = env.reset()
        assert c.full.all() and not c.time_steps.any() and not raw.scene.env_origins.any()
        expected_xy = c.box_pose[:1, :2] + c.box_axes.new_tensor(args.box_offset_m) @ c.box_axes
        torch.testing.assert_close(raw.sim.data.qpos[:, -7:-5], expected_xy.expand(raw.num_envs, -1))
        localization = None
        if visual_pickup:
            estimate = c.pickup_offset[:, :2] @ c.box_axes.T
            target = estimate.new_tensor(args.box_offset_m)
            err = (estimate - target).norm(dim=1)
            localization = dict(
                mean_estimated_forward_left_m=estimate.mean(0).tolist(),
                mean_error_cm=100 * err.mean().item(),
                max_error_cm=100 * err.max().item(),
            )
        initial_robot_qpos = raw.sim.data.qpos[:, :-7].cpu().numpy().copy()
        assert abs(raw.step_dt - 0.02) < 1e-8
        if native_vibe:
            settings_actor = native_agent(settings["dagger"]["teacher_checkpoint"])["actor"]
            settings_actor.pop("class_name")
            actor_class = NativeCarryActor
        else:
            settings_actor = agent_config(settings["bank_root"] + "/rank_0", settings, 20)["actor"].copy()
            for key in ["teleop_checkpoint", "class_name"]:
                settings_actor.pop(key)
            actor_class = ReferenceCarryActor if original else CarryActor
        actor = actor_class(obs, {"actor": ["policy"]}, "actor", 29, **settings_actor).to(raw.device).eval()
        actor.load_state_dict(state["actor_state_dict"], strict=True)
        iteration = int(state["iter"])
        records.clear()
        records.append(sample(raw))
        done_once = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        if args.visual_input != "normal":
            from vibe_grail_joint.da2_locomotion.evaluate import VisualIntervention

            mask = VisualIntervention(args.visual_input, 0, raw)
        for tick in range(c.motion.time_step_total + 2):
            actor_obs = mask(obs) if args.visual_input != "normal" else obs
            action = actor(actor_obs, stochastic_output=False)
            if capture is not None and not done_once[args.replay_env]:
                capture.sample(raw, actor, actor_obs, tick)
            obs, _, done, _ = env.step(action)
            done_once |= done.bool()
            if done_once.all():
                break
        if not done_once.all():
            raise RuntimeError("Missing terminal pose")
        if capture is not None:
            capture.save(args.output / "visual_features.npz")
        data = torch.stack(records).cpu().numpy()
        nq = raw.sim._mj_model.nq
        tracking = UpperBodyTracking(args.bundle / "motion.npz")
        rows = []
        for index in range(raw.num_envs):
            trial = data[:, index]
            end = int(np.flatnonzero(trial[1:, -1])[0]) + 1
            trial = trial[: end + 1]
            assert np.isfinite(trial).all()
            xyz = trial[:, nq : nq + 3]
            held = (
                (trial[1:, nq + 3] > 0.15)
                & (trial[1:, nq + 4 : nq + 6] > 0.5).all(1)
                & (trial[1:, nq + 6] < 0.5)
            )
            spans = np.flatnonzero(np.diff(np.r_[False, held, False])).reshape(-1, 2)
            max_carry = max(
                (np.linalg.norm(xyz[1 + a : 1 + b, :2] - xyz[1 + a, :2], axis=1).max() for a, b in spans),
                default=0.0,
            )
            lifted, carried, placed = map(bool, trial[-1, nq + 7 : nq + 10])
            assert carried == (max_carry >= 0.75)
            rows.append(
                dict(
                    env=index,
                    frames=end + 1,
                    tracking=tracking.measure(trial[:, :nq]),
                    lifted=lifted,
                    carried=carried,
                    placed=placed,
                    max_continuous_carry_m=float(max_carry),
                    max_box_bottom_m=float(trial[:, nq + 3].max()),
                    box_net_displacement_m=float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
                    final_target_error_m=float(
                        np.linalg.norm(xyz[-1, :2] - c.box_pose[-1, :2].cpu().numpy())
                    ),
                    final_stable_seconds=round(float(trial[-1, nq + 10]) * raw.step_dt, 4),
                )
            )
            if smooth:
                peak = float(trial[1:, nq + 11].max())
                rows[-1].update(
                    peak_nonhand_force_n=peak,
                    nonhand_contact=peak > 5,
                    clean_placed=placed and peak <= 5,
                    max_joint_error_deg=float(np.rad2deg(trial[1:, nq + 12].max())),
                    max_pd_second_difference_deg=float(np.rad2deg(trial[1:, nq + 13].max())),
                )
        chosen = max(
            rows, key=lambda r: (r.get("clean_placed", False), r["placed"], r["carried"], r["lifted"])
        )
        if args.replay_env is not None:
            chosen = rows[args.replay_env]
        trace = data[: chosen["frames"], chosen["env"]]
        replay = dict(
            qpos=trace[:, :nq],
            box_xyz=trace[:, nq : nq + 3],
            box_bottom=trace[:, nq + 3],
            hand_force=trace[:, nq + 4 : nq + 6],
            floor_force=trace[:, nq + 6],
            stable_steps=trace[:, nq + 10],
            fps=50,
        )
        np.savez_compressed(args.output / "closed_loop.npz", **replay)
        metrics = dict(
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            iteration=iteration,
            seed=args.seed,
            perturb_start=args.perturb_start,
            deterministic_policy=True,
            visual_pickup=visual_pickup,
            initial_localization=localization,
            policy_family="ViBe UoLM" if native_vibe else "DA2",
            grasp_feedback=feedback,
            full_body=version == 3,
            box_offset_forward_left_m=args.box_offset_m,
            initial_robot_qpos_sha256=hashlib.sha256(initial_robot_qpos.tobytes()).hexdigest(),
            vision=args.visual_input,
            episodes=raw.num_envs,
            counts={
                k: sum(r[k] for r in rows)
                for k in (
                    ["lifted", "carried", "placed"] + (["nonhand_contact", "clean_placed"] if smooth else [])
                )
            },
            selected=chosen,
            trials=rows,
        )
        (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("Evaluation:", metrics["counts"], chosen, flush=True)
        if not args.metrics_only:
            render(raw.sim._mj_model, replay, args.output / "hoi_carry.mp4", iteration)
    finally:
        if capture is not None:
            capture.close()
        raw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("bundle", "checkpoint", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--replay-env", type=int, help="Record this paired environment regardless of outcome")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument(
        "--visual-pickup",
        action="store_true",
        help="Evaluate the new controller on an earlier compensation checkpoint",
    )
    parser.add_argument(
        "--feature-heatmap",
        action="store_true",
        help="Save the selected actor's RGB, DA2 tokens and attention weights at each visual update",
    )
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--perturb-start", action="store_true")
    parser.add_argument(
        "--box-offset-m", type=float, nargs=2, metavar=("FORWARD", "LEFT"), default=[0.0, 0.0]
    )
    parser.add_argument(
        "--visual-input",
        choices=("normal", "black_rgb", "freeze"),
        default="normal",
        help="Use live head RGB, DA2 features of black frames, or hold the first observation",
    )
    parser.add_argument(
        "--audit-smooth-grasp",
        action="store_true",
        help="Measure contact and motion spikes on the original compensation policy as well",
    )
    parser.add_argument(
        "--grasp-feedback",
        action="store_true",
        help="Explore physical force feedforward on a reference-conditioned checkpoint",
    )
    main(parser.parse_args())
