"""Three-point DAgger, inherited DA2 geometry and safe-flat priors share one update."""

import torch
from vibe_grail_joint.da2_flat_repair.algorithm import FlatRepairPPO
from vibe_grail_joint.velocity.posture_policy import UpperBodyPreservationPPO

from .contracts import physical_action_loss


class DA2TeleopPPO(FlatRepairPPO):
    def __init__(self, *args, vr_teleop, **kwargs):
        self.vr_teleop = dict(vr_teleop)
        super().__init__(*args, **kwargs)

    def _extra_backward(self):
        actor = self._raw_actor
        original_obs = actor._latest_update_obs
        try:
            # Replace the action-unit DAgger term with vr-vibe's grouped PD-target
            # loss. Do not call DaggerPPO._extra_backward and count it twice.
            UpperBodyPreservationPPO._extra_backward(self)
            obs, label = self.dagger_buffer.sample(self.dagger_batch_size, self.device)
            student = actor(obs, stochastic_output=False)
            cfg = self.vr_teleop
            mse, groups = physical_action_loss(
                student,
                label,
                self._upper_action_scales,
                cfg["dagger_leg_weight"],
                cfg["dagger_waist_weight"],
                cfg["dagger_arm_weight"],
            )
            loss = mse / cfg["dagger_std_rad"] ** 2
            (self.dagger_coef * loss).backward()
            self._dagger_sum += float(mse.detach())
            self._dagger_count += 1
            self._vr_loss_sum += float(loss.detach())
            self._vr_group_sum += groups.detach()
            self._geometry_backward()
            self._flat_backward(original_obs)
        finally:
            actor._latest_update_obs = original_obs

    def _geometry_backward(self):
        self._aux_minibatches += 1
        cfg = self.geometry_aux
        if (
            cfg["coefficient"] == 0
            or (self._aux_minibatches - 1) % cfg["loss_interval_minibatches"]
        ):
            return
        batch = self.collector.sample(cfg["batch_size"], self.device)
        loss, mae = self._raw_actor.auxiliary_loss(batch)
        if self.gradient_audit is None:
            grads = torch.autograd.grad(
                loss,
                list(self._raw_actor.extractors.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            norm = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** 0.5
            if not norm > 0:
                raise RuntimeError("Terrain supervision did not reach DA2 action-path fusion")
            self.gradient_audit = {"action_path_fusion_gradient_norm": norm}
        warmup = min(1.0, (self.geometry_updates + 1) / cfg["warmup_updates"])
        weighted = cfg["coefficient"] * warmup * loss
        weighted.backward()
        self._aux_count += 1
        for name, value in {
            "terrain_mse": loss,
            "terrain_mae_m": mae,
            "terrain_weighted": weighted,
        }.items():
            self._aux_sums[name] = self._aux_sums.get(name, 0.0) + float(value.detach())

    def _flat_backward(self, obs):
        weights = obs["flat_supervision"].flatten()
        eligible = (weights > 0).nonzero().flatten()
        if len(eligible):
            ids = eligible[
                torch.randperm(len(eligible), device=self.device)[: self.flat_repair["samples"]]
            ]
            weight = weights[ids]
        else:
            ids = torch.zeros(1, device=self.device, dtype=torch.long)
            weight = weights.new_zeros(1)
        sample = obs[ids]
        target = self._raw_actor.sonic_action(sample)
        prediction = self._raw_actor(sample, stochastic_output=False)
        delta = ((prediction - target) * self._upper_action_scales)[:, :12]
        per_state = delta.square().mean(-1)
        weighted = (
            self.flat_repair["coefficient"]
            * 0.5
            * (weight * per_state).mean()
            / self.flat_repair["std_rad"] ** 2
        )
        weighted.backward()
        self._flat_sum += float(weighted.detach())
        self._flat_rad2 += float((weight * per_state.detach()).sum())
        self._flat_weight += float(weight.sum())
        self._flat_count += 1

    def update(self):
        self._vr_loss_sum = 0.0
        self._vr_group_sum = torch.zeros(3, device=self.device)
        losses, info = super().update()
        count = max(1, self._dagger_count)
        losses["dagger_target_mse_rad2"] = losses.pop("dagger_action_mse")
        losses["vr_dagger_normalized"] = self._vr_loss_sum / count
        losses["dagger_weighted"] = self.dagger_coef * self._vr_loss_sum / count
        info.pop("dagger_action_rmse", None)
        for name, value in zip(("legs", "waist", "arms"), self._vr_group_sum / count):
            info["VR/flat_dagger_" + name + "_target_rmse_rad"] = float(value.sqrt())
        return losses, info

    def save(self):
        state = super().save()
        state["vr_teleop"] = dict(self.vr_teleop)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("vr_teleop") != self.vr_teleop:
            raise ValueError("Resume requires identical TELEOP input and loss settings")
        return super().load(loaded_dict, load_cfg, strict)
