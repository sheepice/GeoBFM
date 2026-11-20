import math
import torch
from ..da2_locomotion.algorithm import GeometryPPO


class FlatRepairPPO(GeometryPPO):
    """Leg SONIC supervision on actual safe-flat PPO states, including RGB stairs."""
    def __init__(self, *args, flat_repair, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat_repair = dict(flat_repair)
        self._flat_sum = self._flat_rad2 = self._flat_weight = self._flat_count = 0

    def _extra_backward(self):
        actor = self._raw_actor
        original_obs = actor._latest_update_obs
        super()._extra_backward()
        try:
            weights = original_obs['flat_supervision'].flatten()
            eligible = (weights > 0).nonzero().flatten()
            if len(eligible):
                order = torch.randperm(len(eligible), device=self.device)[:self.flat_repair['samples']]
                ids = eligible[order]
                weight = weights[ids]
            else:
                ids = torch.zeros(1, device=self.device, dtype=torch.long)
                weight = weights.new_zeros(1)
            sample = original_obs[ids]
            target = actor.sonic_action(sample)
            prediction = actor(sample, stochastic_output=False)
            delta = ((prediction - target) * self._upper_action_scales)[:, :12]
            per_state = delta.square().mean(-1)
            loss = .5 * (weight * per_state).mean() / self.flat_repair['std_rad'] ** 2
            weighted = self.flat_repair['coefficient'] * loss
            weighted.backward()
            self._flat_sum += float(weighted.detach())
            self._flat_rad2 += float((weight * per_state.detach()).sum())
            self._flat_weight += float(weight.sum())
            self._flat_count += 1
        finally:
            actor._latest_update_obs = original_obs

    def update(self):
        self._flat_sum = self._flat_rad2 = self._flat_weight = self._flat_count = 0
        losses, info = super().update()
        losses['safe_flat_leg_prior'] = self._flat_sum / max(1, self._flat_count)
        info['FlatRepair/leg_delta_rmse_rad'] = math.sqrt(self._flat_rad2 / max(1, self._flat_weight))
        info['FlatRepair/weighted_samples'] = self._flat_weight
        return losses, info

    def save(self):
        state = super().save()
        state['flat_repair'] = self.flat_repair
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get('flat_repair') != self.flat_repair:
            raise ValueError('Resume requires identical flat-repair settings')
        return super().load(loaded_dict, load_cfg, strict)
