"""Pillar PPO with stairs-style flat DAgger and optional legacy flat PPO."""
import torch
from ..velocity.posture_policy import UpperBodyPreservationPPO
from ..da2_locomotion.algorithm import GeometryPPO


class StonePPO(UpperBodyPreservationPPO):
    # Reuse the tensor-wise broadcast and the auxiliary checkpoint contract.
    broadcast_parameters = GeometryPPO.broadcast_parameters
    attach_geometry = GeometryPPO.attach_geometry

    def __init__(self, *args, geometry_aux, flat_repair, flat_training=None, standing_dagger=None, flat_dagger=None, **kwargs):
        if standing_dagger is not None and flat_dagger is not None:
            raise ValueError('Choose one flat DAgger objective')
        super().__init__(*args, **kwargs)
        self.geometry_aux = dict(geometry_aux)
        self.flat_repair = dict(flat_repair)
        self.flat_training = dict(flat_training) if flat_training else None
        # Legacy attributes preserve version-1 checkpoints and integrations.
        self.standing_dagger = dict(flat_dagger or standing_dagger) if (flat_dagger or standing_dagger) else None
        self.standing_dagger_buffer = None
        self.standing_teacher_sha256 = None
        self.collector = None
        self.geometry_updates = 0
        self.gradient_audit = None

    def uses_rollout_flat_teacher(self):
        return bool(self.flat_training) and not (
            self.standing_dagger and 'walk' in self.standing_dagger.get('roles', ()))

    def _extra_backward(self):
        actor = self._raw_actor
        original = actor._latest_update_obs
        super()._extra_backward()
        try:
            weights = original['flat_supervision'].flatten()
            ids = (weights > 0).nonzero().flatten()[:self.flat_repair['samples']]
            if not len(ids):
                ids = torch.zeros(1, dtype=torch.long, device=self.device)
            sample = original[ids]
            delta = ((actor(sample, stochastic_output=False) - actor.sonic_action(sample)) * self._upper_action_scales)[:, :12]
            prior = .5 * (weights[ids] * delta.square().mean(-1)).mean() / self.flat_repair['std_rad'] ** 2
            (self.flat_repair['coefficient'] * prior).backward()
            self._prior_sum += float(prior.detach())
            if self.uses_rollout_flat_teacher():
                available = original['flat_teacher_supervision'].flatten().nonzero().flatten()
                if not len(available):
                    raise RuntimeError('Mixed rollout minibatch has no flat teacher samples')
                order = torch.randperm(len(available), device=self.device)[:self.flat_training['teacher_samples']]
                flat = original[available[order]]
                teacher = actor.sonic_action(flat)
                student = actor(flat, stochastic_output=False)
                mse = (student - teacher).square().mean()
                (self.flat_training['teacher_coefficient'] * mse).backward()
                self._flat_teacher_sum += float(mse.detach())
            if self.standing_dagger:
                obs, label = self.standing_dagger_buffer.sample(self.standing_dagger['batch_size'], self.device)
                student = actor(obs, stochastic_output=False)
                mse = (student - label).square().mean()
                (self.standing_dagger['coefficient'] * mse).backward()
                self._standing_dagger_sum += float(mse.detach())
            self._batches += 1
            cfg = self.geometry_aux
            if cfg.get('coefficient', .1) > 0 and (self._batches - 1) % cfg['loss_interval_minibatches'] == 0:
                loss, mae = actor.auxiliary_loss(self.collector.sample(cfg['batch_size'], self.device))
                if self.gradient_audit is None:
                    gradients = torch.autograd.grad(loss, list(actor.extractors.parameters()), retain_graph=True, allow_unused=True)
                    norm = sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** .5
                    if not norm > 0:
                        raise RuntimeError('Geometry supervision does not reach DA2 fusion')
                    self.gradient_audit = dict(action_path_fusion_gradient_norm=norm)
                warmup = min(1., (self.geometry_updates + 1) / cfg['warmup_updates'])
                (cfg['coefficient'] * warmup * loss).backward()
                self._geometry_sum += float(loss.detach())
                self._geometry_mae += float(mae)
                self._geometry_count += 1
        finally:
            actor._latest_update_obs = original

    def update(self):
        if self.geometry_aux.get('coefficient', .1) > 0:
            self.collector.prepare()
        self.collector.unwrapped.da2_visual.encoder.assert_frozen()
        self._prior_sum = self._geometry_sum = self._geometry_mae = self._flat_teacher_sum = self._standing_dagger_sum = 0.
        self._batches = self._geometry_count = 0
        losses, info = super().update()
        losses.pop('flat_sonic_preservation', None)
        self.geometry_updates += 1
        losses.update(safe_approach_leg_prior=self._prior_sum / max(1, self._batches))
        if self.geometry_aux.get('coefficient', .1) > 0:
            losses.update(terrain_mse=self._geometry_sum / max(1, self._geometry_count),
                          terrain_mae_m=self._geometry_mae / max(1, self._geometry_count))
        if self.uses_rollout_flat_teacher():
            losses['flat_teacher_action_mse'] = self._flat_teacher_sum / max(1, self._batches)
        if self.standing_dagger:
            mse = self._standing_dagger_sum / max(1, self._batches)
            prefix = 'flat_dagger' if self.standing_dagger['version'] == 2 else 'standing_dagger'
            losses.update({prefix + '_action_mse': mse,
                           prefix + '_weighted': self.standing_dagger['coefficient'] * mse})
            self.standing_dagger_buffer.commit()
            info[prefix + '_history_samples'] = len(self.standing_dagger_buffer)
            for key, count in enumerate(self.standing_dagger_buffer.seen.tolist()):
                info[f'{prefix}_stratum_{key}_collected'] = count
        return losses, info

    def save(self):
        state = super().save()
        state['stone_aux'] = dict(geometry_aux=self.geometry_aux, flat_repair=self.flat_repair,
                                 updates=self.geometry_updates, gradient_audit=self.gradient_audit)
        state['stone_aux']['flat_training'] = self.flat_training
        if self.standing_dagger:
            key = 'flat_dagger' if self.standing_dagger['version'] == 2 else 'standing_dagger'
            state['stone_aux'][key] = self.standing_dagger
            state['stone_aux']['standing_teacher_sha256'] = self.standing_teacher_sha256
        return state

    def load(self, state, load_cfg=None, strict=True):
        aux = state['stone_aux']
        if aux.get('flat_dagger', aux.get('standing_dagger')) != self.standing_dagger:
            raise ValueError('Changed flat DAgger recipe; use actor initialization instead of resume')
        if self.standing_dagger and aux.get('standing_teacher_sha256') != self.standing_teacher_sha256:
            raise ValueError('Changed standing SONIC teacher on resume')
        if aux.get('flat_training') != self.flat_training:
            raise ValueError('Changed flat training recipe on resume')
        if aux['geometry_aux'] != self.geometry_aux or aux['flat_repair'] != self.flat_repair:
            raise ValueError('Changed auxiliary losses on resume')
        result = super().load(state, load_cfg, strict)
        self.geometry_updates = aux['updates']
        self.gradient_audit = aux['gradient_audit']
        return result
