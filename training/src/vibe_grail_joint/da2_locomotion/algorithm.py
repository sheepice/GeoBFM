"""Shared PPO/DAgger/geometry optimization, with explicit gradient and resume checks."""
import torch
from ..velocity.dagger_policy import DaggerPPO


class GeometryPPO(DaggerPPO):
    def __init__(self, *args, geometry_aux, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_aux = dict(geometry_aux)
        self.collector = None
        self.geometry_updates = 0
        self.gradient_audit = None

    def attach_geometry(self, collector):
        self.collector = collector

    @torch.no_grad()
    def broadcast_parameters(self):
        if not torch.distributed.is_initialized():
            return
        for model in (self._raw_actor, self._raw_critic):
            for value in model.state_dict().values():
                if value.numel():
                    payload = value.contiguous()
                    torch.distributed.broadcast(payload, src=0)
                    if not value.is_contiguous():
                        value.copy_(payload)

    def _extra_backward(self):
        super()._extra_backward()
        self._aux_minibatches += 1
        cfg = self.geometry_aux
        if (self._aux_minibatches - 1) % cfg['loss_interval_minibatches']:
            return
        if cfg['coefficient'] == 0:
            # No target transfer, auxiliary forward or gradient. The collector
            # shares its private RNG between capture and replay sampling.
            self.collector.discard_sample(cfg['batch_size'])
            return
        batch = self.collector.sample(cfg['batch_size'], self.device)
        loss, mae = self._raw_actor.auxiliary_loss(batch)
        if self.gradient_audit is None:
            # Fixed-query ablations keep unused state projections frozen.
            # allow_unused does not accept inputs with requires_grad=False.
            parameters = [p for p in self._raw_actor.extractors.parameters() if p.requires_grad]
            if not parameters:
                raise RuntimeError('Terrain supervision has no trainable action-path visual fusion')
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            norm = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** .5
            if not norm > 0:
                raise RuntimeError('Terrain supervision did not reach the action-path visual fusion')
            self.gradient_audit = dict(action_path_fusion_gradient_norm=norm)
        warmup = min(1., (self.geometry_updates + 1) / cfg['warmup_updates'])
        weighted = cfg['coefficient'] * warmup * loss
        weighted.backward()
        self._aux_count += 1
        for name, value in dict(terrain_mse=loss, terrain_mae_m=mae, terrain_weighted=weighted).items():
            self._aux_sums[name] = self._aux_sums.get(name, 0.) + float(value.detach())

    def update(self):
        self.collector.prepare()
        self.collector.unwrapped.da2_visual.encoder.assert_frozen()
        self._aux_sums = {}
        self._aux_count = self._aux_minibatches = 0
        losses, info = super().update()
        self.geometry_updates += 1
        losses.update({name: value / max(1, self._aux_count) for name, value in self._aux_sums.items()})
        info.update({f'Geometry/{name}': value for name, value in self.collector.last_metrics.items()})
        info['Geometry/backward_calls'] = self._aux_count
        if self.gradient_audit:
            info['Geometry/action_path_gradient_norm'] = self.gradient_audit['action_path_fusion_gradient_norm']
        return losses, info

    def save(self):
        state = super().save()
        state['geometry_aux'] = dict(settings=self.geometry_aux, updates=self.geometry_updates,
                                     gradient_audit=self.gradient_audit)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        saved = loaded_dict['geometry_aux']
        if saved['settings'] != self.geometry_aux:
            raise ValueError('Geometry settings changed on resume')
        result = super().load(loaded_dict, load_cfg, strict)
        self.geometry_updates, self.gradient_audit = saved['updates'], saved['gradient_audit']
        return result
