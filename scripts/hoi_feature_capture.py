"""Read the RGB and features passed to the actor on each new visual observation."""

import numpy as np
import torch


class FeatureCapture:
    def __init__(self, visual, env_id):
        self.env_id = env_id
        self.rows = []
        self.last_rgb = None
        self.rgb_step = None
        self.history = visual.history
        self.push = self.history.push
        self.handle = visual.encoder.register_forward_hook(self._observe_encoder)
        self.history.push = self._observe_push

    def _observe_encoder(self, module, inputs, output):
        self.encoded_rgb = inputs[0].detach()

    def _observe_push(self, ids, features, step):
        matches = (ids == self.env_id).nonzero().flatten()
        if len(matches):
            self.last_rgb = self.encoded_rgb[matches.item()].cpu().numpy().copy()
            self.rgb_step = int(step)
        self.push(ids, features, step)

    def sample(self, raw, actor, obs, tick):
        if self.rgb_step != raw.common_step_counter:
            return
        assert self.rgb_step == int(self.history.last_step[self.env_id])
        extractor = actor.extractors["kv_tokens"]
        weights = extractor.last_attn[self.env_id]
        assert extractor.query_groups == ["q_proprio", "velocity_command"]
        assert tuple(weights.shape) == (8, 2, 128)
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights[..., 0]))
        features = obs["kv_tokens"]["img_tokens"][self.env_id, -64:]
        torch.testing.assert_close(features, self.history.values[self.env_id, -1], rtol=0, atol=0)
        self.rows.append((tick, self.last_rgb, features.cpu().numpy().copy(), weights.cpu().numpy().copy()))

    def save(self, path):
        ticks, rgb, tokens, weights = map(np.stack, zip(*self.rows, strict=True))
        assert ticks[0] == 0 and np.all(np.diff(ticks) == 5)
        assert np.isfinite(tokens).all() and np.isfinite(weights).all()
        np.savez_compressed(
            path,
            tick=ticks,
            rgb=rgb,
            tokens=tokens,
            attention=weights,
            queries=["q_proprio", "velocity_command"],
            env_id=self.env_id,
            control_hz=50,
        )

    def close(self):
        self.handle.remove()
        self.history.push = self.push
