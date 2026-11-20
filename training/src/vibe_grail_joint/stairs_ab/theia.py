"""Native 192D Theia tokens and CLS, one frame at the matched 10 Hz rate."""
import torch
from mjlab.managers.manager_base import ManagerTermBase
from ..vision import FrozenTheiaFeature
from rsl_rl.modules import CrossAttentionExtractor


class NativeTheiaFusion(CrossAttentionExtractor):
    """Native ViBe math; keep running statistics in FP32 with FP16 storage."""
    def update_normalization(self, token_td, *query_vecs):
        super().update_normalization(token_td.to(torch.float32), *query_vecs)


class FrozenTheiaEncoder:
    def __init__(self, device, dtype):
        self.model = FrozenTheiaFeature._load('theia', 'theia-tiny-patch16-224-cddsv', device, dtype)
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor([.485, .456, .406], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.assert_frozen()

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Theia backbone must remain frozen and in eval mode')

    @torch.no_grad()
    def __call__(self, rgb):
        x = rgb.permute(0, 3, 1, 2).to(self.dtype) / 255.
        x = (x - self.mean) / self.std
        values = self.model.backbone.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        expected = (rgb.shape[1] // 16) * (rgb.shape[2] // 16) + 1
        if values.shape[1:] != (expected, 192) or not torch.isfinite(values).all():
            raise RuntimeError(f'Unexpected native Theia features: {values.shape}')
        return values[:, 1:], values[:, 0]


class TheiaTokens(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.settings = dict(cfg.params['visual'])
        # Backbone construction must not advance the environment's RNG stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.encoder = FrozenTheiaEncoder(env.device, getattr(torch, self.settings['dtype']))
        sensor = next(s for s in env.cfg.scene.sensors if s.name == 'head_cam')
        self.image_shape = (sensor.height, sensor.width, 3)
        patches = (sensor.height // 16) * (sensor.width // 16)
        self.tokens = torch.zeros(env.num_envs, patches, 192, device=env.device, dtype=torch.float16)
        self.cls = torch.zeros(env.num_envs, 192, device=env.device, dtype=torch.float16)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.calls = self.images = 0
        env.da2_visual = self

    def reset(self, env_ids=None):
        if env_ids is None:
            self.last_step.fill_(-1)
        else:
            self.last_step[env_ids] = -1

    @torch.no_grad()
    def ensure(self, env):
        step = int(env.common_step_counter)
        ids = ((self.last_step < 0) | (step - self.last_step >= self.settings['update_interval_steps'])).nonzero().flatten()
        if len(ids):
            rgb = env.scene['head_cam'].data.rgb
            if rgb.shape[1:] != self.image_shape:
                raise RuntimeError('Rendered RGB differs from the declared camera')
            for chunk in ids.split(self.settings['encoder_batch_size']):
                tokens, cls = self.encoder(rgb[chunk])
                self.tokens[chunk], self.cls[chunk] = tokens.to(self.tokens.dtype), cls.to(self.cls.dtype)
                self.last_step[chunk] = step
                self.calls += 1
                self.images += len(chunk)

    def __call__(self, env, **kwargs):
        self.ensure(env)
        return self.tokens.clone()


def cached_cls(env):
    env.da2_visual.ensure(env)
    return env.da2_visual.cls.float().clone()
