"""Episode-safe, timestamped visual history with chunked frozen inference."""
import torch
from mjlab.managers.manager_base import ManagerTermBase
from .encoder import FrozenVisualEncoder


from .history import VisualHistory


class FrozenVisualTokens(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.settings = dict(cfg.params['visual'])
        self.encoder = FrozenVisualEncoder(self.settings, env.device)
        grid = self.settings['patch_grid']
        self.history = VisualHistory(env.num_envs, self.settings['frames'], grid[0] * grid[1], 96, env.device)
        self.calls = self.images = 0
        env.da2_visual = self

    def reset(self, env_ids=None):
        self.history.reset(env_ids)

    @torch.no_grad()
    def __call__(self, env, **kwargs):
        step = int(env.common_step_counter)
        last = self.history.last_step
        due = (last < 0) | (step - last >= self.settings['update_interval_steps'])
        ids = due.nonzero().flatten()
        if len(ids):
            rgb = env.scene['head_cam'].data.rgb
            if rgb.shape[1:] != (112, 112, 3):
                raise RuntimeError('Camera shape does not match the encoder contract')
            for chunk in ids.split(self.settings['encoder_batch_size']):
                frame=rgb[chunk]
                if hasattr(env,'lighting_gain'):
                    from ..da2_stones.appearance import photometric_rgb
                    frame=photometric_rgb(frame,env,chunk)
                rc=env.scene.sensor_context.render_context
                if hasattr(rc,'groot_state'):
                    from .groot_domain import augment_rgb
                    frame=augment_rgb(frame,rc.groot_state,chunk)
                self.history.push(chunk, self.encoder(frame), step)
                self.calls += 1
                self.images += len(chunk)
        return self.history.snapshot()
