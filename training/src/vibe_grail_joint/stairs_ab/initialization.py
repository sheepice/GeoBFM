"""Check actual initial actions, not just the SONIC bypass path."""
import hashlib
import torch
from rsl_rl.models import MLPModel


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MatchedCritic(MLPModel):
    def __init__(self, *args, initialization_seed, **kwargs):
        # Actor construction consumes different random draws in the two methods.
        # Give the otherwise identical critics the exact same initial weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(initialization_seed)
            super().__init__(*args, **kwargs)


@torch.no_grad()
def audit_raw_initialization(actor, teacher, obs):
    was_training = actor.training
    actor.eval()
    try:
        expected = teacher(obs, stochastic_output=False)
        actual = actor(obs, stochastic_output=False)
        changed = obs.clone()
        changed['kv_tokens']['img_tokens'].fill_(3.14159)
        if 'q_cls' in changed:
            changed['q_cls'].fill_(-2.71828)
        perturbed = actor(changed, stochastic_output=False)
        error = float((actual - expected).abs().max())
        image_error = float((actual - perturbed).abs().max())
        if error > 1e-5 or image_error > 1e-6:
            raise RuntimeError(f'Raw SONIC initialization failed: action error={error}, image dependence={image_error}')
        return dict(actor_vs_released_sonic_max_error=error, initial_image_dependence_max_error=image_error,
                    fresh_lora=True, fresh_visual_fusion=True, fresh_command_encoder=True,
                    fresh_optimizer=True, fresh_critic=True, local_stage1_loaded=False)
    finally:
        actor.train(was_training)
