"""Explicit first-stage transfer across the Theia -> DA2 visual interface."""
from da2_vibe.paths import load_checkpoint
import hashlib
from pathlib import Path
import torch


def audit_raw_sonic(actor, obs, agent, settings, device):
    """Verify the actual freshly constructed policy against an independent SONIC."""
    from ..stairs_ab.initialization import audit_raw_initialization
    from ..velocity.dagger_policy import build_teacher
    devices = [torch.device(device).index] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        teacher, digest = build_teacher(obs, agent['actor'], agent['obs_groups'],
                                        settings['sonic_checkpoint'], device)
        report = audit_raw_initialization(actor, teacher, obs)
    report.update(checkpoint=str(Path(settings['sonic_checkpoint']).resolve()), sha256=digest,
                  initialization='raw_sonic_fresh_visual', inherited_trained_lora=False,
                  visual_backend=settings['visual']['backend'],
                  fresh_command_encoder=hasattr(actor, 'command_encoder'))
    return report


def load_stage1(actor, checkpoint, *, visual_backend='da2_metric_small'):
    state = load_checkpoint(checkpoint, map_location='cpu', weights_only=False)
    source = state['actor_state_dict']
    current = actor.state_dict()
    # The feature space, query count and attention architecture changed. Retain
    # every compatible nonvisual SONIC/LoRA tensor, but initialize DA2 fusion and
    # its height head afresh. This is not a DA2 model11100 warm start.
    transferred = {k: v for k, v in source.items() if not k.startswith('extractors.')}
    for key, value in transferred.items():
        if key not in current or current[key].shape != value.shape:
            raise ValueError(f'Incompatible nonvisual first-stage weight: {key}')
    if not any(k.startswith('decoder.adapters.') for k in transferred):
        raise ValueError('First-stage LoRA weights were not found')
    missing, unexpected = actor.load_state_dict(transferred, strict=False)
    allowed = ('extractors.', 'terrain_head.', 'command_encoder.')
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise ValueError(f'Unexpected stage-one transfer gap: {missing}, {unexpected}')
    for key, value in transferred.items():
        if not torch.equal(actor.state_dict()[key].cpu(), value):
            raise RuntimeError(f'First-stage tensor was not preserved: {key}')
    return dict(checkpoint=str(Path(checkpoint).resolve()),
                sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                source_iteration=state['iter'], copied_keys=list(transferred),
                new_keys=missing, old_visual_keys_omitted=[k for k in source if k.startswith('extractors.')],
                inherited_trained_lora=True, fresh_da2_fusion=visual_backend == 'da2_metric_small',
                fresh_visual_fusion=True, visual_backend=visual_backend, fresh_critic_optimizer=True,
                exact_nonvisual_transfer=True, da2_model11100_loaded=False)
