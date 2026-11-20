"""Behavioral contracts for independent DA2 policy conditioning."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from vibe_grail_joint.da2_locomotion.encoder import group_channels
from vibe_grail_joint.da2_locomotion.model import VisualQueryExtractor, TerrainConditionedActor
from vibe_grail_joint.da2_locomotion.observation import VisualHistory

ROOT = Path(__file__).resolve().parents[1]


def test_channel_compression_never_mixes_spatial_locations():
    values = torch.arange(3.).view(1, 3, 1).expand(2, -1, 384).clone()
    compressed = group_channels(values)
    torch.testing.assert_close(compressed, values[..., :32])


def test_history_advances_only_for_new_images_and_resets_only_selected_worlds():
    history = VisualHistory(2, 2, 4, 3, 'cpu')
    first = torch.ones(2, 4, 3)
    history.push(torch.tensor([0, 1]), first, 0)
    snapshot = history.snapshot()
    history.push(torch.tensor([0, 1]), first * 2, 5)
    torch.testing.assert_close(snapshot.float(), torch.ones_like(snapshot).float())
    assert (history.values[:, 0] == 1).all() and (history.values[:, 1] == 2).all()
    history.reset(torch.tensor([0]))
    history.push(torch.tensor([0]), first[:1] * 9, 6)
    assert (history.values[0] == 9).all()
    assert (history.values[1, 0] == 1).all() and (history.values[1, 1] == 2).all()
    assert history.last_step.tolist() == [6, 5]


@pytest.mark.parametrize('heads,frames', [(1, 2), (8, 2), (8, 1)])
def test_attention_responds_to_image_and_both_queries(heads, frames):
    torch.manual_seed(12)
    model = VisualQueryExtractor({'q_proprio': 93, 'velocity_command': 8}, frames=frames, num_heads=heads)
    tokens = torch.randn(3, frames * 64, 96, requires_grad=True)
    queries = [torch.randn(3, d, requires_grad=True) for d in (93, 8)]
    z, weights = model.attention(tokens, *queries)
    assert z.shape == (3, 128) and weights.shape == (3, heads, 2, frames * 64)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    (z * torch.randn_like(z)).sum().backward()
    assert tokens.grad.abs().sum() > 0
    assert all(q.grad.abs().sum() > 0 for q in queries)
    assert torch.isfinite(z).all()


def test_auxiliary_gradient_updates_the_same_fusion_used_for_actions():
    torch.manual_seed(9)
    fusion = VisualQueryExtractor({'q_proprio': 7, 'velocity_command': 8})
    tokens = TensorDict({'img_tokens': torch.randn(4, 128, 96)}, [4])
    queries = [torch.randn(4, d) for d in (7, 8)]
    head = torch.nn.Linear(128, 99)
    before = fusion(tokens, *queries).detach().clone()
    loss = (head(fusion(tokens, *queries)) - torch.randn(4, 99)).square().mean()
    loss.backward()
    assert fusion.w_k.weight.grad.abs().sum() > 0
    assert fusion.w_v.weight.grad.abs().sum() > 0
    torch.optim.SGD([*fusion.parameters(), *head.parameters()], lr=.05).step()
    assert not torch.allclose(before, fusion(tokens, *queries))


def test_geometry_loss_ignores_missing_ray_hits_and_uses_fixed_metric_scale():
    surrogate = SimpleNamespace(terrain_head=torch.nn.Identity(), encode=lambda obs: obs,
                                geometry_aux={'height_scale_m': 2.})
    batch = {'obs': torch.tensor([[2., 100.]], requires_grad=True),
             'height': torch.tensor([[1., -100.]]), 'valid': torch.tensor([[True, False]])}
    loss, mae = TerrainConditionedActor.auxiliary_loss(surrogate, batch)
    assert loss.item() == 1 and mae.item() == 2
    loss.backward()
    assert batch['obs'].grad[0, 1] == 0
