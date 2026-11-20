"""CPU observation fixture matching the saved actor input contract."""

import torch
from tensordict import TensorDict


def observations(batch=4, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(90213)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.1).to(device)

    reference = rand(batch, 267)
    reference[:, 249:261] = 0
    # 240 lower values + 9 points + 12 quaternions + 6 root orientation.
    reference[:, 249] = 1
    reference[:, 253] = 1
    reference[:, 257] = 1
    reference[:, 261:] = reference.new_tensor([1, 0, 0, 1, 0, 0])
    return TensorDict(
        dict(
            policy=rand(batch, 930),
            tokenizer=rand(batch, 640),
            augmentation=rand(batch, 6),
            kv_tokens=TensorDict(dict(img_tokens=rand(batch, 49, 192)), batch_size=[batch]),
            q_task_cmd=rand(batch, 6),
            q_proprio=rand(batch, 64),
            q_cls=rand(batch, 192),
            velocity_command=rand(batch, 8),
            teleop_reference=reference,
            reference_mode=torch.zeros(batch, 1, device=device),
        ),
        batch_size=[batch],
        device=device,
    )
