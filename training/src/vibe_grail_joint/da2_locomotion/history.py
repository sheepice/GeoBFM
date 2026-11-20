"""Simulator-independent visual history shared by training and deployment."""
import torch


class VisualHistory:
    def __init__(self, count, frames, patches, channels, device):
        self.values = torch.zeros(count, frames, patches, channels, device=device, dtype=torch.float16)
        self.last_step = torch.full((count,), -1, device=device, dtype=torch.long)

    def reset(self, ids=None):
        if ids is None:
            self.last_step.fill_(-1)
        else:
            self.last_step[ids] = -1

    def push(self, ids, features, step):
        features = features.to(self.values.dtype)
        previous = self.values[ids].clone()
        fresh = self.last_step[ids] < 0
        self.values[ids, :-1] = previous[:, 1:]
        self.values[ids, -1] = features
        if fresh.any():
            self.values[ids[fresh]] = features[fresh, None].expand(-1, self.values.shape[1], -1, -1)
        self.last_step[ids] = step

    def snapshot(self):
        # Never mutate observations retained by PPO when the next camera frame arrives.
        return self.values.flatten(1, 2).clone()
