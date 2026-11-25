"""Checkpointing and distributed initialization shared by both HOI trainers."""

import json
import os

import torch
from mjlab.rl import MjlabOnPolicyRunner


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def native_broadcast(algo):
    for module in (algo._raw_actor, algo._raw_critic):
        for tensor in module.state_dict().values():
            if tensor.numel():
                torch.distributed.broadcast(tensor, src=0)


class Runner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path + ".tmp", infos)
        os.replace(path + ".tmp", path)
