"""Single-robot deployment adapter for the exported encoder and action policy."""
import json
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from .history import VisualHistory
from .encoder import file_sha256


class ExportedPolicy:
    def __init__(self, bundle, device='cuda:0'):
        root = Path(bundle)
        self.manifest = json.loads((root/'manifest.json').read_text())
        for name, key in (('policy.onnx','policy_sha256'),('encoder.ts','encoder_sha256')):
            if file_sha256(root/name) != self.manifest[key]:
                raise ValueError(f'Export checksum mismatch: {name}')
        if 'camera_sha256' in self.manifest:
            if file_sha256(root/'camera.json') != self.manifest['camera_sha256']:
                raise ValueError('Export camera checksum mismatch')
            if json.loads((root/'camera.json').read_text()) != self.manifest['camera']:
                raise ValueError('Export camera contract mismatch')
        self.device = device
        self.encoder = torch.jit.load(str(root/'encoder.ts'),map_location=device).eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.policy = ort.InferenceSession(str(root/'policy.onnx'),options,providers=['CPUExecutionProvider'])
        visual = self.manifest['settings']['visual']
        self.history = VisualHistory(1,visual['frames'],int(np.prod(visual['patch_grid'])),96,device)
        self.interval = visual['update_interval_steps']
        self.step = 0

    def reset(self):
        self.history.reset()
        self.step = 0

    @torch.no_grad()
    def __call__(self, rgb, observations):
        if np.asarray(rgb).shape != (112, 112, 3) or np.asarray(rgb).dtype != np.uint8:
            raise ValueError('DA2 runtime requires calibrated 112x112 RGB uint8; arbitrary resizing is not allowed')
        if self.history.last_step[0] < 0 or self.step % self.interval == 0:
            frame=torch.as_tensor(rgb,device=self.device,dtype=torch.uint8).reshape(1,112,112,3)
            self.history.push(torch.tensor([0],device=self.device),self.encoder(frame),self.step)
        feeds={name:np.asarray(observations[name],dtype=np.float32).reshape(shape)
               for name,shape in self.manifest['input_shapes'].items() if name!='img_tokens'}
        feeds['img_tokens']=self.history.snapshot().cpu().numpy()
        action=self.policy.run(None,feeds)[0]
        self.step+=1
        return action
