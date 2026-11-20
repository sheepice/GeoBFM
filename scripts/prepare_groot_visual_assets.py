#!/usr/bin/env python3
"""Cache CC0 Poly Haven HDRs and integrate diffuse SH; never run in PPO workers."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

NAMES = ('studio_small_09', 'kloofendal_48d_partly_cloudy', 'moonless_golf',
         'overcast_soil', 'small_hangar_01')


def read_rgbe(path):
    """Decode standard Radiance scanline RLE without changing the shared venv."""
    with path.open('rb') as f:
        while f.readline().strip(): pass
        axis_y, height, axis_x, width = f.readline().decode().split()
        if (axis_y,axis_x) != ('-Y','+X'): raise ValueError('Unsupported HDR orientation')
        height,width=int(height),int(width)
        rgbe=np.empty((height,width,4),dtype=np.uint8)
        for y in range(height):
            header=f.read(4)
            if header != bytes((2,2,width>>8,width&255)):
                raise ValueError('Unsupported HDR encoding')
            for c in range(4):
                x=0
                while x<width:
                    n=f.read(1)[0]
                    if n>128:
                        length=n-128; value=f.read(1)[0]
                        rgbe[y,x:x+length,c]=value
                    elif n>0:
                        length=n;rgbe[y,x:x+length,c]=np.frombuffer(f.read(length),np.uint8)
                    else:raise ValueError('Invalid HDR run')
                    x+=length
                if x!=width:raise ValueError('Invalid HDR scanline')
        return rgbe[...,:3].astype(np.float32)*np.exp2(rgbe[...,3:4].astype(np.float32)-136)


def sh_basis(x, y, z):
    return np.stack((np.ones_like(x)*.282095, .488603*y, .488603*z, .488603*x,
                     1.092548*x*y, 1.092548*y*z, .315392*(3*z*z-1),
                     1.092548*x*z, .546274*(x*x-y*y)), -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    def download(name):
        url = f'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{name}_1k.hdr'
        path = out / f'{name}_1k.hdr'
        if not path.exists():
            errors = []
            for opener in (urllib.request.build_opener(),
                           urllib.request.build_opener(urllib.request.ProxyHandler({}))):
                try:
                    with opener.open(url, timeout=40) as response:
                        data = response.read()
                    if not data.startswith(b'#?'): raise ValueError('Not a Radiance HDR')
                    path.with_suffix('.tmp').write_bytes(data)
                    path.with_suffix('.tmp').replace(path)
                    break
                except Exception as exc: errors.append(repr(exc))
            else: raise RuntimeError(f'{name}: {errors}')
        rgb = read_rgbe(path)
        if not np.isfinite(rgb).all():
            raise ValueError(f'Cannot decode HDR: {path}')
        h,w,_=rgb.shape
        if h%64 or w%128:raise ValueError('HDR must have 2:1 resolution divisible by 128x64')
        rgb=rgb.reshape(64,h//64,128,w//128,3).mean((1,3))
        theta = (np.arange(64)+.5)*np.pi/64
        phi = (np.arange(128)+.5)*2*np.pi/128 - np.pi
        z = np.cos(theta)[:,None]*np.ones((1,128))
        x = np.sin(theta)[:,None]*np.cos(phi)[None]
        y = np.sin(theta)[:,None]*np.sin(phi)[None]
        weight = np.sin(theta)[:,None]*np.ones((1,128))
        luminance = rgb @ np.array([.2126,.7152,.0722])
        # Decouple appearance from exposure. The runtime samples the HDR exposure.
        scale = float((luminance*weight).sum()/weight.sum())
        rgb = np.maximum(rgb/max(scale,1e-6), 0).astype(np.float32)
        basis = sh_basis(x,y,z)
        solid_angle = weight * (np.pi/64)*(2*np.pi/128)
        coeff = np.einsum('hwc,hwk,hw->kc', rgb, basis, solid_angle)
        # Cosine convolution divided by pi: constant unit radiance -> unit irradiance.
        coeff *= np.array([1,2/3,2/3,2/3,1/4,1/4,1/4,1/4,1/4])[:,None]
        entry = dict(name=name, url=url, source=f'https://polyhaven.com/a/{name}',
                     license='CC0-1.0', license_url='https://polyhaven.com/license',
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     original_luminance_scale=scale)
        print('Prepared', name, flush=True)
        return rgb, coeff.astype(np.float32), entry

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download,NAMES))
    bank = out/'hdr_bank.npz'
    np.savez(bank, radiance=np.stack([r[0] for r in results]),
             irradiance_sh=np.stack([r[1] for r in results]))
    manifest = dict(version=1, sources=[r[2] for r in results],
                    bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
                    radiance_shape=[len(results),64,128,3],
                    diffuse_method='real_SH_l2_cosine_convolution_normalized_by_pi')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(bank=str(bank.resolve()),sha256=manifest['bank_sha256']),indent=2))


if __name__ == '__main__': main()
