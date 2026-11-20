"""Verify actual captured RGB, HDR, materials and frozen DA2 before training."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    import mujoco_warp as mjwarp
    from PIL import Image,ImageDraw
    from da2_vibe.paths import load_settings
    from .scenes import evaluation_env
    from .groot_domain import randomize_episode,randomize_hdr,augment_rgb
    from .geoloco_warp_render import render as previous_render
    torch.set_num_threads(2);torch.manual_seed(91)
    settings=load_settings(a.config);o=settings['groot_visual']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    if 'stones' in settings:
        from mjlab.rl import RslRlVecEnvWrapper
        from ..da2_stones.task import env_config,StoneEnv
        cfg=env_config(settings['schema_bank'],settings,20,0)
        from dataclasses import replace
        cfg.scene.sensors=tuple(replace(sensor,data_types=('rgb','depth','segmentation'))
            if sensor.name=='head_cam' else sensor for sensor in cfg.scene.sensors)
        # Audit only: bring the pillars into view, keeping the training geometry.
        cfg.commands['motion'].approach_spawn_x=(.1,.1)
        env=RslRlVecEnvWrapper(StoneEnv(cfg,device='cuda:0',settings=settings))
        env.unwrapped.command_manager.get_term('motion').frontier=settings['stones']['levels']-1
    else:
        env,_,_=evaluation_env(settings,seed=91,duration=4,heights=(.2,.25),speeds=(.4,),
                                approaches=(0,),limit_cases=8,diagnostic_geometry=True)
    try:
        env.reset();raw=env.unwrapped
        rc0=raw.scene.sensor_context.render_context
        addresses={k:v.data_ptr() for k,v in rc0.groot_state.items() if torch.is_tensor(v)}
        raw.sim.expand_model_fields(('light_dir','light_castshadow','light_active','cam_pos','cam_quat'))
        ctx=raw.scene.sensor_context;rc=ctx.render_context;s=rc.groot_state
        assert addresses=={k:v.data_ptr() for k,v in s.items() if torch.is_tensor(v)}
        cam=raw.scene['head_cam'];qpos=raw.sim.data.qpos.clone();qvel=raw.sim.data.qvel.clone()
        def frame(renderer=None):
            raw.sim.forward()
            if renderer is None:raw.sim.sense()
            else:
                ctx.prepare();mjwarp.refit_bvh(raw.sim.wp_model,raw.sim.wp_data,rc)
                renderer(raw.sim.wp_model,raw.sim.wp_data,rc);ctx.unpack_rgb();ctx.finalize()
            return ctx.get_rgb(cam.camera_idx).clone(),ctx.get_depth(cam.camera_idx).clone()
        # Training-style stone reset already randomizes camera extrinsics.
        # Establish the SAME nominal camera for the initial depth baseline and
        # all appearance-only comparisons, as the stairs evaluation builder does.
        for name in ('cam_pos','cam_quat'):
            default=raw.sim.get_default_field(name)
            getattr(raw.sim.model,name)[:,cam.camera_idx]=default[cam.camera_idx]
        s['controls'][0,3]=0
        rc.geoloco_tensors[0].fill_(1.);rc.geoloco_tensors[1].fill_(1.)
        nominal,depth=frame();previous,previous_depth=frame(previous_render)
        assert torch.equal(nominal,previous) and torch.equal(depth,previous_depth)
        # A training-style reset has already sampled HDR at this same step.
        # Explicit diagnostic resampling must bypass the production throttle.
        s['last_step']=None
        randomize_hdr(raw,None,o,settings['render_domain'])
        randomize_episode(raw,torch.arange(raw.num_envs,device=raw.device),o)
        # Isolate appearance by restoring nominal camera before comparisons.
        cid=cam.camera_idx
        for name in ('cam_pos','cam_quat'):
            default=raw.sim.get_default_field(name)
            getattr(raw.sim.model,name)[:,cid]=default[cid]
        rc.geoloco_tensors[1].zero_()  # isolate actual HDR illumination from sun
        s['controls'][0,:3]=s['controls'].new_tensor([1.,0.,1.])
        images={'old_nominal':nominal};differences={}
        for i in range(len(s['bank'])):
            s['sh'].copy_(s['sh_bank'][i]);s['sky'].copy_(s['bank'][i])
            rgb,d=frame();assert torch.equal(depth,d)
            images[f'hdr_{i}']=rgb
        reference=images['hdr_0']
        for name,rgb in images.items():
            differences[name]=float((rgb.float()-reference.float()).abs().mean())
        assert all(differences[f'hdr_{i}']>.05 for i in range(1,len(s['bank']))),differences
        s['sh'].copy_(s['sh_bank'][0]);s['sky'].copy_(s['bank'][0])
        s['controls'][0,:2]=s['controls'].new_tensor([-1.,0.])
        yaw,d=frame();assert torch.equal(depth,d);images['yaw_pi']=yaw
        yaw_change=float((yaw.float()-reference.float()).abs().mean());assert yaw_change>.05
        s['controls'][0,:2]=s['controls'].new_tensor([1.,0.])
        # Same HDR, same state; independently change colors, pattern and reflectance.
        for name in ('colors','surface','pattern'):s[name].copy_(s[name][:1].expand_as(s[name]))
        baseline,d=frame();images['materials_base']=baseline
        s['colors'][:,:2]=s['colors'].new_tensor([.8,.12,.12])
        color,d=frame();images['color_red']=color
        assert torch.equal(depth,d)
        s['surface'][:,:,3]=0
        solid,d=frame();s['surface'][:,:,3]=2;s['pattern'][:,:,3]=.35
        tiles,d=frame();images['pattern_tiles']=tiles
        s['surface'][:,:,:3]=s['surface'].new_tensor([.8,0.,0.])
        matte,d=frame();s['surface'][:,:,:3]=s['surface'].new_tensor([.1,.8,.8])
        reflective,d=frame();images['reflective']=reflective
        material_changes=dict(color=float((color.float()-baseline.float()).abs().mean()),
                              pattern=float((tiles.float()-solid.float()).abs().mean()),
                              reflection=float((reflective.float()-matte.float()).abs().mean()))
        assert min(material_changes.values())>.02,material_changes
        ids=torch.arange(raw.num_envs,device=raw.device)
        augmented=augment_rgb(reference,s,ids);images['rgb_augmented']=augmented
        assert (augmented!=reference).any()
        f0=raw.da2_visual.encoder(reference);f1=raw.da2_visual.encoder(augmented)
        assert torch.isfinite(f0).all() and torch.isfinite(f1).all()
        feature_change=float((f1-f0).abs().mean());assert feature_change>0
        assert torch.equal(qpos,raw.sim.data.qpos) and torch.equal(qvel,raw.sim.data.qvel)
        before=s['controls'].clone();randomize_hdr(raw,torch.empty(0,dtype=torch.long,device=raw.device),o,settings['render_domain'])
        assert torch.equal(before,s['controls'])
        s['last_step']=None
        randomize_hdr(raw,ids,o,settings['render_domain'])
        randomize_episode(raw,ids,o)
        production,d=frame();images['production_reset']=production
        images['production_augmented']=augment_rgb(production,s,ids)
        report=dict(nominal_matches_previous=True,captured_rgb_verified=True,buffer_addresses_preserved=True,
                    hdr_differences=differences,hdr_yaw_difference=yaw_change,material_changes=material_changes,
                    da2_feature_change=feature_change,physics_unchanged=True,depth_unchanged=True,
                    empty_reset_is_noop=True,fovy=cam.cfg.fovy,images=list(images))
        if 'stones' in settings:
            report['material_groups']={name:int((s['groups']==group).sum()) for group,name in
                enumerate(('floor','pillar','robot','mat'))}
            assert all(report['material_groups'].values())
        report['production_rgb_mean']=production.float().mean((0,1,2)).tolist()
        report['production_near_black_fraction']=float((production<3).all(-1).float().mean())
        report['production_near_white_fraction']=float((production>250).all(-1).float().mean())
        mosaic=Image.new('RGB',(len(images)*224,248*2),'white');draw=ImageDraw.Draw(mosaic)
        for j,(name,rgb) in enumerate(images.items()):
            for i in range(2):
                im=Image.fromarray(rgb[i].cpu().numpy());im.save(out/f'{name}_{i}.png')
                mosaic.paste(im.resize((224,224)),(j*224,i*248+24))
                draw.text((j*224+3,i*248+5),name,fill='black')
        mosaic.save(out/'visual_mosaic.png')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:env.close()


if __name__=='__main__':main()
