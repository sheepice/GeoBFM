# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# GR00T-inspired HDR irradiance, procedural surfaces and material shading.
# Diffuse irradiance uses real HDR SH; this is not Isaac RTX/MDL equivalence.
# Upstream render.py SHA256: f06e0df52854f2f3d01777b85dbcaf5f233a3a6c481b7c1bd8c41960ec2e70dc
# Geometry, textures, depth, segmentation and shadow visibility are unchanged.
import warp as wp
from mujoco_warp._src.render import cast_ray, compute_lighting, sample_skybox, sample_texture
from mujoco_warp._src.render_util import compute_ray, pack_rgba_to_uint32
from mujoco_warp._src.types import Model, Data, RenderContext, ObjType
from mujoco_warp._src.warp_util import event_scope


@wp.func
def rotate_environment(v: wp.vec3, control: wp.vec4):
  return wp.vec3(control[0]*v[0]+control[1]*v[1],
                 -control[1]*v[0]+control[0]*v[1],v[2])


@wp.func
def hdr_diffuse(n: wp.vec3, sh: wp.array[wp.vec3], control: wp.vec4):
  d=rotate_environment(wp.normalize(n),control)
  x=d[0];y=d[1];z=d[2]
  e=(sh[0]*0.282095 + sh[1]*(0.488603*y) + sh[2]*(0.488603*z)
     + sh[3]*(0.488603*x) + sh[4]*(1.092548*x*y) + sh[5]*(1.092548*y*z)
     + sh[6]*(0.315392*(3.0*z*z-1.0)) + sh[7]*(1.092548*x*z)
     + sh[8]*(0.546274*(x*x-y*y)))
  return wp.max(e,wp.vec3(0.0))*control[2]


@wp.func
def hdr_radiance(direction: wp.vec3, sky: wp.array2d[wp.vec3], control: wp.vec4):
  d=rotate_environment(wp.normalize(direction),control)
  u=(wp.atan2(d[1],d[0])+wp.pi)/(2.0*wp.pi)*128.0-0.5
  v=wp.acos(wp.clamp(d[2],-1.0,1.0))/wp.pi*64.0-0.5
  u0=int(wp.floor(u));v0=int(wp.floor(v))
  a=u-wp.floor(u);b=v-wp.floor(v)
  x0=(u0+128)%128;x1=(x0+1)%128
  y0=wp.clamp(v0,0,63);y1=wp.clamp(v0+1,0,63)
  color=(sky[y0,x0]*(1.0-a)+sky[y0,x1]*a)*(1.0-b)+(sky[y1,x0]*(1.0-a)+sky[y1,x1]*a)*b
  return color*control[2]


@wp.func
def surface_pattern(p: wp.vec3, family: float, params: wp.vec4):
  scale=params[0];phase=params[1]*6.283185;amount=params[3]
  value=float(0.0)
  if family>0.5 and family<1.5:
    value=wp.sin(p[0]*scale+2.0*wp.sin(p[1]*2.0)+phase)
  elif family>1.5 and family<2.5:
    u=p[0]*scale+phase;v=p[1]*scale+phase
    fu=u-wp.floor(u);fv=v-wp.floor(v)
    value=0.2
    if fu<0.04 or fv<0.04:value=-1.0
  elif family>2.5:
    value=wp.sin(p[0]*scale*7.0+phase)*wp.sin(p[1]*scale*9.0+phase)*wp.sin(p[2]*scale*5.0+1.0)
  return 1.0+amount*value


@wp.func
def color_texture(p: wp.vec3, normal: wp.vec3, base: wp.vec3, family: float, params: wp.vec4):
  """Blend fine stripes or grain toward a second color in surface coordinates."""
  n = wp.vec3(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
  u = p[0]
  v = p[1]
  if n[0] > n[1] and n[0] > n[2]:
    u = p[1]
    v = p[2]
  elif n[1] > n[2]:
    v = p[2]
  phase = params[1] * 6.283185
  angle = params[2] * 6.283185
  x = (u * wp.cos(angle) + v * wp.sin(angle)) * params[0]
  y = (-u * wp.sin(angle) + v * wp.cos(angle)) * params[0]
  mask = wp.clamp((wp.sin(x * 6.283185 + phase) - 0.35) * 1.538462, 0.0, 1.0)
  if family > 4.5:
    grain = wp.sin((x * 7.0 + y * 5.0) + phase) * wp.sin(x * 3.0 - y * 11.0 + phase)
    mask = wp.clamp(grain * 2.0, 0.0, 1.0)
  accent = wp.vec3(0.94, 0.98, 1.0)
  if wp.dot(base, wp.vec3(0.299, 0.587, 0.114)) > 0.5:
    accent = wp.vec3(0.12, 0.07, 0.04)
  mix = mask * params[3]
  return base * (1.0 - mix) + accent * mix


@event_scope
def render(m: Model, d: Data, rc: RenderContext):
  """Render the current frame.

  Outputs are stored in buffers within the render context.

  Args:
    m: The model on device.
    d: The data on device.
    rc: The render context on device.
  """
  rc.rgb_data.fill_(rc.background_color)
  rc.depth_data.fill_(0.0)
  rc.seg_data.fill_(wp.vec2i(-1, -1))

  @wp.kernel(module="unique", enable_backward=False)
  def _render_megakernel(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    cam_projection: wp.array[int],
    cam_fovy: wp.array2d[float],
    cam_sensorsize: wp.array[wp.vec2],
    cam_intrinsic: wp.array2d[wp.vec4],
    light_type: wp.array2d[int],
    light_castshadow: wp.array2d[bool],
    light_active: wp.array2d[bool],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_radius: wp.array[float],
    mesh_faceadr: wp.array[int],
    mat_texid: wp.array3d[int],
    mat_texrepeat: wp.array2d[wp.vec2],
    mat_rgba: wp.array2d[wp.vec4],
    ambient_gain: wp.array[wp.vec3],
    direct_gain: wp.array[wp.vec3],
    hdr_sh: wp.array[wp.vec3],
    hdr_sky: wp.array2d[wp.vec3],
    hdr_control: wp.array[wp.vec4],
    material_color: wp.array2d[wp.vec3],
    material_surface: wp.array2d[wp.vec4],
    material_pattern: wp.array2d[wp.vec4],
    material_group: wp.array[int],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    cam_xpos_in: wp.array2d[wp.vec3],
    cam_xmat_in: wp.array2d[wp.mat33],
    light_xpos_in: wp.array2d[wp.vec3],
    light_xdir_in: wp.array2d[wp.vec3],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    nrender: int,
    use_shadows: bool,
    bvh_ngeom: int,
    bvh_nflexgeom: int,
    cam_res: wp.array[wp.vec2i],
    cam_id_map: wp.array[int],
    ray: wp.array[wp.vec3],
    rgb_adr: wp.array[int],
    depth_adr: wp.array[int],
    seg_adr: wp.array[int],
    render_rgb: wp.array[bool],
    render_depth: wp.array[bool],
    render_seg: wp.array[bool],
    bvh_id: wp.uint64,
    group_root: wp.array[int],
    flex_bvh_id: wp.array[wp.uint64],
    flex_group_root: wp.array2d[int],
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    mesh_facetexcoord: wp.array[wp.vec3i],
    mesh_texcoord: wp.array[wp.vec2],
    mesh_texcoord_offsets: wp.array[int],
    hfield_bvh_id: wp.array[wp.uint64],
    flex_rgba: wp.array[wp.vec4],
    flex_geom_flexid: wp.array[int],
    flex_geom_edgeid: wp.array[int],
    textures: wp.array[wp.Texture2D],
    # Out:
    rgb_out: wp.array2d[wp.uint32],
    depth_out: wp.array2d[float],
    seg_out: wp.array2d[wp.vec2i],
  ):
    worldid, rayid = wp.tid()

    # Map global rayid -> (cam_idx, rayid_local) using cumulative sizes
    cam_idx = int(-1)
    rayid_local = int(-1)
    accum = int(0)
    for i in range(nrender):
      num_i = cam_res[i][0] * cam_res[i][1]
      if rayid < accum + num_i:
        cam_idx = i
        rayid_local = rayid - accum
        break
      accum += num_i
    if cam_idx == -1 or rayid_local < 0:
      return

    if not render_rgb[cam_idx] and not render_depth[cam_idx] and not render_seg[cam_idx]:
      return

    # Map active camera index to MuJoCo camera ID
    mujoco_cam_id = cam_id_map[cam_idx]

    if wp.static(rc.use_precomputed_rays):
      ray_dir_local_cam = ray[rayid]
    else:
      img_w = cam_res[cam_idx][0]
      img_h = cam_res[cam_idx][1]
      px = rayid_local % img_w
      py = rayid_local // img_w
      ray_dir_local_cam = compute_ray(
        cam_projection[mujoco_cam_id],
        cam_fovy[worldid % cam_fovy.shape[0], mujoco_cam_id],
        cam_sensorsize[mujoco_cam_id],
        cam_intrinsic[worldid % cam_intrinsic.shape[0], mujoco_cam_id],
        img_w,
        img_h,
        px,
        py,
        wp.static(rc.znear),
      )

    ray_dir_world = cam_xmat_in[worldid, mujoco_cam_id] @ ray_dir_local_cam
    ray_origin_world = cam_xpos_in[worldid, mujoco_cam_id]

    geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
      geom_type,
      geom_dataid,
      geom_size,
      flex_vertadr,
      flex_edge,
      flex_radius,
      geom_xpos_in,
      geom_xmat_in,
      flexvert_xpos_in,
      bvh_id,
      group_root[worldid],
      worldid,
      bvh_ngeom,
      bvh_nflexgeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      flex_geom_flexid,
      flex_geom_edgeid,
      flex_bvh_id,
      flex_group_root,
      ray_origin_world,
      ray_dir_world,
      wp.static(rc.enable_backface_culling),
    )

    if render_seg[cam_idx] and geom_id != -1:
      if geom_id == -2:
        seg_out[worldid, seg_adr[cam_idx] + rayid_local] = wp.vec2i(mesh_id, int(ObjType.FLEX))
      else:
        seg_out[worldid, seg_adr[cam_idx] + rayid_local] = wp.vec2i(geom_id, int(ObjType.GEOM))

    # Early Out
    if geom_id == -1:
      if hdr_control[0][3]>0.5 and render_rgb[cam_idx]:
        sky=hdr_radiance(ray_dir_world,hdr_sky,hdr_control[0])
        sky=wp.cw_div(sky,wp.vec3(1.0)+sky)
        rgb_out[worldid,rgb_adr[cam_idx]+rayid_local]=pack_rgba_to_uint32(sky[0]*255.0,sky[1]*255.0,sky[2]*255.0,255.0)
        return
      if wp.static(rc.render_skybox) and render_rgb[cam_idx]:
        skybox_color = sample_skybox(
          textures[wp.static(rc.skybox_tex_id)],
          wp.static(1.0 / float(rc.skybox_face_width)),
          ray_dir_world,
        )
        rgb_out[worldid, rgb_adr[cam_idx] + rayid_local] = pack_rgba_to_uint32(
          skybox_color[0] * 255.0,
          skybox_color[1] * 255.0,
          skybox_color[2] * 255.0,
          255.0,
        )
      return

    if render_depth[cam_idx]:
      # Planar depth: project Euclidean distance onto the camera's optical axis.
      # In camera-local coordinates, the optical axis is -Z. The Z-component of the
      # normalized ray direction is negative, so -ray_dir_local_cam[2] gives cos(θ)
      # between the ray and the optical axis.
      depth_out[worldid, depth_adr[cam_idx] + rayid_local] = dist * (-ray_dir_local_cam[2])

    if not render_rgb[cam_idx]:
      return

    # Shade the pixel
    hit_point = ray_origin_world + ray_dir_world * dist

    if geom_id == -2:
      # We encode flex_id in mesh_id for flex ray hits during cast_ray
      color = flex_rgba[mesh_id]
    elif geom_matid[worldid % geom_matid.shape[0], geom_id] == -1:
      color = geom_rgba[worldid % geom_rgba.shape[0], geom_id]
    else:
      color = mat_rgba[worldid % mat_rgba.shape[0], geom_matid[worldid % geom_matid.shape[0], geom_id]]

    base_color = wp.vec3(color[0], color[1], color[2])
    hit_color = base_color

    if wp.static(rc.use_textures):
      if geom_id != -2:
        mat_id = geom_matid[worldid % geom_matid.shape[0], geom_id]
        if mat_id >= 0:
          tex_id = mat_texid[worldid % mat_texid.shape[0], mat_id, 1]
          if tex_id >= 0:
            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              geom_id,
              mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
              textures[tex_id],
              geom_xpos_in[worldid, geom_id],
              geom_xmat_in[worldid, geom_id],
              mesh_facetexcoord,
              mesh_texcoord,
              mesh_texcoord_offsets,
              hit_point,
              u,
              v,
              f,
              mesh_id,
            )
            base_color = wp.cw_mul(base_color, tex_color)

    roughness=float(0.5);metallic=float(0.0);specular=float(0.0)
    group=int(2)
    if geom_id>=0:group=material_group[geom_id]
    if hdr_control[0][3]>0.5:
      surface=material_surface[worldid,group]
      roughness=surface[0];metallic=surface[1];specular=surface[2]
      if group==2:
        base_color=wp.cw_mul(base_color,material_color[worldid,group])
      else:
        world_to_surface = wp.transpose(geom_xmat_in[worldid, geom_id])
        local = world_to_surface * (hit_point - geom_xpos_in[worldid, geom_id])
        params = material_pattern[worldid, group]
        base_color = material_color[worldid, group]
        if surface[3] > 3.5:
          base_color = color_texture(local, world_to_surface * normal, base_color, surface[3], params)
        else:
          base_color = base_color * surface_pattern(local, surface[3], params)
    result = wp.vec3(0.0, 0.0, 0.0)
    if wp.static(rc.use_ambient_lighting):
      len_n = wp.length(normal)
      n = normal if len_n > 0.0 else wp.vec3(0.0, 0.0, 1.0)
      n = wp.normalize(n)
      hemispheric = 0.5 * (n[2] + 1.0)
      ambient_color = wp.vec3(0.4, 0.4, 0.45) * hemispheric + wp.vec3(0.1, 0.1, 0.12) * (1.0 - hemispheric)
      result = 0.5 * wp.cw_mul(base_color, wp.cw_mul(ambient_color, ambient_gain[0]))
      if hdr_control[0][3]>0.5:
        irradiance=hdr_diffuse(n,hdr_sh,hdr_control[0])
        result=0.25*wp.cw_mul(base_color,irradiance)*(1.0-0.6*metallic)
        view=wp.normalize(ray_origin_world-hit_point)
        reflection=2.0*wp.dot(n,view)*n-view
        reflected=hdr_radiance(reflection,hdr_sky,hdr_control[0])
        reflected=reflected*(1.0-roughness)*(1.0-roughness)+irradiance*roughness*roughness
        f0=wp.vec3(0.04)*(1.0-metallic)+base_color*metallic
        result=result+0.12*specular*wp.cw_mul(f0,reflected)

    # Apply lighting and shadows
    for l in range(wp.static(m.nlight)):
      light_contribution = compute_lighting(
        geom_type,
        geom_dataid,
        geom_size,
        flex_vertadr,
        flex_edge,
        flex_radius,
        geom_xpos_in,
        geom_xmat_in,
        flexvert_xpos_in,
        use_shadows,
        bvh_id,
        group_root[worldid],
        bvh_ngeom,
        bvh_nflexgeom,
        enabled_geom_ids,
        worldid,
        mesh_bvh_id,
        hfield_bvh_id,
        flex_geom_flexid,
        flex_geom_edgeid,
        flex_bvh_id,
        flex_group_root,
        light_active[worldid % light_active.shape[0], l],
        light_type[worldid % light_type.shape[0], l],
        light_castshadow[worldid % light_castshadow.shape[0], l],
        light_xpos_in[worldid, l],
        light_xdir_in[worldid, l],
        normal,
        hit_point,
        wp.static(rc.enable_backface_culling),
      )
      if hdr_control[0][3]>0.5:
        result=result+wp.cw_mul(base_color,direct_gain[l])*light_contribution*(1.0-0.6*metallic)
        if light_contribution>0.0:
          L=wp.normalize(-light_xdir_in[worldid,l])
          if light_type[worldid % light_type.shape[0],l]!=1:
            L=wp.normalize(light_xpos_in[worldid,l]-hit_point)
          n=wp.normalize(normal);view=wp.normalize(ray_origin_world-hit_point)
          halfway=wp.normalize(L+view)
          exponent=wp.max(2.0,2.0/wp.max(roughness*roughness,0.02)-2.0)
          highlight=wp.pow(wp.max(0.0,wp.dot(n,halfway)),exponent)
          visibility=light_contribution/wp.max(wp.dot(n,L),0.001)
          f0=wp.vec3(0.04)*(1.0-metallic)+base_color*metallic
          result=result+specular*highlight*visibility*wp.cw_mul(f0,direct_gain[l])
      else:
        result = result + wp.cw_mul(base_color, direct_gain[l]) * light_contribution

    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

    rgb_out[worldid, rgb_adr[cam_idx] + rayid_local] = pack_rgba_to_uint32(
      hit_color[0] * 255.0,
      hit_color[1] * 255.0,
      hit_color[2] * 255.0,
      255.0,
    )

  wp.launch(
    kernel=_render_megakernel,
    dim=(d.nworld, rc.total_rays),
    inputs=[
      m.geom_type,
      m.geom_dataid,
      m.geom_matid,
      m.geom_size,
      m.geom_rgba,
      m.cam_projection,
      m.cam_fovy,
      m.cam_sensorsize,
      m.cam_intrinsic,
      m.light_type,
      m.light_castshadow,
      m.light_active,
      m.flex_vertadr,
      m.flex_edge,
      m.flex_radius,
      m.mesh_faceadr,
      m.mat_texid,
      m.mat_texrepeat,
      m.mat_rgba,
      rc.geoloco_ambient_gain,
      rc.geoloco_direct_gain,
      rc.groot_sh,
      rc.groot_sky,
      rc.groot_controls,
      rc.groot_colors,
      rc.groot_surface,
      rc.groot_pattern,
      rc.groot_groups,
      d.geom_xpos,
      d.geom_xmat,
      d.cam_xpos,
      d.cam_xmat,
      d.light_xpos,
      d.light_xdir,
      d.flexvert_xpos,
      rc.nrender,
      rc.use_shadows,
      rc.bvh_ngeom,
      rc.bvh_nflexgeom,
      rc.cam_res,
      rc.cam_id_map,
      rc.ray,
      rc.rgb_adr,
      rc.depth_adr,
      rc.seg_adr,
      rc.render_rgb,
      rc.render_depth,
      rc.render_seg,
      rc.bvh_id,
      rc.group_root,
      rc.flex_bvh_id,
      rc.flex_group_root,
      rc.enabled_geom_ids,
      rc.mesh_bvh_id,
      rc.mesh_facetexcoord,
      rc.mesh_texcoord,
      rc.mesh_texcoord_offsets,
      rc.hfield_bvh_id,
      rc.flex_rgba,
      rc.flex_geom_flexid,
      rc.flex_geom_edgeid,
      rc.textures,
    ],
    outputs=[
      rc.rgb_data,
      rc.depth_data,
      rc.seg_data,
    ],
  )
