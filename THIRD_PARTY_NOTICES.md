# Third-party code and provenance

This repository contains selected source snapshots used by the existing DA2-ViBe experiments. Each dependency retains its original copyright and license notices. No pretrained model weights, motion datasets, robot meshes, or generated media are included.

| Snapshot | Source revision | License / notices |
| --- | --- | --- |
| [ViBe](https://github.com/lok-i/vibe) | `ed4ee6c581af9a7effcb53ee219d10784a839ca3` | [BSD-3-Clause](third_party/vibe/LICENSE), [NOTICE](third_party/vibe/NOTICE), [additional notices](third_party/vibe/THIRD_PARTY_NOTICES.md) |
| [mocke](https://github.com/lok-i/mocke) | `b50fb6d11e7616598dec8b444e88c55b559ec87b` | [BSD-3-Clause](third_party/mocke/LICENSE), [additional licenses and notices](third_party/mocke/THIRD_PARTY_NOTICES.md) |
| [ORCS](https://github.com/lok-i/orcs) | `49886d16d4b39e89442ce82ecadc7ae4288fae17` | [BSD-3-Clause](third_party/orcs/LICENSE), [additional notices](third_party/orcs/THIRD_PARTY_NOTICES.md) |
| [SONIC-compatible RSL-RL fork](https://github.com/lok-i/rsl_rl) | `215b6f4ceda569d2ca39df64dd679b9af15375cb` | [BSD-3-Clause](third_party/rsl_rl/LICENSE), [fork notes](third_party/rsl_rl/FORK.md) |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Existing metric-depth source snapshot; upstream commit not recorded | [Apache-2.0 source license](third_party/depth_anything_v2/LICENSE); individual notices in source files retained |

The DA2 snapshot did not contain its own Git metadata. We record file hashes in [third_party/manifest.json](third_party/manifest.json) rather than attributing a parent repository's commit to it. Model weights, if obtained separately, are subject to their respective model licenses; this source notice does not grant rights to external assets.

The installation script obtains [drcl-deploy/assets](https://github.com/drcl-deploy/assets) revision `90e226008bda5edc9c07ca3eb69f9f79c4603de1` into an ignored external directory. Its license remains with that separate checkout.

[GeoLoco](https://github.com/liuyufei-nubot/GeoLoco) informed the repository organization and the existing visual-prior approach. The project-specific DA2 task implementations originate in the author's GRAIL experiments and are maintained under [training/](training/).

No project-wide license for the author's original code has been selected by this cleanup. This does not change the licenses of the third-party materials above.

The opt-in [GeoLoco-style Warp radiance adapter](training/src/vibe_grail_joint/da2_locomotion/geoloco_warp_render.py)
adapts the `render` function from MuJoCo-Warp's `mujoco_warp/_src/render.py`,
Copyright 2026 The Newton Developers, Apache-2.0. Its original notice is retained
in the file; a copy of the [Apache-2.0 license](third_party/orcs/LICENSES/Apache-2.0.txt)
is included in this repository. The pinned source SHA256 is recorded in the
adapter and enforced at runtime. Only ambient/direct RGB radiance inputs were
added; geometry intersection and shadow functions use the installed dependency.
