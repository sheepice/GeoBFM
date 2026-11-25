"""Station approach for the LightNav G1 recording. Dimensions are in meters."""

from dataclasses import dataclass
import mujoco
import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    x: float
    y: float
    length: float
    width: float
    height: float
    style: str


def fixtures(tile):
    start = tile["start"]
    return [
        Fixture("crate", start * 0.28, 0.76, 0.80, 0.92, 0.73, "crate"),
        Fixture("barricade", start * 0.62, -1.17, 0.46, 1.06, 0.57, "barrier"),
        Fixture("stack", start * 0.85, 1.12, 0.78, 1.02, 0.92, "crate"),
    ]


def build_scene(cfg, tile):
    size = cfg.scene.terrain.terrain_generator.size
    origin = np.array(tile["origin"]) - [size[0] / 2, size[1] / 2, 0.0]
    if getattr(cfg.commands["motion"], "local_coordinates", False):
        origin = np.zeros(3)
    previous = cfg.scene.spec_fn

    def decorate(spec):
        if previous:
            previous(spec)

        def box(name, center, dimensions, color, solid=False):
            return spec.worldbody.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.asarray(dimensions) / 2,
                pos=origin + center,
                rgba=(*color, 1),
                contype=int(solid),
                conaffinity=int(solid),
                group=0,
            )

        # A workshop with an open camera-facing side, windows and a raised landing.
        span = tile["goal"] + 5
        back_y = 2.50
        box("station_wall", (span / 2 - 2, back_y, 1.65), (span, 0.12, 3.3), (0.64, 0.70, 0.70), True)
        box("station_wainscot", (span / 2 - 2, back_y - 0.08, 0.44), (span, 0.06, 0.88), (0.24, 0.39, 0.42))
        for i, x in enumerate(np.arange(-0.50, span - 2, 2.45)):
            box(f"window_frame_{i}", (x, back_y - 0.09, 2.04), (1.91, 0.08, 1.08), (0.14, 0.22, 0.24))
            box(f"window_glass_{i}", (x, back_y - 0.14, 2.04), (1.77, 0.035, 0.94), (0.53, 0.75, 0.79))
            box(f"window_mullion_{i}", (x, back_y - 0.17, 2.04), (0.04, 0.04, 0.96), (0.24, 0.34, 0.35))
        far = tile["goal"] + 2.65
        box("station_end", (far, 0, 1.72), (0.12, 5.0, 3.44), (0.56, 0.66, 0.66), True)
        box("landing_door", (far - 0.08, 0, 1.94), (0.08, 1.03, 2.04), (0.16, 0.33, 0.35))
        box("landing_handle", (far - 0.14, -0.36, 1.82), (0.06, 0.10, 0.04), (0.88, 0.70, 0.37))
        spec.worldbody.add_light(
            name="station_daylight",
            pos=origin + [3, -2.5, 6],
            dir=(0.1, 0.3, -1),
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=(0.76, 0.75, 0.68),
            ambient=(0.29, 0.30, 0.31),
            castshadow=False,
        )
        for item in fixtures(tile):
            x, y, L, W, H = item.x, item.y, item.length, item.width, item.height
            wood = item.style == "crate"
            box(
                "nav_" + item.name,
                (x, y, H / 2),
                (L, W, H),
                (0.61, 0.37, 0.18) if wood else (0.92, 0.31, 0.075),
                True,
            )
            if wood:
                for j, offset in enumerate([-0.32, 0.32]):
                    box(
                        f"{item.name}_band_{j}",
                        (x + offset * L, y, H / 2),
                        (L * 0.07, W + 0.008, H + 0.008),
                        (0.26, 0.19, 0.13),
                    )
                for j, h in enumerate([0.2, 0.8]):
                    box(
                        f"{item.name}_rim_{j}",
                        (x, y, h * H),
                        (L + 0.014, W + 0.013, 0.043),
                        (0.76, 0.53, 0.30),
                    )
            else:
                for j, h in enumerate([0.40, 0.78]):
                    box(
                        f"{item.name}_stripe_{j}",
                        (x, y, h * H),
                        (L + 0.012, W + 0.010, 0.075),
                        (0.95, 0.87, 0.65),
                    )
        # Edge insets are distributed only on the side margins of the physical staircase.
        for j, (edge, level) in enumerate(zip(tile["edges"], np.cumsum(tile["jumps"]))):
            for side in (-1, 1):
                box(
                    f"stair_mark_{j}_{side}",
                    (edge + 0.035, side * 1.73, level + 0.003),
                    (0.052, 0.36, 0.006),
                    (0.91, 0.71, 0.35),
                )

    cfg.scene.spec_fn = decorate


class FixtureContacts:
    """Check robot contacts at the 50 Hz recorded poses using compiled MuJoCo geometry."""

    def __init__(self, sim, tile):
        self.sim = sim
        self.model = sim._mj_model
        self.data = mujoco.MjData(self.model)
        self.names = {}
        for name in ["nav_" + f.name for f in fixtures(tile)] + ["station_wall", "station_end"]:
            geom = self.model.geom(name).id
            self.names[geom] = name
        self.robot = {i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith("robot/")}
        if not self.robot:
            raise ValueError("Robot geometry group empty")
        self.counts = dict.fromkeys(self.names.values(), 0)

    def sample(self):
        self.data.qpos[:] = self.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_fwdPosition(self.model, self.data)
        hits = set()
        for contact in self.data.contact:
            a, b = map(int, contact.geom)
            if contact.dist > 0:
                continue
            if a in self.names and b in self.robot:
                hits.add(self.names[a])
            if b in self.names and a in self.robot:
                hits.add(self.names[b])
        for name in hits:
            self.counts[name] += 1
        return sorted(hits)


class AisleSpeedGuide:
    """Limit speed and steer away from the three mapped fixtures along the aisle."""

    def __init__(self, tile, clearance=0.58):
        self.items = fixtures(tile)
        self.clearance = clearance
        self.x_knots = [-1.0]
        self.y_knots = [0.0]
        for item in self.items:
            side = np.sign(item.y)
            inner = item.y - side * item.width / 2
            lane = inner - side * clearance
            self.x_knots.extend([item.x - item.length / 2 - 0.15, item.x + item.length / 2 + 0.15])
            self.y_knots.extend([lane, lane])
        self.x_knots.append(self.x_knots[-1] + 1.2)
        self.y_knots.append(0.0)

    def update(self, xy, yaw, v, w):
        x, y = xy
        ahead = 0.67
        near = any(abs(x - item.x) < item.length / 2 + 0.52 for item in self.items)
        target_y = np.interp(x + ahead, self.x_knots, self.y_knots)
        # Allow the predicted turn to bias the target within a narrow band.
        target_y += np.clip(0.12 * w, -0.045, 0.045)
        heading = float(np.clip(np.arctan2(target_y - y, ahead), -0.65, 0.65))
        error = np.arctan2(np.sin(heading - yaw), np.cos(heading - yaw))
        return min(v, 0.25 if near else 0.4), float(np.clip(1.4 * error, -0.35, 0.35)), heading


class PhysicsStepContacts:
    """Measure fixture penetration and nearest collidable surface at every physics step."""

    def __init__(self, raw, tile):
        self.observer = FixtureContacts(raw.sim, tile)
        self.samples = 0
        self.step_hits = set()
        self.min_distance = 1.0
        self.closest = None
        self.solids = {}
        m = self.observer.model
        for geom, name in self.observer.names.items():
            if not name.startswith("nav_"):
                continue
            self.solids[geom] = np.array(
                sorted(
                    i
                    for i in self.observer.robot
                    if m.geom_contype[i] & m.geom_conaffinity[geom]
                    or m.geom_conaffinity[i] & m.geom_contype[geom]
                ),
                dtype=int,
            )
        previous = raw.metrics_manager.compute_substep

        def observe_substep():
            previous()
            self.step_hits.update(self.observer.sample())
            self.samples += 1
            m, d = self.observer.model, self.observer.data
            # Bounding spheres only cull pairs. The recorded distance is MuJoCo's narrow-phase distance.
            for geom, others in self.solids.items():
                bound = (
                    np.linalg.norm(d.geom_xpos[others] - d.geom_xpos[geom], axis=1)
                    - m.geom_rbound[others]
                    - m.geom_rbound[geom]
                )
                for i in others[bound < self.min_distance]:
                    dist = mujoco.mj_geomDistance(m, d, int(i), geom, self.min_distance, None)
                    if dist < self.min_distance:
                        self.min_distance = float(dist)
                        self.closest = [m.geom(int(i)).name, self.observer.names[geom]]

        raw.metrics_manager.compute_substep = observe_substep
