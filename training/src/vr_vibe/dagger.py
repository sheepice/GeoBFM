"""Keep student-driven rollouts; use a mode-matched original SONIC teacher."""

from vibe_grail_joint.da2_locomotion.collection import GeometryDaggerEnv
from vibe_grail_joint.velocity.dagger_buffer import DaggerBuffer
from .model import build_teacher


class VRDaggerEnv(GeometryDaggerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.unwrapped.command_manager.get_term("motion")
        self.buffer = DaggerBuffer(
            self.settings["dagger"]["buffer_capacity"],
            len(c.flat_sources) + 2 * len(c.cfg.planner_speeds),
            self.settings["seed"] + self.rank,
        )

    def attach(self, algorithm, actor_cfg, obs_groups):
        self.actor = algorithm.get_policy()
        self.teacher, digest = build_teacher(
            self._full_obs[:1],
            actor_cfg,
            obs_groups,
            self.settings["dagger"]["teacher_checkpoint"],
            self.device,
        )
        algorithm.dagger_buffer = self.buffer
        algorithm.teacher_sha256 = digest
