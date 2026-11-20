"""Inherit every trained DA2 actor parameter and add the frozen TELEOP encoder."""

from vibe_grail_joint.da2_locomotion.model import TerrainConditionedActor

from .model import DualEncoderMixin


class DA2TeleopActor(DualEncoderMixin, TerrainConditionedActor):
    def __init__(self, *args, teleop_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_teleop(teleop_checkpoint)

    def as_onnx(self, *args, **kwargs):
        raise NotImplementedError(
            "DA2 TELEOP requires an export with DA2 history and three-point inputs"
        )
