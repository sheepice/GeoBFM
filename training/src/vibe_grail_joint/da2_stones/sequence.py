"""Contact-event state machine: visit each new pillar with alternating feet."""
import torch


def lateral_foot_separation(feet, root_quaternion):
    """Positive means left ankle is left of right ankle in robot heading frame.

    Uses only yaw heading (world xy), so a world rotation leaves this unchanged.
    """
    w, x, y, z = root_quaternion.unbind(-1)
    forward = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z)), -1)
    forward = forward / forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    return ((feet[:, 0, :2] - feet[:, 1, :2]) * left).sum(-1)


def loaded_top_bridge(surface_ids, top_contact, normal_load, minimum_load=15.):
    """Only distinct tops actually bearing load constitute one-foot bridging."""
    loaded = top_contact & (normal_load > minimum_load) & (surface_ids >= 0)
    smallest = torch.where(loaded, surface_ids, 10000).amin(-1)
    largest = torch.where(loaded, surface_ids, -1).amax(-1)
    return loaded.any(-1) & (smallest != largest)


class AlternatingSequence:
    def __init__(self, count, device):
        self.progress = torch.zeros(count, dtype=torch.long, device=device)
        self.last_foot = torch.full_like(self.progress, -1)
        self.air = torch.zeros(count, 2, dtype=torch.long, device=device)
        self.armed = torch.zeros(count, 2, dtype=torch.bool, device=device)
        self.streak = self.air.clone()
        self.last_surface = torch.full_like(self.air, -2)
        self.last_event_surface = torch.full_like(self.air, -2)
        self.wrong_side = torch.zeros_like(self.progress)
        self.first_correct = torch.zeros_like(self.progress)
        self.first_wrong = torch.zeros_like(self.progress)

    def reset(self, ids):
        self.progress[ids] = 0
        self.last_foot[ids] = -1
        self.air[ids] = 0
        self.armed[ids] = False
        self.streak[ids] = 0
        self.last_surface[ids] = -2
        self.last_event_surface[ids] = -2
        self.wrong_side[ids] = 0
        self.first_correct[ids] = 0
        self.first_wrong[ids] = 0

    def step(self, support, surface, valid_support, bridge, first_foot=None, forbidden=None):
        # first_foot: left=0, right=1, -1=unconstrained (legacy/flat worlds).
        self.wrong_side.zero_()
        self.first_correct.zero_()
        self.first_wrong.zero_()
        self.air = torch.where(~support, self.air + 1, 0)
        self.armed |= self.air >= 2
        same = surface == self.last_surface
        self.streak = torch.where(support & same, self.streak + 1, support.long())
        self.last_surface.copy_(surface)
        # A full swing and two supported control steps are required. Standing,
        # contact chatter, sliding and one foot bridging two tops earn no step.
        landing = self.armed & (self.streak >= 2)
        event = landing & valid_support & ~bridge & (surface >= 0)
        self.armed &= ~landing
        advance = torch.zeros_like(self.progress)
        wrong = torch.zeros_like(self.progress)
        simultaneous = event.all(-1)
        for foot in (0, 1):
            good = event[:, foot] & ~simultaneous
            next_top = surface[:, foot] == self.progress
            alternate = (self.last_foot < 0) | (self.last_foot != foot)
            side_ok = torch.ones_like(good) if first_foot is None else (
                (first_foot < 0) | ((first_foot + surface[:, foot]) % 2 == foot))
            bad_side = event[:, foot] & (surface[:, foot] >= self.progress) & ~side_ok
            self.wrong_side += bad_side.long()
            self.first_wrong += (bad_side & (surface[:, foot] == 0) & (self.progress == 0)).long()
            allowed = torch.ones_like(good) if forbidden is None else ~forbidden
            accepted = good & next_top & alternate & side_ok & allowed
            self.first_correct += (accepted & (self.progress == 0)).long()
            wrong += (good & (surface[:, foot] >= self.progress) & ~accepted).long()
            advance += accepted.long()
            self.progress += accepted.long()
            self.last_foot = torch.where(accepted, foot, self.last_foot)
        wrong += simultaneous.long()
        return advance.float(), wrong.float()
