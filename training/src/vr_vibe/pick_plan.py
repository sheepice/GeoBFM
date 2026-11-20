"""Scripted, feedback-gated task planner; observations use simulation poses."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def smooth(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3 - 2 * value)


class PickPlan:
    # Approximate centre of the power-grasp workspace in the G1 wrist frame.
    grasp_offset = np.array([.1315, .035, 0.])
    training_offset = np.array([.18, .025, 0.])

    def __init__(self, model, data, mapping, configuration):
        self.model, self.mapping = model, mapping
        self.grasp_offset = np.asarray(configuration.get('grasp_offset_m', self.grasp_offset))
        self.wrist = model.body('right_wrist_yaw_link').id
        self.payload = model.body('payload').id
        self.start_object = data.xpos[self.payload].copy()
        self.start_hand = data.xpos[self.wrist] + data.xmat[self.wrist].reshape(3, 3) @ self.grasp_offset
        self.phase, self.since = 'SETTLE', 0.
        self.correction = np.zeros(3)
        self.orientation_correction = np.zeros(3)
        self.events = []
        self.close = 0.
        self.carry_local = None
        self.last_target = self.start_hand.copy()
        self.route_goal = configuration['stairs_start_m'] + 1.92 + .5
        self.hold = 0
        self.stage_start = self.start_hand.copy()
        self.grasp_target = self.start_object.copy()
        self.previous_hand = self.start_hand.copy()
        self.hand_velocity = np.zeros(3)

    def transition(self, phase, t):
        self.events.append(dict(t=float(t), phase=phase))
        self.phase, self.since = phase, t

    def update(self, data, t, default_points, default_orientations):
        root = data.qpos[:3]
        root_rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
        wrist_rotation = Rotation.from_matrix(data.xmat[self.wrist].reshape(3, 3))
        actual_hand = data.xpos[self.wrist] + wrist_rotation.apply(self.grasp_offset)
        self.hand_velocity = .8*self.hand_velocity + .2*(actual_hand-self.previous_hand)/.02
        self.previous_hand = actual_hand.copy()
        object_position = data.xpos[self.payload]
        elapsed = t - self.since
        target = self.start_hand.copy()
        speed = 0.
        if self.phase == 'SETTLE' and elapsed >= 1.:
            self.start_hand = actual_hand.copy()
            self.stage_start = actual_hand.copy()
            self.transition('RAISE_HAND', t)
            elapsed = 0
        waypoints = {
            'RAISE_HAND': (np.array([.18, -.22, 1.03]), 2., 'PREGRASP'),
            'PREGRASP': (self.start_object + [0, -.05, .10], 2., 'LOWER_HAND'),
            'LOWER_HAND': (self.start_object + [0, -.05, 0], 1.5, 'APPROACH'),
            'APPROACH': (self.start_object + [.008, -.01, .015], 4., 'CLOSE'),
        }
        if self.phase in waypoints:
            end, duration, next_phase = waypoints[self.phase]
            w = smooth(elapsed/duration)
            target = self.stage_start*(1-w) + end*w
            tolerance = .010 if self.phase == 'LOWER_HAND' else .035
            to_object = actual_hand-self.start_object
            aligned = (np.linalg.norm(to_object)<.022 and -.012<to_object[1]<.008
                and abs(to_object[2])<.012 and wrist_rotation.magnitude()<.22)
            if self.phase == 'APPROACH' and elapsed>.5 and aligned:
                self.grasp_target = actual_hand.copy()
                self.transition('CLOSE', t)
            elif self.phase != 'APPROACH' and elapsed >= duration+.4 and np.linalg.norm(actual_hand-end) < tolerance:
                self.stage_start = end.copy()
                self.transition(next_phase, t)
            elif elapsed > duration+14:
                self.transition('REACH_TIMEOUT', t)
        elif self.phase == 'CLOSE':
            target = self.grasp_target.copy()
            self.close = smooth(max(0, elapsed-.5)/1.2)
            if elapsed >= 2.:
                self.transition('LIFT', t)
        elif self.phase == 'LIFT':
            target = self.grasp_target + [0, 0, .16*smooth(elapsed/3.)]
            self.close = 1.
            lifted = object_position[2] - self.start_object[2] > .09
            retained = np.linalg.norm(actual_hand-object_position) < .09
            self.hold = self.hold+1 if lifted and retained else 0
            if elapsed > 3.2 and self.hold >= 25:
                self.carry_local = root_rotation.inv().apply(target-root)
                self.transition('CARRY_UPSTAIRS', t)
            elif elapsed > 6:
                self.transition('GRASP_FAILED', t)
        elif self.phase == 'CARRY_UPSTAIRS':
            target = root + root_rotation.apply(self.carry_local)
            speed = .4
            self.close = 1.
            if np.linalg.norm(actual_hand-object_position) > .2:
                self.transition('OBJECT_DROPPED', t)
            elif root[0] > self.route_goal:
                self.transition('COMPLETE', t)
        elif self.phase == 'COMPLETE':
            target = root + root_rotation.apply(self.carry_local)
            self.close = 1.
        elif self.phase in ('REACH_TIMEOUT', 'GRASP_FAILED', 'OBJECT_DROPPED'):
            target = self.last_target.copy()
        # Small reference feedback correction compensates the checkpoint's hand
        # tracking bias. It changes target input only, never robot joint actions.
        error = target - actual_hand
        if self.phase not in ('SETTLE', 'COMPLETE', 'OBJECT_DROPPED'):
            self.correction = np.clip(self.correction + error * .02 * .7, -.14, .14)
        orientation_error = wrist_rotation.inv().as_rotvec()
        if self.phase not in ('SETTLE', 'RAISE_HAND', 'OBJECT_DROPPED') and np.linalg.norm(error)<.07:
            self.orientation_correction = np.clip(self.orientation_correction + orientation_error*.02*.3, -.30, .30)
        feedback = np.clip(.4*error-.08*self.hand_velocity, -.04, .04)
        desired_training_point = target + wrist_rotation.apply(self.training_offset-self.grasp_offset) + self.correction + feedback
        points = default_points.copy()
        orientations = default_orientations.copy()
        if self.phase != 'SETTLE':
            points[1] = root_rotation.inv().apply(desired_training_point-root)
            commanded_rotation = Rotation.from_rotvec(self.orientation_correction)
            orientations[1] = (root_rotation.inv()*commanded_rotation).as_quat()[[3, 0, 1, 2]]
        self.last_target = target.copy()
        # Six independently actuated channels. Distal joints use URDF mimic constraints.
        closure = np.array([.7, .75, 1.35, 1.35, 1.35, 1.35]) * self.close
        if self.phase == 'CLOSE':
            closure[0] = .7*smooth((t-self.since)/.6)
        for side, drives in self.mapping['hand_drives'].items():
            requested = closure if side == 'right' else np.zeros(6)
            data.ctrl[drives] += np.clip(requested-data.ctrl[drives], -.03, .03)
        contacts = []
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if self.model.geom_bodyid[a] == self.payload:
                contacts.append(self.model.geom(b).name)
            elif self.model.geom_bodyid[b] == self.payload:
                contacts.append(self.model.geom(a).name)
        return speed, points, orientations, dict(phase=self.phase, grasp_error_m=float(np.linalg.norm(error)),
            grasp_position=actual_hand.tolist(), object_position=object_position.tolist(),
            object_lift_m=float(object_position[2]-self.start_object[2]), hand_closure=float(self.close),
            target=target.tolist(), reference_correction=self.correction.tolist(), object_contacts=contacts,
            wrist_orientation_error_deg=float(np.rad2deg(wrist_rotation.magnitude())))
