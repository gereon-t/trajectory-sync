import collections
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Position:
    timestamp: float
    x: float
    y: float
    z: float
    v: float


@dataclass
class PositionInfo:
    position: Position
    path_length: float = 0.0

    @property
    def timestamp(self):
        return self.position.timestamp


class SensorState:
    def __init__(self):
        self.buffer: collections.deque[PositionInfo] = collections.deque()
        self.path_length = 0.0
        self.last_position = None
        self.offsets = []
        self.avg_offset = 0.0


class Synchronizer:
    def __init__(
        self,
        calibration_samples: int = 20,
        initial_dt: float = 0.0,
        delta_t_var: float = 1e-12,
        bias_var: float = 1e-3,
        delta_path_length_var: float = 0.1,
    ):
        self.calibration_samples = calibration_samples
        self.is_calibrated = False
        self._lock = threading.Lock()

        self.sensors = {"primary": SensorState(), "secondary": SensorState()}

        # kalman filter
        self.x = np.array([[initial_dt], [0.0]])
        self.P = np.diag([1.0, 10.0])
        self.delta_t_var = delta_t_var
        self.bias_var = bias_var
        self.R = np.array([[delta_path_length_var]])
        self.last_update_time = None

    def clear(self):
        with self._lock:
            self.is_calibrated = False
            for s in self.sensors.values():
                s.buffer.clear()
                s.path_length = 0.0
                s.last_position = None
                s.offsets.clear()
                s.avg_offset = 0.0

            self.x = np.array([[0.0], [0.0]])
            self.P = np.diag([1.0, 10.0])
            self.last_update_time = None

    @property
    def state(self):
        p = self.sensors["primary"]
        s = self.sensors["secondary"]
        return {
            "delta_t": self.x[0, 0] + s.avg_offset - p.avg_offset,
            "bias": self.x[1, 0],
            "sigma_delta_t": np.sqrt(self.P[0, 0]),
            "sigma_bias": np.sqrt(self.P[1, 1]),
        }

    def on_new_position_sensor_primary(self, position: Position):
        self._handle_sensor_input("primary", position)

    def on_new_position_sensor_secondary(self, position: Position):
        self._handle_sensor_input("secondary", position)

    def _handle_sensor_input(self, role: str, position: Position):
        with self._lock:
            s = self.sensors[role]

            # Get rough time offset for calibration
            if not self.is_calibrated:
                s.offsets.append(time.time() - position.timestamp)
                self._check_calibration()
                return

            # apply average offset to the timestamp for synchronization
            position.timestamp += s.avg_offset

            # update path length
            if s.last_position is not None:
                p1 = np.array([position.x, position.y, position.z])
                p2 = np.array([s.last_position.x, s.last_position.y, s.last_position.z])
                s.path_length += np.linalg.norm(p1 - p2)

            s.last_position = position
            s.buffer.append(PositionInfo(position=position, path_length=s.path_length))

            self._try_matching()

    def _check_calibration(self):
        p, s = self.sensors["primary"], self.sensors["secondary"]
        if (
            len(p.offsets) >= self.calibration_samples
            and len(s.offsets) >= self.calibration_samples
        ):
            p.avg_offset = sum(p.offsets) / len(p.offsets)
            s.avg_offset = sum(s.offsets) / len(s.offsets)
            print(
                f"Calibration done. Shift P: {p.avg_offset:.3f}s, Shift S: {s.avg_offset:.3f}s"
            )
            self.is_calibrated = True

    def _try_matching(self):
        p_buf = self.sensors["primary"].buffer
        s_buf = self.sensors["secondary"].buffer

        # go through all queued secondary positions
        while s_buf:
            sec_pos = s_buf[0]

            # remove too old primary positions
            while len(p_buf) > 1 and p_buf[1].timestamp < sec_pos.timestamp:
                p_buf.popleft()

            # break if interpolation is not possible
            if len(p_buf) < 2:
                break

            prim_before, prim_after = p_buf[0], p_buf[1]

            if prim_before.timestamp <= sec_pos.timestamp <= prim_after.timestamp:
                s_buf.popleft()
                self._process_matched_pair(prim_before, prim_after, sec_pos)
            elif sec_pos.timestamp < prim_before.timestamp:
                s_buf.popleft()
            else:
                break

    def _process_matched_pair(
        self, prim_before: PositionInfo, prim_after: PositionInfo, sec_pos: PositionInfo
    ):
        dt_total = prim_after.timestamp - prim_before.timestamp
        ratio = (
            (sec_pos.timestamp - prim_before.timestamp) / dt_total
            if dt_total > 0
            else 0.0
        )

        interp_path_length = prim_before.path_length + ratio * (
            prim_after.path_length - prim_before.path_length
        )

        interp_v = prim_before.position.v + ratio * (
            prim_after.position.v - prim_before.position.v
        )

        delta_path_length = sec_pos.path_length - interp_path_length

        self._update(
            measurement_time=sec_pos.timestamp,
            delta_path_length=delta_path_length,
            velocity=interp_v,
        )

        estimated = self.state
        print(
            f"Estimated State at {sec_pos.timestamp:.3f}s: Delta_t={estimated['delta_t']:.6f} s, Bias={estimated['bias']:.3f} m"
        )

    def _update(
        self, measurement_time: float, delta_path_length: float, velocity: float
    ):
        # Calculate time step since last update
        if self.last_update_time is None:
            dt_step = 0.1
        else:
            dt_step = measurement_time - self.last_update_time
        self.last_update_time = measurement_time

        # prediction
        Q = np.diag([self.delta_t_var, self.bias_var]) * dt_step

        self.P = self.P + Q

        # Jacobian matrix H = ∂h/∂x evaluated at current state estimates
        H = np.array([[velocity, 1.0]])

        h_x = H @ self.x  # predicted measurement based on current state

        # Actual measurements
        z = np.array([[delta_path_length]])

        # innovation
        y = z - h_x

        # Calculate Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ H) @ self.P
