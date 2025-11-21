import numpy as np


# HELPER CLASS: 1D Kalman Filter
class KalmanSmoother1D:
    """
    A 1D Kalman Filter for smoothing a single value.
    It models the state as [position, velocity].
    """

    def __init__(self, R=1e-2, Q=1e-3, P_init=1.0, dt=1.0):
        self.R = R  # Measurement noise covariance
        self.Q = np.diag([Q, Q])  # Process noise covariance

        self.F = np.array([[1, dt], [0, 1]])  # State transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix

        self.P = np.diag([P_init, P_init])  # Initial state covariance
        self.x = np.zeros(2)  # Initial state (pos, vel)
        self.is_initialized = False

    def smooth(self, z):
        """Smooths a new measurement z."""
        if not self.is_initialized:
            self.x[0] = z
            self.is_initialized = True
            return self.x[0]

        # --- Predict step ---
        # x_pred = F * x
        self.x = self.F @ self.x
        # P_pred = F * P * F.T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        # --- Update step ---
        # y = z - H * x_pred
        y = z - self.H @ self.x

        # S = H * P_pred * H.T + R
        S = self.H @ self.P @ self.H.T + self.R

        # K = P_pred * H.T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # x_new = x_pred + K * y
        self.x = self.x + K @ y

        # P_new = (I - K * H) * P_pred
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x[0]


class OneEuroFilter:
    """
    Implementation of the 1-Euro Filter.
    Code based on: http://www.lifl.fr/~casiez/1euro/
    """

    def __init__(self, freq, fcmin=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.fcmin = fcmin
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = self._ema_filter()
        self.dx_filter = self._ema_filter()
        self.last_t = None

    def _ema_filter(self):
        """Creates a new exponential moving average filter."""
        return {"val": 0.0, "s": 0.0, "initialized": False}

    def _alpha(self, cutoff, dt):
        """Calculates the alpha value for a given cutoff frequency and dt."""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _ema(self, f, x, alpha):
        """Applies the EMA filter."""
        if not f["initialized"]:
            f["s"] = x
            f["initialized"] = True
        else:
            f["s"] = alpha * x + (1.0 - alpha) * f["s"]
        f["val"] = f["s"]
        return f

    def filter(self, x, t=None):
        """Filters a new value x at time t."""
        # 1. Handle time
        if t is None:
            if self.last_t is None:
                self.last_t = 0.0
            t = self.last_t + 1.0 / self.freq

        # Calculate dynamic dt
        if self.last_t is not None and t != self.last_t:
            dt = t - self.last_t
        else:
            dt = 1.0 / self.freq

        self.last_t = t

        # 2. Calculate derivative (Velocity)
        if self.x_filter["initialized"]:
            dx = (x - self.x_filter["s"]) / dt  # velocity = delta_x / delta_t
        else:
            dx = 0.0

        # 3. Filter the derivative
        dx_alpha = self._alpha(self.d_cutoff, dt)  # <--- calling with dt
        self.dx_filter = self._ema(self.dx_filter, abs(dx), dx_alpha)

        # 4. Calculate adaptive cutoff
        cutoff = self.fcmin + self.beta * self.dx_filter["val"]

        # 5. Filter the signal
        x_alpha = self._alpha(cutoff, dt)  # <--- calling with dt
        self.x_filter = self._ema(self.x_filter, x, x_alpha)

        return self.x_filter["val"]


# MAIN SMOOTHER CLASSES


class TransformSmoother:
    """
    Smooths the 12-D transformation parameters.
    - Translation (3 params): One-Euro Filter
    - Rotation (as 4D quaternion): One-Euro Filter
    - Scale (1 param): One-Euro Filter
    """

    def __init__(
        self,
        translation_fcmin,
        translation_beta,
        scale_fcmin,
        scale_beta,
        rotation_fcmin,
        rotation_beta,
        freq=30.0,
    ):
        # 3 One-Euro filters for 3 translation params
        self.one_euro_x = OneEuroFilter(
            freq=freq, fcmin=translation_fcmin, beta=translation_beta
        )
        self.one_euro_y = OneEuroFilter(
            freq=freq, fcmin=translation_fcmin, beta=translation_beta
        )
        self.one_euro_z = OneEuroFilter(
            freq=freq, fcmin=translation_fcmin, beta=translation_beta
        )

        # One-Euro filter for 1 scale param
        self.one_euro_s = OneEuroFilter(freq=freq, fcmin=scale_fcmin, beta=scale_beta)

        # 4 One-Euro filters for 4 quaternion params
        self.one_euro_q = [
            OneEuroFilter(freq=freq, fcmin=rotation_fcmin, beta=rotation_beta)
            for _ in range(4)
        ]

        # A quaternion and its negative capture the same rotation
        # So our filter should be mindful of that
        self.last_q_smooth = None

    def _rotation_matrix_to_quaternion(self, R):
        """Converts a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
        qw = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        return np.array([qw, qx, qy, qz])

    def _quaternion_to_rotation_matrix(self, q):
        """Converts a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
        w, x, y, z = q
        R = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )
        return R

    def smooth(self, params, t=None):
        # 1. DECOMPOSITION
        T = params.reshape(3, 4)
        t_vec = T[:, 3]
        M = T[:, :3]

        # Extract scale (norm of the first column)
        scale = np.linalg.norm(M[:, 0])
        # Extract rotation matrix
        R = M / (scale + 1e-8)  # Add epsilon to avoid division by zero

        # Convert rotation to quaternion
        q = self._rotation_matrix_to_quaternion(R)

        # 2. FILTERING
        # Filter translation with One-Euro filters
        t_x_smooth = self.one_euro_x.filter(t_vec[0], t)
        t_y_smooth = self.one_euro_y.filter(t_vec[1], t)
        t_z_smooth = self.one_euro_z.filter(t_vec[2], t)
        t_smooth = np.array([t_x_smooth, t_y_smooth, t_z_smooth])

        # Filter scale with One-Euro Filter
        s_smooth = self.one_euro_s.filter(scale, t)

        # Filter quaternion with One-Euro Filters

        # The quaternion vector may be opposite to the one in the previous frame
        # For nice linear smoothing, we will have to flip it
        if self.last_q_smooth is not None:
            if np.dot(q, self.last_q_smooth) < 0.0:
                q = -q

        q_smooth = np.array(
            [
                self.one_euro_q[0].filter(q[0], t),
                self.one_euro_q[1].filter(q[1], t),
                self.one_euro_q[2].filter(q[2], t),
                self.one_euro_q[3].filter(q[3], t),
            ]
        )

        # Re-normalize quaternion
        q_smooth /= np.linalg.norm(q_smooth) + 1e-8

        self.last_q_smooth = q_smooth

        # 3. RECONSTRUCTION
        # Convert smoothed quaternion back to rotation matrix
        R_smooth = self._quaternion_to_rotation_matrix(q_smooth)

        # Rebuild scaled rotation matrix
        M_smooth = s_smooth * R_smooth

        # Rebuild 3x4 transform matrix
        T_smooth = np.hstack([M_smooth, t_smooth.reshape(3, 1)])

        return T_smooth.flatten()


class ShapeSmoother:
    """
    Smooths the 40-D shape parameters using Exponential Moving Average (EMA).
    Assumes shape is constant, so uses a very strong smoothing factor.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.last_params = None

    def smooth(self, params):
        if self.last_params is None:
            self.last_params = params
            return params

        smoothed = self.alpha * params + (1 - self.alpha) * self.last_params
        self.last_params = smoothed
        return smoothed


class ExpressionSmoother:
    """
    Smooths the 10-D expression parameters using One-Euro Filters.
    """

    def __init__(self, freq=30.0, fcmin=1.0, beta=0.05):
        self.filters = [OneEuroFilter(freq, fcmin, beta) for _ in range(10)]

    def smooth(self, params, t=None):
        smoothed_params = np.array(
            [self.filters[i].filter(params[i], t) for i in range(10)]
        )
        return smoothed_params


# Gotta use 'em all!
class TemporalSmoother:
    """
    Main class to apply temporal smoothing to 3DDFA-V2 parameters.
    Splits the 62-D vector and applies specialized smoothing to each part.
    """

    def __init__(
        self,
        video_fps=30.0,
        translation_fcmin=0.1,
        translation_beta=0.05,
        scale_fcmin=0.01,
        scale_beta=0.01,
        rotation_fcmin=0.05,
        rotation_beta=0.01,
        shape_alpha=0.05,
        expr_fcmin=1.0,
        expr_beta=0.05,
    ):

        self.transform_smoother = TransformSmoother(
            translation_fcmin,
            translation_beta,
            scale_fcmin,
            scale_beta,
            rotation_fcmin,
            rotation_beta,
            freq=video_fps,
        )
        self.shape_smoother = ShapeSmoother(alpha=shape_alpha)
        self.expression_smoother = ExpressionSmoother(
            freq=video_fps, fcmin=expr_fcmin, beta=expr_beta
        )

    def smooth(self, params, t=None):
        # 1. Split the 62-D vector (as per paper)
        pose_params = params[:12]  # 12 transform params
        shape_params = params[12:52]  # 40 shape params
        exp_params = params[52:]  # 10 expression params

        # 2. Smooth each part individually
        smoothed_pose = self.transform_smoother.smooth(pose_params, t)
        smoothed_shape = self.shape_smoother.smooth(shape_params)
        smoothed_exp = self.expression_smoother.smooth(exp_params, t)

        # 3. Recombine and return
        return np.concatenate([smoothed_pose, smoothed_shape, smoothed_exp])
