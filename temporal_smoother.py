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


# HELPER CLASS: One-Euro Filter
class OneEuroFilter:
    """
    Implementation of the 1-Euro Filter.
    Code based on: http://www.lifl.fr/~casiez/1euro/
    """

    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = self._ema_filter()
        self.dx_filter = self._ema_filter()
        self.last_t = None

    def _ema_filter(self):
        """Creates a new exponential moving average filter."""
        return {"val": 0.0, "s": 0.0, "initialized": False}

    def _alpha(self, cutoff):
        """Calculates the alpha value for a given cutoff frequency."""
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

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
        if t is None:
            if self.last_t is None:
                self.last_t = 0.0
            t = self.last_t + 1.0 / self.freq
        self.last_t = t

        # Calculate derivative
        if self.x_filter["initialized"]:
            dx = (x - self.x_filter["s"]) * self.freq
        else:
            dx = 0.0

        # Filter the derivative
        self.dx_filter = self._ema(self.dx_filter, abs(dx), self._alpha(self.d_cutoff))

        # Calculate adaptive cutoff
        cutoff = self.min_cutoff + self.beta * self.dx_filter["val"]

        # Filter the signal
        self.x_filter = self._ema(self.x_filter, x, self._alpha(cutoff))

        return self.x_filter["val"]


# MAIN SMOOTHER CLASSES


class TransformSmoother:
    """
    Smooths the 12-D transformation parameters.
    - Translation (3 params): Kalman Filter
    - Rotation (as 4D quaternion): One-Euro Filter
    - Scale (1 param): One-Euro Filter
    """

    def __init__(self, freq=30.0, one_euro_beta=0.1, kf_R=1e-2, kf_Q=1e-3):
        # 3 Kalman filters for 3 translation parameters
        self.kf_x = KalmanSmoother1D(R=kf_R, Q=kf_Q)
        self.kf_y = KalmanSmoother1D(R=kf_R, Q=kf_Q)
        self.kf_z = KalmanSmoother1D(R=kf_R, Q=kf_Q)

        # 5 One-Euro filters for 1 scale + 4 quaternion params
        self.one_euro_s = OneEuroFilter(freq=freq, beta=one_euro_beta)
        self.one_euro_q = [
            OneEuroFilter(freq=freq, beta=one_euro_beta) for _ in range(4)
        ]

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

    def smooth(self, params):
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
        # Filter translation with Kalman Filters
        t_x_smooth = self.kf_x.smooth(t_vec[0])
        t_y_smooth = self.kf_y.smooth(t_vec[1])
        t_z_smooth = self.kf_z.smooth(t_vec[2])
        t_smooth = np.array([t_x_smooth, t_y_smooth, t_z_smooth])

        # Filter scale with One-Euro Filter
        s_smooth = self.one_euro_s.filter(scale)

        # Filter quaternion with One-Euro Filters
        q_smooth = np.array(
            [
                self.one_euro_q[0].filter(q[0]),
                self.one_euro_q[1].filter(q[1]),
                self.one_euro_q[2].filter(q[2]),
                self.one_euro_q[3].filter(q[3]),
            ]
        )

        # Re-normalize quaternion
        q_smooth /= np.linalg.norm(q_smooth) + 1e-8

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

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.05):
        self.filters = [OneEuroFilter(freq, min_cutoff, beta) for _ in range(10)]

    def smooth(self, params):
        smoothed_params = np.array(
            [self.filters[i].filter(params[i]) for i in range(10)]
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
        shape_alpha=0.05,
        expr_freq=30.0,
        expr_beta=0.05,
        trans_freq=1.0,
        trans_beta=0.0,
        kf_R=1e-2,
        kf_Q=1e-3,
    ):

        self.transform_smoother = TransformSmoother(
            freq=trans_freq, one_euro_beta=trans_beta, kf_R=kf_R, kf_Q=kf_Q
        )
        self.shape_smoother = ShapeSmoother(alpha=shape_alpha)
        self.expression_smoother = ExpressionSmoother(freq=expr_freq, beta=expr_beta)

    def smooth(self, params):
        # 1. Split the 62-D vector (as per paper)
        pose_params = params[:12]  # 12 transform params
        shape_params = params[12:52]  # 40 shape params
        exp_params = params[52:]  # 10 expression params

        # 2. Smooth each part individually
        smoothed_pose = self.transform_smoother.smooth(pose_params)
        smoothed_shape = self.shape_smoother.smooth(shape_params)
        smoothed_exp = self.expression_smoother.smooth(exp_params)

        # 3. Recombine and return
        return np.concatenate([smoothed_pose, smoothed_shape, smoothed_exp])
