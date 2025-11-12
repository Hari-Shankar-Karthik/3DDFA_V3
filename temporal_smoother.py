import numpy as np


class SimpleKalman1D:
    """
    A simple 1D Kalman Filter implementation.

    It assumes a constant position model (F=1) and is used to
    smooth a single scalar value.
    """

    def __init__(self, R=0.1, Q=0.1):
        """
        :param R: Measurement Noise Covariance
        :param Q: Process Noise Covariance
        """
        self.x = 0.0  # Initial state (position)
        self.P = 1.0  # Initial state covariance
        self.F = 1.0  # State transition (constant position)
        self.H = 1.0  # Measurement function
        self.R = R  # Measurement noise
        self.Q = Q  # Process noise

    def predict(self):
        # Predict the next state
        self.x = self.x * self.F
        self.P = (self.F * self.P * self.F) + self.Q
        return self.x

    def update(self, z):
        # Update the state based on a new measurement z
        y = z - self.x * self.H  # Measurement residual
        S = (self.H * self.P * self.H) + self.R  # Residual covariance
        K = self.P * self.H * (1.0 / S)  # Kalman gain

        self.x = self.x + K * y
        self.P = (1 - K * self.H) * self.P
        return self.x


class TransformSmoother:
    """
    Smoothes the 12-dimensional transformation (pose) parameters.
    """

    def __init__(self, method="ewma", **kwargs):
        self.method = method
        self.last_params = None

        if self.method == "ewma":
            # Default alpha for EWMA. 0.1=very smooth, 0.9=very responsive
            self.alpha = kwargs.get("alpha", 0.4)

        elif self.method == "kalman":
            # Measurement noise: how much to trust the new measurement
            R = kwargs.get("R", 0.01)
            # Process noise: how much the state can change
            Q = kwargs.get("Q", 0.1)
            # We need 12 independent 1D Kalman filters
            self.kalman_filters = [SimpleKalman1D(R=R, Q=Q) for _ in range(12)]
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def smooth(self, params):
        """Applies smoothing to the 12-D pose parameter vector."""
        if params.shape[0] != 12:
            raise ValueError(
                f"TransformSmoother expects 12 parameters, got {params.shape[0]}"
            )

        if self.method == "ewma":
            return self._ewma_smooth(params)
        elif self.method == "kalman":
            return self._kalman_smooth(params)

    def _ewma_smooth(self, params):
        if self.last_params is None:
            self.last_params = params
            return params

        smoothed = self.alpha * params + (1 - self.alpha) * self.last_params
        self.last_params = smoothed
        return smoothed

    def _kalman_smooth(self, params):
        smoothed_params = np.zeros(12)
        for i in range(12):
            kf = self.kalman_filters[i]
            kf.predict()
            kf.update(params[i])
            smoothed_params[i] = kf.x

        # We need to set self.last_params for the Kalman filter
        # to handle the first frame issue in TemporalSmoother
        if self.last_params is None:
            self.last_params = smoothed_params

        return smoothed_params


class ShapeSmoother:
    """Skeletal class for smoothing the 40-D shape parameters."""

    def __init__(self, **kwargs):
        # Add any shape-specific initialization here
        pass

    def smooth(self, params):
        """Pass-through for now."""
        if params.shape[0] != 40:
            raise ValueError(
                f"ShapeSmoother expects 40 parameters, got {params.shape[0]}"
            )
        return params


class ExpressionSmoother:
    """Skeletal class for smoothing the 10-D expression parameters."""

    def __init__(self, **kwargs):
        # Add any expression-specific initialization here
        pass

    def smooth(self, params):
        """Pass-through for now."""
        if params.shape[0] != 10:
            raise ValueError(
                f"ExpressionSmoother expects 10 parameters, got {params.shape[0]}"
            )
        return params


class TemporalSmoother:
    """
    Main class to coordinate smoothing of all 3DMM parameters.
    """

    def __init__(self, transform_method="ewma", **kwargs):
        """
        Initializes all sub-smoothers.
        :param transform_method: 'ewma' or 'kalman'
        :param **kwargs: Arguments passed to the smoothers
                         (e.g., 'alpha' for ewma, 'R' and 'Q' for kalman)
        """
        self.transform_smoother = TransformSmoother(method=transform_method, **kwargs)
        self.shape_smoother = ShapeSmoother(**kwargs)
        self.expression_smoother = ExpressionSmoother(**kwargs)

    def smooth(self, params):
        """
        Takes the 62-D param vector, splits it, smooths each part,
        and recombines.
        """
        if params.shape[0] != 62:
            raise ValueError(
                f"TemporalSmoother expects 62 parameters, got {params.shape[0]}"
            )

        # 1. Split the 62-D vector
        pose_params = params[:12]
        shape_params = params[12:52]
        exp_params = params[52:]

        # 2. Smooth each part individually
        smoothed_pose = self.transform_smoother.smooth(pose_params)
        smoothed_shape = self.shape_smoother.smooth(shape_params)
        smoothed_exp = self.expression_smoother.smooth(exp_params)

        # 3. Recombine and return
        return np.concatenate([smoothed_pose, smoothed_shape, smoothed_exp])
