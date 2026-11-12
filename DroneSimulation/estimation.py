"""State estimation: GPS-like position sensor + Extended Kalman Filter."""
import numpy as np


class PositionSensor:
    """Noisy GPS / motion-capture position. Only position is measured; velocity and tilt
    must be inferred by the filter through the dynamics."""

    def __init__(self, dim, nx, noise_std=0.1):
        self.dim, self.nx, self.noise_std = dim, nx, noise_std
        self.R = noise_std**2 * np.eye(dim)

    def h(self, x):
        return x[:self.dim]

    def jacobian(self, x):
        H = np.zeros((self.dim, self.nx))
        H[:, :self.dim] = np.eye(self.dim)
        return H

    def measure(self, x_true, rng):
        return self.h(x_true) + rng.normal(0, self.noise_std, self.dim)


class EKF:
    """Predict with the (possibly nonlinear) model, correct with the sensor.
    The EKF linearizes the model around the current estimate: that is what makes it
    work for the quadrotor (for a linear model it reduces to a plain Kalman filter)."""

    def __init__(self, model, sensor, x0, P0):
        self.model, self.sensor = model, sensor
        self.x, self.P = np.asarray(x0, float), np.asarray(P0, float)

    def predict(self, u):
        F = self.model.jacobian(self.x, u)
        self.x = self.model.step(self.x, u)               # noise-free prediction
        self.P = F @ self.P @ F.T + self.model.Q

    def update(self, z):
        H = self.sensor.jacobian(self.x)
        y = z - self.sensor.h(self.x)                     # innovation
        S = H @ self.P @ H.T + self.sensor.R
        K = self.P @ H.T @ np.linalg.inv(S)               # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.sensor.R @ K.T   # Joseph form for numerical stability
