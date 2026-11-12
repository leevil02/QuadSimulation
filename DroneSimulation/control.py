"""CONTROL: decides HOW to track the reference. Output = actuator command."""
import numpy as np
from scipy.linalg import solve_discrete_are


class LQRController:
    """u = u_hover - K (x - x_ref), K from the discrete Riccati equation on the model
    linearized at hover. Works for any Model (linear or nonlinear)."""

    def __init__(self, model, q_pos=10.0, q_vel=4.0, q_extra=1.0, r=1.0):
        self.model = model
        A, B = model.linearize(np.zeros(model.nx), model.hover_input)
        d = model.dim
        Q = np.diag([q_pos] * d + [q_vel] * d + [q_extra] * (model.nx - 2 * d))
        R = r * np.eye(model.nu)
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    def compute(self, x_est, ref_pos, ref_vel):
        x_ref = self.model.reference_state(ref_pos, ref_vel)
        u = self.model.hover_input - self.K @ (x_est - x_ref)
        return np.clip(u, self.model.u_min, self.model.u_max)      # actuator limits
