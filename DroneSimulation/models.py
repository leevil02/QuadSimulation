"""Drone dynamics. Every model shares one interface so the EKF, LQR, planner and
(later) CBF code never need to know which drone they are talking to.

State layout convention:  x = [position (dim), velocity (dim), extra states...]
"""
import numpy as np


def numerical_jacobian(fun, x, eps=1e-6):
    """Central-difference Jacobian of fun at x. Lets us write only the dynamics,
    not their derivatives (swap in analytic Jacobians later if you want speed)."""
    J = np.zeros((len(fun(x)), len(x)))
    for i in range(len(x)):
        d = np.zeros(len(x)); d[i] = eps
        J[:, i] = (fun(x + d) - fun(x - d)) / (2 * eps)
    return J


class Model:
    """Base class. Subclasses define `dynamics` and a few constants."""
    dim: int            # position dimension (2 or 3)
    nx: int             # state size
    nu: int             # input size
    u_min: np.ndarray   # actuator limits
    u_max: np.ndarray
    hover_input: np.ndarray

    def __init__(self, dt=0.05, accel_noise_std=0.05):
        self.dt, self.accel_noise_std = dt, accel_noise_std
        # Disturbance (e.g. wind gust) acts as a random acceleration on the position states.
        G = np.zeros((self.nx, self.dim))
        G[:self.dim] = 0.5 * dt**2 * np.eye(self.dim)
        G[self.dim:2 * self.dim] = dt * np.eye(self.dim)
        self.Q = accel_noise_std**2 * G @ G.T          # process noise covariance for the EKF

    # --- to be implemented by each drone model -------------------------------------
    def dynamics(self, x, u, w):
        """Continuous-time x_dot = f(x, u, w).  w = disturbance acceleration (dim,)."""
        raise NotImplementedError

    # --- shared machinery ----------------------------------------------------------
    def step(self, x, u, rng=None):
        """Advance one dt with RK4. Pass rng to add a random disturbance (true physics);
        omit it for the noise-free prediction used by the EKF."""
        w = rng.normal(0, self.accel_noise_std, self.dim) if rng is not None else np.zeros(self.dim)
        f, dt = self.dynamics, self.dt
        k1 = f(x, u, w)
        k2 = f(x + 0.5 * dt * k1, u, w)
        k3 = f(x + 0.5 * dt * k2, u, w)
        k4 = f(x + dt * k3, u, w)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def jacobian(self, x, u):
        """F = d x_{k+1} / d x_k (used by the EKF prediction step)."""
        return numerical_jacobian(lambda x_: self.step(x_, u), x)

    def linearize(self, x_eq, u_eq):
        """Discrete (A, B) around an equilibrium (used by LQR)."""
        A = numerical_jacobian(lambda x_: self.step(x_, u_eq), x_eq)
        B = numerical_jacobian(lambda u_: self.step(x_eq, u_), u_eq)
        return A, B

    def position(self, x):  return x[:self.dim]
    def velocity(self, x):  return x[self.dim:2 * self.dim]

    def reference_state(self, ref_pos, ref_vel):
        """Full state we want the drone to be at (extra states, e.g. tilt, desired = 0)."""
        x_ref = np.zeros(self.nx)
        x_ref[:self.dim], x_ref[self.dim:2 * self.dim] = ref_pos, ref_vel
        return x_ref


class DoubleIntegrator(Model):
    """Linear point mass: acceleration is the input. x = [p, v], u = a."""

    def __init__(self, dim=2, dt=0.05, accel_noise_std=0.05, a_max=4.0):
        self.dim, self.nx, self.nu = dim, 2 * dim, dim
        self.u_min, self.u_max = -a_max * np.ones(dim), a_max * np.ones(dim)
        self.hover_input = np.zeros(dim)
        super().__init__(dt, accel_noise_std)

    def dynamics(self, x, u, w):
        return np.concatenate([x[self.dim:], u + w])


class PlanarQuadrotor(Model):
    """Quadrotor in the vertical x-z plane (z = up). NONLINEAR: thrust direction depends on tilt.

    x = [x, z, vx, vz, theta, omega]     theta = tilt angle (0 = level), omega = tilt rate
    u = [T, tau]                         total thrust [N], torque about the pitch axis [N m]
    """

    def __init__(self, dt=0.05, accel_noise_std=0.05, mass=1.0, inertia=0.05, g=9.81):
        self.dim, self.nx, self.nu = 2, 6, 2
        self.m, self.J, self.g = mass, inertia, g
        self.u_min = np.array([0.0, -2.0])
        self.u_max = np.array([2 * mass * g, 2.0])
        self.hover_input = np.array([mass * g, 0.0])      # thrust that cancels gravity
        super().__init__(dt, accel_noise_std)

    def dynamics(self, x, u, w):
        _, _, vx, vz, theta, omega = x
        T, tau = u
        ax = -(T / self.m) * np.sin(theta) + w[0]
        az = (T / self.m) * np.cos(theta) - self.g + w[1]
        return np.array([vx, vz, ax, az, omega, tau / self.J])
