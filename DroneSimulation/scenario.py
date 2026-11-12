"""One place that defines the scenario, so main.py and animate.py stay in sync."""
import numpy as np
from models import DoubleIntegrator, PlanarQuadrotor
from estimation import PositionSensor, EKF
from planning import StraightLinePlanner
from control import LQRController
from avoidance import PassThrough
import simulation

MODEL = "double_integrator"      # "planar_quadrotor" or "double_integrator"
DRONE_RADIUS = 0.3
START, GOAL = np.array([0.0, 0.0]), np.array([10.0, 6.0])   # (x, z) for double_integrator, (x, z) for the quadrotor
GOAL_TOL = 0.1                  # terminate when distance to goal < this [m]
OBSTACLES = []                  # list of sets.Ball, empty for now


def run_scenario():
    model = PlanarQuadrotor() if MODEL == "planar_quadrotor" else DoubleIntegrator(dim=2)
    sensor = PositionSensor(model.dim, model.nx, noise_std=0.1)

    x0 = np.zeros(model.nx)
    x0[:model.dim] = START

    x_guess = x0.copy()
    x_guess[:2 * model.dim] += 0.5                         # deliberately wrong initial guess
    P0 = np.diag([1.0] * (2 * model.dim) + [0.1] * (model.nx - 2 * model.dim))
    ekf = EKF(model, sensor, x_guess, P0)

    hist = simulation.run(model, sensor, ekf, StraightLinePlanner(), LQRController(model),
                          PassThrough(), x0, GOAL, DRONE_RADIUS, OBSTACLES, GOAL_TOL)
    return model, hist
