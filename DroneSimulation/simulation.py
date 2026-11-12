"""The closed loop. Reading this file should explain the whole architecture."""
from dataclasses import dataclass, field
import numpy as np
from sets import Ball


@dataclass
class History:
    t: list = field(default_factory=list)
    x_true: list = field(default_factory=list)
    x_est: list = field(default_factory=list)
    P: list = field(default_factory=list)
    u: list = field(default_factory=list)


def run(model, sensor, ekf, planner, controller, safety, x0, goal,
        drone_radius=0.3, obstacles=(), goal_tol=0.1, max_time=120.0, seed=0):
    """Runs until the drone is within goal_tol of the goal.
    max_time is only a safety cap for a failed run"""
    rng = np.random.default_rng(seed)
    x_true = np.asarray(x0, float)
    hist = History()

    reached = False
    for k in range(int(max_time / model.dt)):
        # 1. SENSE + ESTIMATE
        ekf.update(sensor.measure(x_true, rng))
        x_est = ekf.x.copy()

        # 2. PLAN      -> where to go
        ref_pos, ref_vel = planner.plan(model.position(x_est), goal, obstacles)
        # 3. CONTROL   -> how to get there
        u_nom = controller.compute(x_est, ref_pos, ref_vel)
        # 4. AVOID     -> keep it safe
        u = safety.filter(x_est, u_nom, obstacles)

        # 5. ACT: true system moves, estimator predicts
        x_true = model.step(x_true, u, rng)
        ekf.predict(u)

        # 6. SAFETY CHECK on ground truth, using sets
        drone = Ball(model.position(x_true), drone_radius)
        if any(drone.distance_to(o) < 0 for o in obstacles):
            print(f"COLLISION at t={k * model.dt:.2f}s")

        hist.t.append(k * model.dt); hist.x_true.append(x_true.copy())
        hist.x_est.append(x_est); hist.P.append(ekf.P.copy()); hist.u.append(u)

        # 7. TERMINATE when the true position is close enough to the goal
        if np.linalg.norm(model.position(x_true) - goal) < goal_tol:
            reached = True
            break
    print(f"{'Goal reached' if reached else 'GOAL NOT REACHED'} at t={hist.t[-1]:.2f}s")

    for name in ("x_true", "x_est", "P", "u", "t"):
        setattr(hist, name, np.array(getattr(hist, name)))
    return hist
