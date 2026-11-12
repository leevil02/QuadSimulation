"""Static plot of the scenario."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sets import Ellipsoid
from scenario import run_scenario, START, GOAL

model, hist = run_scenario()
d = model.dim
print(f"final distance to goal: {np.linalg.norm(hist.x_true[-1, :d] - GOAL):.3f} m")
print(f"max EKF position error: {np.linalg.norm(hist.x_true[:, :d] - hist.x_est[:, :d], axis=1).max():.2f} m "
      f"(last 50 steps: {np.linalg.norm(hist.x_true[-50:, :d] - hist.x_est[-50:, :d], axis=1).max():.2f} m)")
if model.nx > 2 * d:
    print(f"max tilt: {np.degrees(np.abs(hist.x_true[:, 2 * d]).max()):.1f} deg")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(*hist.x_true[:, :d].T, label="true"); ax.plot(*hist.x_est[:, :d].T, "--", label="EKF")
for i in range(0, len(hist.t), 20):
    ax.plot(*Ellipsoid(hist.x_est[i, :d], hist.P[i][:d, :d]).boundary_2d().T, color="gray", lw=0.5)
ax.plot(*START, "go"); ax.plot(*GOAL, "r*", ms=12); ax.set_aspect("equal"); ax.legend()
plt.savefig("result.png", dpi=120)
