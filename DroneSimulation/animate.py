"""Render the flight as a GIF: true drone (with tilt), EKF estimate, 2-sigma uncertainty set."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from sets import Ellipsoid
from scenario import run_scenario, START, GOAL, OBSTACLES, DRONE_RADIUS

STRIDE = 2          # draw every 2nd sim step
model, hist = run_scenario()
d = model.dim
has_tilt = model.nx > 2 * d
frames = list(range(0, len(hist.t), STRIDE)) + [len(hist.t) - 1]

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.set_xlim(-1, 12); ax.set_ylim(-1, 8.5); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x [m]"); ax.set_ylabel("z (up) [m]" if has_tilt else "y [m]")
ax.plot(*START, "go", label="start"); ax.plot(*GOAL, "r*", ms=14, label="goal")
for o in OBSTACLES:
    ax.add_patch(Circle(o.center, o.radius, color="tab:red", alpha=0.5))

trail, = ax.plot([], [], color="tab:blue", lw=1.5, label="true path")
est_dot, = ax.plot([], [], "x", color="tab:orange", label="EKF estimate")
ellipse, = ax.plot([], [], color="tab:orange", lw=1)
drone = Circle(START, DRONE_RADIUS, color="tab:blue", alpha=0.4, zorder=5); ax.add_patch(drone)
body, = ax.plot([], [], color="k", lw=3, zorder=6)            # rotor bar, shows tilt
title = ax.set_title("")
ax.legend(loc="upper left", fontsize=8)


def draw(i):
    p = hist.x_true[i, :d]
    drone.center = p
    trail.set_data(*hist.x_true[:i + 1, :d].T)
    est_dot.set_data([hist.x_est[i, 0]], [hist.x_est[i, 1]])
    ellipse.set_data(*Ellipsoid(hist.x_est[i, :d], hist.P[i][:d, :d]).boundary_2d().T)
    if has_tilt:
        th = hist.x_true[i, 2 * d]
        c = 0.5 * np.array([np.cos(th), np.sin(th)])
        body.set_data([p[0] - c[0], p[0] + c[0]], [p[1] - c[1], p[1] + c[1]])
    title.set_text(f"t = {hist.t[i]:.1f} s   dist to goal = {np.linalg.norm(p - GOAL):.2f} m")


FuncAnimation(fig, draw, frames=frames).save("drone.gif", writer=PillowWriter(fps=20))
print("saved drone.gif")
