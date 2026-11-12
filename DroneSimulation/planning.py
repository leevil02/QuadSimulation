"""PLANNING: decides WHERE to go. Output = reference (position, velocity). No actuator commands."""
import numpy as np


class Planner:
    """Interface. Swap in RRT/RRT* later without touching anything else."""
    def plan(self, p_est, goal, obstacles):
        """Return (ref_position, ref_velocity)."""
        raise NotImplementedError


class StraightLinePlanner(Planner):
    """Head directly to the goal; slow down on approach. Ignores obstacles."""

    def __init__(self, v_max=2.0, lookahead=1.0, slowdown_gain=1.0):
        self.v_max, self.lookahead, self.k = v_max, lookahead, slowdown_gain

    def plan(self, p_est, goal, obstacles):
        to_goal = goal - p_est
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return goal, np.zeros_like(goal)
        direction = to_goal / dist
        speed = min(self.v_max, self.k * dist)
        ref_pos = goal if dist < self.lookahead else p_est + self.lookahead * direction
        return ref_pos, speed * direction
