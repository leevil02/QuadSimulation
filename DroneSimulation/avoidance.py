"""AVOIDANCE (safety filter): modifies the command so the drone never enters unsafe sets.
Sits between controller and plant. Later: CLF-CBF-QP goes here."""
import numpy as np


class SafetyFilter:
    def filter(self, x_est, u_nominal, obstacles):
        """Return the safe command closest to u_nominal."""
        raise NotImplementedError


class PassThrough(SafetyFilter):
    """No obstacles yet: trust the nominal controller."""
    def filter(self, x_est, u_nominal, obstacles):
        return u_nominal
