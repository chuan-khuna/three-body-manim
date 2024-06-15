import numpy as np

from .constants import GRAVIATIONAL_CONST as G


class Body:
    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, name: str = None):
        self.mass = float(mass)
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name
        self.G = G

    def calculate_force(self, other_body: 'Body'):
        r = other_body.position - self.position
        dist = np.linalg.norm(r)
        # dist += 1e-20  # to avoid division by zero
        force_magnitude = self.G * self.mass * other_body.mass / dist**2
        force = force_magnitude * (r / dist)
        return force

    compute_force = calculate_force

    def update(self, other_bodies: list, dt: float) -> 'Body':
        if isinstance(other_bodies, Body):
            other_bodies = [other_bodies]

        other_bodies = [body for body in other_bodies if body != self]

        force = np.sum(
            np.array([self.calculate_force(other_body) for other_body in other_bodies]), axis=0
        )
        acceleration = force / self.mass
        new_velocity = self.velocity + (acceleration * dt)
        new_position = self.position + (new_velocity * dt)

        new = Body(self.mass, new_position, new_velocity, self.name)
        return new

    def __repr__(self):
        if self.name is not None:
            return (
                f"{self.name}(mass={self.mass}, position={self.position}, velocity={self.velocity})"
            )
        return f"Body(mass={self.mass}, position={self.position}, velocity={self.velocity})"
