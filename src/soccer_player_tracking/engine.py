import numpy as np


class GPS:
    def __init__(self, std):
        self.std = std

    def measure(self, position):
        return position + np.random.normal(0, self.std, size=2)


class Player:
    def __init__(self, name, position, home, sigma, gps):
        self.name     = name
        self.position = np.array(position, dtype=float)
        self.home     = np.array(home, dtype=float)
        self.sigma    = sigma
        self.gps      = gps

    def measure_position(self):
        return self.gps.measure(self.position)


class Team:
    def __init__(self, name, players):
        self.name = name
        self.players = players

    def build_covariance(self, theta, dt):
        alpha = np.exp(-theta * dt)
        positions = np.array([p.position for p in self.players])
        sigmas = np.array([p.sigma for p in self.players])
        sigma_steps = sigmas * np.sqrt(1 - alpha**2)
        dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        corr = np.exp(-dists / 20.0)
        C = np.outer(sigma_steps, sigma_steps) * corr
        return np.linalg.cholesky(C)

def ou_step(positions, L, alpha, home):
    n = positions.shape[0]
    noise_x = L @ np.random.normal(0, 1, n)
    noise_y = L @ np.random.normal(0, 1, n)
    noise = np.column_stack([noise_x, noise_y])
    return (1 - theta * dt) @ positions + home + noiseimport


class GPS:
    def __init__(self, std):
        self.std = std

    def measure(self, position):
        return position + np.random.normal(0, self.std, size=2)


class Player:
    def __init__(self, name, position, home, sigma, gps):
        self.name     = name
        self.position = np.array(position, dtype=float)
        self.home     = np.array(home, dtype=float)
        self.sigma    = sigma
        self.gps      = gps

    def measure_position(self):
        return self.gps.measure(self.position)


class Team:
    def __init__(self, name, players):
        self.name = name
        self.players = players

    def build_covariance(self, theta, dt):
        """
        Build covariance matrix from current player positions.
        Called at each step so correlation reflects live spacing —
        players close together share more noise, players far apart
        deviate independently.
        """
        alpha = np.exp(-theta * dt)
        positions = np.array([p.position for p in self.players])
        sigmas = np.array([p.sigma for p in self.players])
        sigma_steps = sigmas * np.sqrt(1 - alpha**2)
        dists = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        corr = np.exp(-dists / 20.0)
        C = np.outer(sigma_steps, sigma_steps) * corr
        return np.linalg.cholesky(C)

def ou_step(positions, L, alpha, home, theta, dt):
    n = positions.shape[0]
    noise_x = L @ np.random.normal(0, 1, n)
    noise_y = L @ np.random.normal(0, 1, n)
    noise = np.column_stack([noise_x, noise_y])
    return (1 - theta * dt) @ positions + home + noise 
