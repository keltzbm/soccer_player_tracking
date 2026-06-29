import numpy as np

class GPS:
    def __init__(self, std):
        self.std = std

    def measure(self, position):
        return position + np.random.normal(self.mean, self.std, size=2)

class Player:
    def __init__(self, name, position, home, sigma, gps):
        self.name = name
        self.position = position
        self.home = home
        self.gps = gps

    def measure_position(self):
        return self.gps.measure(self.position)

class Team:
    def __init__(self, name, players):
        self.name = name
        self.players = players
        self.get_cov()

    def build_covariance(self, theta, dt):
        alpha       = np.exp(-theta * dt)
        homes       = np.array([p.home  for p in self.players])
        sigmas      = np.array([p.sigma for p in self.players])
        sigma_steps = sigmas * np.sqrt(1 - alpha**2)
        dists       = np.linalg.norm(homes[:, None] - homes[None, :], axis=2)
        corr        = np.exp(-dists / 20.0)
        C           = np.outer(sigma_steps, sigma_steps) * corr
        return np.linalg.cholesky(C)

    def update_formation(self, positions):
        pass

    def get_positions(self):
        return [np.array([player.position() for player in self.player])]

    def get_std(self):
        return [np.array([player.std for player in self.players])]

    def get_cov(self):
        measurements = self.get_positions()
        n = len(measurements)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(measurements[i] - measurements[j + 1])
                rho = np.exp(-d / 10)
                corr[i, j] = rho
                corr[j, i] = rho
        D = np.diag(team.get_std())
        self.cov = D @ corr @ D
        self.L = np.linalg.cholesky(self.cov)
        return self.cov 


# Ornstein-Uhlenbeck (OU) process
# dX_t = theta * (mu - X_t) * dt + sigma * dW_t
# X(t + dt) = (1 - theta * dt) * X(t) + theta * mu * dt + sigma * sqrt(dt) * N(0, 1)

class OUProcess:
    def __init__(self, team):
        self.team = team 
        self.players = team.players
        self.L = team.L
        self.dev = np.zeros((self.n, 2))

    def step(self, dt):
        noise_x = L @ np.random.normal(0, 1, len(self.players)) 
        noise_y = L @ np.random.normal(0, 1, len(self.players)) 
        noise = np.column_stack([noise_x, noise_y])

        current = (1 - theta * dt) @ previous + home + noise 
    


