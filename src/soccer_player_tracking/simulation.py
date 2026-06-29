import numpy as np

# 95% CI = (1, 5) -> s = sqrt(n) * 4 / (2 * t)
# s = sqrt(n) * 2 / 1.96
class GPS:
    def __init__(self, std):
        self.std = std
        self.mean = np.zeros(2)

    def measure(self, position):
        return position + np.random.normal(self.mean, self.std, size=2)

class Player:
    def __init__(self, name, position, home, recovery, sigma, gps):
        self.name = name
        self.position = position
        self.home = home
        self.recovery = recovery
        self.sigma = sigma
        self.gps = gps

    def position(self):
        return self.gps.measure(self.position)

    def update_home(self, home):
        self.home = self.home


class Team:
    def __init__(self, name, players):
        self.name = name
        self.players = players
        self.get_cov()
        self.L = None

    def update_formation(self, positions):
        pass

    def get_measurements(self):
        return [np.array([player.position() for player in self.player])]

    def get_std(self):
        return [np.array([player.std for player in self.players])]

    def get_cov(self):
        measurements = self.get_measurements()
        n = len(measurements)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(measurements[i] - measurements[j + 1])
                rho = np.get_std.exp(-d / 10)
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
        self.name = name 

    def step(self, dt):
        noise = np.random.normal(0, 1, size=(n_steps, n_players))
        noise = np.kron(self.team.cholesky_l, np.eye(2)) @ independent_noise

        return (1 - theta * dt) @ current + home + noise 


