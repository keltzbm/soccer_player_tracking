import numpy as np


class KalmanFilter:
    """
    2D constant-velocity Kalman filter.

    State: x = [px, py, vx, vy]

    Parameters
    ----------
    dt      : float  timestep [s]
    sigma_q : float  process noise — unmodeled accelerations [m/s²]
    sigma_r : float  measurement noise — GPS vest accuracy [m]
    """

    def __init__(self, dt, sigma_q, sigma_r):
        # State transition — constant velocity
        self.F = np.array([[1, 0, dt,  0],
                           [0, 1,  0, dt],
                           [0, 0,  1,  0],
                           [0, 0,  0,  1]], dtype=float)

        # Observation — position only, velocity is latent
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Process noise — DWNA model (Bar-Shalom et al., 2001)
        q = sigma_q ** 2
        self.Q = q * np.array([
            [dt**4/4,       0, dt**3/2,       0],
            [      0, dt**4/4,       0, dt**3/2],
            [dt**3/2,       0,   dt**2,       0],
            [      0, dt**3/2,       0,   dt**2]])

        # Measurement noise — per-player GPS sigma
        self.R = (sigma_r ** 2) * np.eye(2)

    def run(self, measurements):
        """
        Forward Kalman filter pass.

        Parameters
        ----------
        measurements : (N, 2)  noisy GPS positions

        Returns
        -------
        xs : (N, 4)    filtered states [px, py, vx, vy]
        Ps : (N, 4, 4) state covariances
        """
        N  = len(measurements)
        xs = np.zeros((N, 4))
        Ps = np.zeros((N, 4, 4))

        # Initialise — estimate velocity from first two measurements
        if N > 1:
            v0 = (measurements[1] - measurements[0]) / 0.1
        else:
            v0 = np.zeros(2)

        x = np.array([measurements[0, 0], measurements[0, 1], v0[0], v0[1]])
        P = np.diag([1., 1., 25., 25.])

        for k, z in enumerate(measurements):
            # Predict
            x_p = self.F @ x
            P_p = self.F @ P @ self.F.T + self.Q

            # Update
            y   = z - self.H @ x_p
            S   = self.H @ P_p @ self.H.T + self.R
            K   = P_p @ self.H.T @ np.linalg.inv(S)
            x   = x_p + K @ y
            P   = (np.eye(4) - K @ self.H) @ P_p

            xs[k] = x
            Ps[k] = P

        return xs, Ps


def run_all(measurements, team, dt=0.1, sigma_q=3.0):
    """
    Run one Kalman filter per player.

    Parameters
    ----------
    measurements : (n_players, N, 2)  noisy GPS positions
    team         : Team  — used to get per-player GPS sigma for R
    sigma_q      : float — process noise [m/s²]

    Returns
    -------
    list of dicts  [{'xs': (N, 4), 'Ps': (N, 4, 4)}, ...]
    """
    filtered = []
    for i, player in enumerate(team.players):
        kf     = KalmanFilter(dt, sigma_q, player.gps.std)
        xs, Ps = kf.run(measurements[i])
        filtered.append({'xs': xs, 'Ps': Ps})
    return filtered


# ── Metrics ───────────────────────────────────────────────────────────────────

def raw_centroid(measurements):
    """Mean of raw GPS positions across all players. Shape: (N, 2)."""
    return measurements.mean(axis=0)


def filtered_centroid(filtered):
    """Mean of filtered positions across all players. Shape: (N, 2)."""
    positions = np.stack([f['xs'][:, :2] for f in filtered])
    return positions.mean(axis=0)


def rmse(positions, measurements, filtered):
    results = []
    for i, f in enumerate(filtered):
        true = positions[i]
        raw  = measurements[i]
        filt = f['xs'][:, :2]
        results.append({
            'raw':      float(np.sqrt(((raw  - true)**2).mean())),
            'filtered': float(np.sqrt(((filt - true)**2).mean())),
        })
    return results


def centroid_rmse(positions, measurements, filtered):
    true_cent = positions.mean(axis=0)
    raw_cent  = raw_centroid(measurements)
    filt_cent = filtered_centroid(filtered)
    return {
        'raw':      float(np.sqrt(((raw_cent  - true_cent)**2).mean())),
        'filtered': float(np.sqrt(((filt_cent - true_cent)**2).mean())),
    }


def team_covariance(filtered):
    X = np.stack([f['xs'][:, 0] for f in filtered])
    return np.cov(X)


def player_centroid_covariance(filtered):
    cent = filtered_centroid(filtered)
    covs = []
    for f in filtered:
        pos   = f['xs'][:, :2]
        joint = np.column_stack([pos, cent])
        C     = np.cov(joint.T)
        covs.append(C[:2, 2:])
    return covs


def team_spread(positions):
    N      = positions.shape[1]
    spread = np.zeros(N)
    for t in range(N):
        pts   = positions[:, t, :]
        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
        spread[t] = dists[np.triu_indices(len(pts), k=1)].mean()
    return spread
