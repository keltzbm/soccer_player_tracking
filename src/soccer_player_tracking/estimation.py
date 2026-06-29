import numpy as np


class KalmanFilter:
    pass


# ── Metrics ───────────────────────────────────────────────────────────────────
def raw_centroid(measurements):
    """Mean of raw GPS positions across all players. Shape: (N, 2)."""
    return measurements.mean(axis=0)


def filtered_centroid(filtered):
    """Mean of filtered positions across all players. Shape: (N, 2)."""
    positions = np.stack([f["xs"][:, :2] for f in filtered])
    return positions.mean(axis=0)


def rmse(positions, measurements, filtered):
    """
    Per-player RMSE for raw GPS and Kalman filter.

    Returns
    -------
    list of dicts  [{'name': str, 'raw': float, 'filtered': float}, ...]
    """
    results = []
    for i, f in enumerate(filtered):
        true = positions[i]
        raw = measurements[i]
        filt = f["xs"][:, :2]
        results.append(
            {
                "raw": float(np.sqrt(((raw - true) ** 2).mean())),
                "filtered": float(np.sqrt(((filt - true) ** 2).mean())),
            }
        )
    return results


def centroid_rmse(positions, measurements, filtered):
    """
    RMSE of raw and filtered centroids against true centroid.

    Returns
    -------
    dict  {'raw': float, 'filtered': float}
    """
    true_cent = positions.mean(axis=0)
    raw_cent = raw_centroid(measurements)
    filt_cent = filtered_centroid(filtered)
    return {
        "raw": float(np.sqrt(((raw_cent - true_cent) ** 2).mean())),
        "filtered": float(np.sqrt(((filt_cent - true_cent) ** 2).mean())),
    }


def team_covariance(filtered):
    """
    11x11 covariance matrix of filtered x-positions over time.
    Entry (i,j) describes how much players i and j move together.

    Returns
    -------
    C     : (11, 11) covariance matrix
    """
    X = np.stack([f["xs"][:, 0] for f in filtered])  # (11, N)
    return np.cov(X)


def player_centroid_covariance(filtered):
    """
    Covariance between each player's filtered position and the team centroid.
    Trace of the (2,2) block describes how strongly the player moves with the team.

    Returns
    -------
    list of (2, 2) covariance matrices, one per player
    """
    cent = filtered_centroid(filtered)
    covs = []
    for f in filtered:
        pos = f["xs"][:, :2]
        joint = np.column_stack([pos, cent])  # (N, 4)
        C = np.cov(joint.T)  # (4, 4)
        covs.append(C[:2, 2:])  # (2, 2) off-diagonal block
    return covs


def team_spread(positions):
    """
    Mean pairwise distance between players at each timestep.
    Describes how spread out the team is over time.

    Returns
    -------
    spread : (N,) mean pairwise distance [m]
    """
    N = positions.shape[1]
    spread = np.zeros(N)
    for t in range(N):
        pts = positions[:, t, :]  # (11, 2)
        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)  # (11, 11)
        # upper triangle only, excluding diagonal
        spread[t] = dists[np.triu_indices(len(pts), k=1)].mean()
    return spread
