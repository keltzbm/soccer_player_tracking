import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_player_trajectory(history, player_name="Player_10"):
    """
    Plots a 2D bird's-eye view tracking dashboard for a single agent.
    Compares Ground Truth vs Noisy Hardware Measurements vs Filtered Estimation.
    """
    if player_name not in history or not history[player_name]["true"]:
        print(f"Error: Logging data for {player_name} is empty or missing.")
        return

    true_pts = np.array(history[player_name]["true"])
    gps_pts = np.array(history[player_name]["gps"])
    filt_pts = np.array(history[player_name]["filtered"])

    # Calculate Performance Assessment Metrics (RMSE)
    rmse_raw = np.sqrt(np.mean(np.sum((gps_pts - true_pts)**2, axis=1)))
    rmse_filt = np.sqrt(np.mean(np.sum((filt_pts - true_pts)**2, axis=1)))

    plt.figure(figsize=(10, 6))
    plt.plot(true_pts[:, 0], true_pts[:, 1], 'k-', label="Ground Truth (True State)", linewidth=2)
    plt.scatter(gps_pts[:, 0], gps_pts[:, 1], c='r', marker='x', alpha=0.4, label="Raw Noisy GPS")
    plt.plot(filt_pts[:, 0], filt_pts[:, 1], 'b--', label="State Estimator (Kalman Filter)", linewidth=1.5)

    plt.title(f"Multi-Agent Kinematic Tracking Profile — {player_name}\n"
              f"Raw Sensor RMSE: {rmse_raw:.2f}m | Filtered Estimate RMSE: {rmse_filt:.2f}m")
    plt.xlabel("Pitch Length (meters)")
    plt.ylabel("Pitch Width (meters)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{player_name}_trajectory_analysis.png", dpi=300)
    plt.show()


def plot_centroid_smoothing(history, player_names):
    """
    Computes and plots the aggregate collective team centroid over time.
    Demonstrates macroscopic smoothing characteristics via node fusion.
    """
    n_steps = len(history[player_names[0]]["true"])
    true_centroid = np.zeros((n_steps, 2))
    gps_centroid = np.zeros((n_steps, 2))
    filt_centroid = np.zeros((n_steps, 2))

    for step in range(n_steps):
        for name in player_names:
            true_centroid[step] += history[name]["true"][step]
            gps_centroid[step] += history[name]["gps"][step]
            filt_centroid[step] += history[name]["filtered"][step]
        
        true_centroid[step] /= len(player_names)
        gps_centroid[step] /= len(player_names)
        filt_centroid[step] /= len(player_names)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # X-Axis Tracking Performance
    axes[0].plot(true_centroid[:, 0], 'k-', label="True Centroid", linewidth=2)
    axes[0].plot(gps_centroid[:, 0], 'r-', alpha=0.3, label="Raw GPS Centroid")
    axes[0].plot(filt_centroid[:, 0], 'b--', label="Filtered Centroid")
    axes[0].set_ylabel("X-Coordinate (meters)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Macroscopic Team Centroid Tracking Optimization")
    axes[0].legend(loc="upper right")

    # Y-Axis Tracking Performance
    axes[1].plot(true_centroid[:, 1], 'k-', label="True Centroid", linewidth=2)
    axes[1].plot(gps_centroid[:, 1], 'r-', alpha=0.3, label="Raw GPS Centroid")
    axes[1].plot(filt_centroid[:, 1], 'b--', label="Filtered Centroid")
    axes[1].set_ylabel("Y-Coordinate (meters)")
    axes[1].set_xlabel("Simulation Increments (Time Steps)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("team_centroid_smoothing.png", dpi=300)
    plt.show()


def plot_covariance_matrix(team_object):
    """
    Renders a matrix heatmap of the dynamic team covariance structure.
    Verifies decoupled diagonal entries and tactical off-diagonal blocks.
    """
    if not hasattr(team_object, 'cov') or team_object.cov is None:
        print("Verification Error: Target Team object contains no active covariance states.")
        return

    plt.figure(figsize=(10, 8))
    # Standardizing display ticks based on player names
    labels = [p.name for p in team_object.players]
    
    sns.heatmap(team_object.cov, annot=False, cmap="coolwarm", cbar=True,
                xticklabels=labels, yticklabels=labels)
    
    plt.title("Structural Verification Heatmap: Team Covariance Matrix ($\Sigma$)\n"
              "Independent Diagonal Variances ($\sigma_i^2$) & Coupled Spatial Off-Diagonals")
    plt.xlabel("Team Node Index")
    plt.ylabel("Team Node Index")
    plt.tight_layout()
    plt.savefig("team_covariance_matrix.png", dpi=300)
    plt.show()
