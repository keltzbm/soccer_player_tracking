"""
visualization.py
----------------
Two figures showing L0 raw GPS vs L1 Kalman filter output.

  1. Pitch snapshot  — all player positions at midpoint timestep
  2. Centroid        — raw vs filtered team centroid over time

Outputs are saved to the project root outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Save to project root outputs/ regardless of where script is run from
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUTS, exist_ok=True)

BG    = '#111827'
WHITE = '#FFFFFF'
GREEN = '#2d6a2d'

ROLE_COLORS = {
    'GK':  '#FFD700',
    'LB':  '#4daf4a', 'CB1': '#4daf4a', 'CB2': '#4daf4a', 'RB':  '#4daf4a',
    'LCM': '#377eb8', 'CM':  '#377eb8', 'RCM': '#377eb8',
    'LW':  '#ff7f00', 'ST':  '#e41a1c', 'RW':  '#ff7f00',
}


def draw_pitch(ax):
    ax.set_facecolor(GREEN)
    for coords in [
        ([0, 105, 105, 0, 0],       [0, 0, 68, 68, 0]),
        ([52.5, 52.5],               [0, 68]),
        ([0, 16.5, 16.5, 0],         [13.84, 13.84, 54.16, 54.16]),
        ([105, 88.5, 88.5, 105],     [13.84, 13.84, 54.16, 54.16]),
    ]:
        ax.plot(coords[0], coords[1], color=WHITE, lw=1.2, alpha=0.6)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(52.5 + 9.15 * np.cos(th), 34 + 9.15 * np.sin(th),
            color=WHITE, lw=1.2, alpha=0.6)
    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 71)
    ax.set_aspect('equal')
    ax.axis('off')


def cov_ellipse(ax, mean, cov, n_std=2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle, **kw))


def snapshot(positions, measurements, filtered, team, step=None):
    """Side-by-side L0 raw GPS vs L1 Kalman filter at one timestep."""
    if step is None:
        step = positions.shape[1] // 2

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle(
        f'L0 Raw GPS vs L1 Kalman Filter  |  t = {step * 0.1:.0f}s  |  4-3-3',
        color=WHITE, fontsize=11)

    for ax, title in [(ax_l, 'L0 — Raw GPS'), (ax_r, 'L1 — Kalman Filter')]:
        draw_pitch(ax)
        ax.set_title(title, color=WHITE, fontsize=10, pad=6)

    for i, player in enumerate(team.players):
        col  = ROLE_COLORS.get(player.name, WHITE)
        meas = measurements[i, step]
        filt = filtered[i]['xs'][step, :2]
        P    = filtered[i]['Ps'][step, :2, :2]

        # Raw panel
        ax_l.scatter(*meas, s=90, c=col, edgecolors=WHITE,
                     linewidths=0.8, zorder=4)
        ax_l.text(meas[0], meas[1] + 2.5, player.name,
                  ha='center', color=col, fontsize=7, fontweight='bold')

        # Filtered panel — with covariance ellipse
        cov_ellipse(ax_r, filt, P, alpha=0.15, color=col, zorder=2)
        ax_r.scatter(*filt, s=90, c=col, edgecolors=WHITE,
                     linewidths=0.8, zorder=4)
        ax_r.text(filt[0], filt[1] + 2.5, player.name,
                  ha='center', color=col, fontsize=7, fontweight='bold')

    ax_r.text(52.5, -2, '2σ covariance ellipses',
              ha='center', color='#aaa', fontsize=7.5)

    out = os.path.join(OUTPUTS, '01_snapshot.png')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def centroid(positions, measurements, filtered):
    """Raw vs filtered team centroid over time."""
    from estimation import raw_centroid, filtered_centroid

    t         = np.arange(positions.shape[1]) * 0.1
    true_cent = positions.mean(axis=0)
    raw_cent  = raw_centroid(measurements)
    filt_cent = filtered_centroid(filtered)

    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(12, 6),
                                      sharex=True, facecolor=BG)
    fig.suptitle('Team Centroid  |  Raw GPS vs Kalman Filter',
                 color=WHITE, fontsize=11)

    for ax, dim, label in [(ax_x, 0, 'x [m]'), (ax_y, 1, 'y [m]')]:
        ax.set_facecolor('#1a2035')
        ax.plot(t, raw_cent[:, dim],  color='#888', lw=0.9,
                alpha=0.8, label='L0 raw')
        ax.plot(t, true_cent[:, dim], color=WHITE, lw=1.2,
                alpha=0.5, ls='--', label='True')
        ax.plot(t, filt_cent[:, dim], color='#4daf4a', lw=1.8,
                label='L1 filtered')
        ax.set_ylabel(label, color=WHITE)
        ax.tick_params(colors=WHITE)
        ax.grid(alpha=0.15, color=WHITE)
        for sp in ax.spines.values():
            sp.set_color('#444')

    ax_x.legend(fontsize=8.5, facecolor=BG, edgecolor=WHITE,
                labelcolor=WHITE, loc='upper right')
    ax_y.set_xlabel('Time [s]', color=WHITE)

    out = os.path.join(OUTPUTS, '02_centroid.png')
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
