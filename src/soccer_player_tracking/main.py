import numpy as np
from engine import GPS, Player, Team, ou_step
from estimation import run_all

DT      = 0.1
THETA   = 0.2
N_STEPS = 100

def make_team():
    specs = [
        ('GK',  [ 8, 34], 0.3),
        ('LB',  [30, 10], 1.5),
        ('CB1', [28, 26], 0.8),
        ('CB2', [28, 42], 0.8),
        ('RB',  [30, 58], 1.5),
        ('LCM', [50, 18], 1.8),
        ('CM',  [50, 34], 1.8),
        ('RCM', [50, 50], 1.8),
        ('LW',  [72, 10], 3.0),
        ('ST',  [74, 34], 3.5),
        ('RW',  [72, 58], 3.0),
    ]
    players = [Player(name, home, home, sigma, GPS(np.random.uniform(1, 5)))
               for name, home, sigma in specs]
    return Team('NOR', players)


def main():
    rng   = np.random.default_rng(42)
    team  = make_team()
    n     = len(team.players)
    alpha = np.exp(-THETA * DT)
    devs  = np.zeros((n, 2))

    positions    = np.zeros((n, N_STEPS, 2))
    measurements = np.zeros((n, N_STEPS, 2))

    for step in range(N_STEPS):
        L     = team.build_covariance(THETA, DT)
        noise = team.sample_noise(L, rng)
        devs  = ou_step(devs, noise, alpha)

        for i, player in enumerate(team.players):
            player.position       = np.clip(player.home + devs[i], [1,1], [104,67])
            positions[i, step]    = player.position
            measurements[i, step] = player.measure_position()

    filtered = run_all(measurements, team)

    print(f"positions    shape: {positions.shape}")
    print(f"measurements shape: {measurements.shape}")
    print(f"filtered     count: {len(filtered)} players")
    print(f"GK x range:  {positions[0,:,0].min():.2f} – {positions[0,:,0].max():.2f}")
    print(f"ST x range:  {positions[8,:,0].min():.2f} – {positions[8,:,0].max():.2f}")

    # Quick RMSE check
    for i, player in enumerate(team.players):
        true = positions[i]
        raw  = measurements[i]
        filt = filtered[i]['xs'][:, :2]
        rmse_raw  = np.sqrt(((raw  - true)**2).mean())
        rmse_filt = np.sqrt(((filt - true)**2).mean())
        print(f"{player.name:<5}  raw: {rmse_raw:.2f}m  filtered: {rmse_filt:.2f}m  "
              f"({'better' if rmse_filt < rmse_raw else 'worse'})")

    print("Done")


if __name__ == "__main__":
    main()
