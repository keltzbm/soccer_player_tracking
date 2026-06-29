def sanity_check(team, positions, measurements, filtered):
    print("=== Sanity Check ===")

    # Shapes
    n_players = len(team.players)
    n_steps   = positions.shape[1]
    print(f"Players     : {n_players}")
    print(f"Steps       : {n_steps}")
    print(f"Positions   : {positions.shape}   expected ({n_players}, {n_steps}, 2)")
    print(f"Measurements: {measurements.shape} expected ({n_players}, {n_steps}, 2)")
    print(f"Filtered    : {filtered.shape}    expected ({n_players}, {n_steps}, 4)")

    # Bounds — no player should leave the pitch
    assert positions[:, :, 0].max() <= 105, "Player left pitch (x > 105)"
    assert positions[:, :, 0].min() >= 0,   "Player left pitch (x < 0)"
    assert positions[:, :, 1].max() <= 68,  "Player left pitch (y > 68)"
    assert positions[:, :, 1].min() >= 0,   "Player left pitch (y < 0)"
    print("Bounds      : OK")

    # GK should barely move
    gk_std = positions[0].std(axis=0)
    print(f"GK std      : ({gk_std[0]:.2f}, {gk_std[1]:.2f})  expected ~(0.3, 0.3)")

    # Measurements should be noisier than true positions
    raw_rmse  = np.sqrt(((measurements - positions)**2).mean())
    filt_rmse = np.sqrt(((filtered[:, :, :2] - positions)**2).mean())
    print(f"Raw RMSE    : {raw_rmse:.3f} m")
    print(f"Filter RMSE : {filt_rmse:.3f} m")
    assert filt_rmse < raw_rmse, "Filter made things worse overall"
    print("Filter      : OK")

    print("=== All checks passed ===")
