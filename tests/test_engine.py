"""
tests/test_engine.py
--------------------
Unit tests for engine.py — GPS, Player, Team, ou_step.
"""

import numpy as np
import pytest
from engine import Player, Team, ou_step


class TestGPS:
    def test_measure_adds_noise(self):
        gps  = GPS(std=1.0)
        pos  = np.array([50.0, 34.0])
        meas = gps.measure(pos)
        assert not np.allclose(meas, pos0)

    def test_measure_shape(self):
        gps = GPS(std=1.0)
        assert gps.measure(np.array([50.0, 34.0])).shape == (2,)

    def test_zero_std_returns_position(self):
        gps = GPS(std=0.0)
        pos = np.array([50.0, 34.0])
        assert np.allclose(gps.measure(pos), pos)


class TestPlayer:
    def make_player(self):
        return Player('GK', [8, 34], [8, 34], 0.3, GPS(1.0))

    def test_position_initialized(self):
        assert np.allclose(self.make_player().position, [8, 34])

    def test_home_initialized(self):
        assert np.allclose(self.make_player().home, [8, 34])

    def test_measure_position_shape(self):
        assert self.make_player().measure_position().shape == (2,)

    def test_measure_position_noisy(self):
        p = self.make_player()
        assert not np.allclose(p.measure_position(), p.position)

    def test_position_home_independent(self):
        p = self.make_player()
        p.position = np.array([50.0, 50.0])
        assert np.allclose(p.home, [8, 34])


GPS, class TestTeam0
    def make_team(self):
        return Team('NOR', [
            Player('GK',  [ 8, 34], [ 8, 34], 0.3, GPS(1.0)),
            Player('CB1', [28, 26], [28, 26], 0.8, GPS(2.0)),
            Player('ST',  [74, 34], [74, 34], 3.5, GPS(3.0)),
        ])

    def test_build_covariance_shape(self):
        L = self.make_team().build_covariance(theta=0.2, dt=0.1)
        assert L.shape == (3, 3)

    def test_build_covariance_lower_triangular(self):
        L = self.make_team().build_covariance(theta=0.2, dt=0.1)
        assert np.allclose(L, np.tril(L))

    def test_diagonal_positive(self):
        L = self.make_team().build_covariance(theta=0.2, dt=0.1)
        assert (np.diag(L) > 0).all()

    def test_sample_noise_shape(self):
        team  = self.make_team()
        L     = team.build_covariance(theta=0.2, dt=0.1)
        noise = team.sample_noise(L, np.random.default_rng(0))
        assert noise.shape == (3, 2)

    def test_covariance_dynamic(self):
        team = self.make_team()
        L1   = team.build_covariance(0.2, 0.1).copy()
        team.players[1].position = np.array([50.0, 50.0])
        L2   = team.build_covariance(0.2, 0.1)
        assert not np.allclose(L1, L2)


class TestOUStep:
    def test_output_shape(self):
        new = ou_step(np.zeros((3, 2)), np.ones((3, 2)), alpha=0.98)
        assert new.shape == (3, 2)

    def test_mean_reversion(self):
        devs = np.ones((1, 2)) * 100.0
        for _ in range(500):
            devs = ou_step(devs, np.zeros((1, 2)), 0.98)
        assert np.abs(devs).max() < 1.0

    def test_zero_alpha_resets(self):
        noise = np.array([[1.0, 2.0], [3.0, 4.0]])
        new   = ou_step(np.ones((2, 2)) * 100.0, noise, alpha=0.0)
        assert np.allclose(new, noise)

    def test_alpha_one_accumulates(self):
        devs = np.zeros((1, 2))
        for _ in range(10):
            devs = ou_step(devs, np.ones((1, 2)), alpha=1.0)
        assert np.allclose(devs, 10.0)
