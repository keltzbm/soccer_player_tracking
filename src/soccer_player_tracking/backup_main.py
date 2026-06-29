import numpy as np
import random

from engine import GPS, Player, Team, OUProcess
from sanity_test import sanity_check

def make_intervals(a, b, n):
    arr = np.linspace(a, b, n) 
    return np.column_stack((arr[:-1], arr[1:]))


if __name__ == "__main__":
    # pitch
    pitch = (105, 68)
    thirds = make_intervals(0, 105, 4)
    channels = make_intervals(0, 68, 6)

    middle_third = thirds[1] 
    x = [middle_third[0], pitch[0] / 2, middle_third[1]]
    y = [channels[1][0], pitch[1] / 2, channels[-1][1]]

    X, Y = np.meshgrid(x, y)



    n_players = 11
    # Game time: 45 min -> 2700 s
    # t_game = 10 * 60
    t_game = 10
    dt = 0.001
    steps = t_game / dt

    theta_intervals = make_intervals(0, 2, 6)
    print(theta_intervals)
    # mapping = {
    #     'GK': theta_intervals[
    players = []

    # roles = ['GK', 'CB', 'OB', 'CM', 'WG', 'ST']
    role = ['GK', 'D', 'M', 'A']
    
    formation = [['GK',], ['LB', 'LCB', 'RCB', 'RB',], ['LCM', 'CM', 'RCM',], ['LW', 'ST', 'RW']]

    homes = [[6, 34]]
    for x in np.linspace(0, pitch[0], 5, endpoint=False)[1:]:
        for y in np.linspace(0, pitch[1], endpoint=False)[1:]:
            home = [x, y]
        players.append(Player('{i}{j}', current, home, GPS(np.random.rand(1, 5) / 2)))
    team = Team('BMK', players)


    ouprocess = OUProcess(team)
    
    current_time = 0
    while current_time <= final_time:
        engine.step(dt)     
        curent += dt




    positions = team.get_positions()
    sanity_check(team, positions, None, None)



    




    
