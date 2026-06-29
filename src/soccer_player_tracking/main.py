import numpy as np

# Player GPS parameters
# Data collection frequency: 10 Hz
# Positional accuracy: 1-5 meters

def make_intervals(a, b, n):
    arr = np.linspace(a, b n) 
    return np.column_stack((arr[:-1], arr[1:]))

if __name__ == "__main__":
    # FIFA Regulation
    field = (105, 68)
    # 
    n_players = 11
    t_match = 10
    dt = 0.01
    n_steps = t_match * 60 / dt

    intervals = make_intervals(0, 2, 6)
    players_n = [] 

import random

numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)

    roles = {
        'GK': (1.8, 2),
        'CB': (1.8, 2),
        'OB': (1.8, 2),
        'CM': (1.8, 2),
        'WG': (1.8, 2),
        'ST': (1.8, 2)
    }
    players = dict() 
    for i in range(n_players):
        gps = GPS(np.random.uniform(1, 5) / 2)


        
        
