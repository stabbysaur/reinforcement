"""
2018-10-04
Exercise from Arthur Juliani's RL tutorial (adapted for Pytorch)
Part 0: Q-Learning with Tables and Neural Networks

Update rule (Bellman):
    Q(s, a) = r + gamma * (max(Q(s', a')))
    The expected long-term reward for a given action = immediate reward +
    discounted future reward of the best future action in the next state!
"""

import gym
import numpy as np

"""
FrozenLake is a 4x4 grid of blocks: 
    --start
    --goal
    --safe
    --dangerous
    
The agent wishes to navigate start --> goal, moving up/down/left/right.
There is a wind that occasionally blows the agent onto a space they didn't choose.
The reward at each step is 0, goal 1.
"""

env = gym.make('FrozenLake-v0')

"""
There are 16 possible states (4x4 grid of blocks) and 4 possible actions (directions).
In total, we need a 16x4 table of Q-values (state, action pairs).
"""

Q = np.zeros([env.observation_space.n, env.action_space.n])