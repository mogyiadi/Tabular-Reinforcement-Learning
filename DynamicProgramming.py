#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""
import time

import numpy as np
from Environment import StochasticWindyGridworld


class QValueIterationAgent:
    """ Class to store the Q-value iteration solution, perform updates, and select the greedy action """

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))

    def select_action(self, s):
        """ Returns the greedy best action in state s """
        a = np.argmax(self.Q_sa[s])
        return a

    def update(self, s, a, p_sas, r_sas):
        """ Function updates Q(s,a) using p_sas and r_sas """

        value = 0
        for i, p in enumerate(p_sas):
            value += p * (r_sas[i] + self.gamma * np.max(self.Q_sa[i]))

        self.Q_sa[s, a] = value

def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    """ Runs Q-value iteration. Returns a converged QValueIterationAgent object """

    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    max_error = np.inf
    i = 0
    while max_error > threshold:
        i += 1
        max_error = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QIagent.Q_sa[s, a]
                p_sas, r_sas = env.model(s, a)
                QIagent.update(s, a, p_sas, r_sas)
                max_error = max(max_error, abs(x - QIagent.Q_sa[s, a]))

        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    return QIagent


def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env, gamma, threshold)
    # time.sleep(5)

    # view optimal policy
    done = False
    s = env.reset()
    sum_rewards = 0
    step = 0
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.5)
        s = s_next
        sum_rewards += r
        step += 1

    mean_reward_per_timestep = sum_rewards / step

    # TO DO: Compute mean reward per timestep under the optimal policy
    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    print("Number of steps it took: ", step)
    print("Sum Reward: ", sum_rewards)


if __name__ == '__main__':
    experiment()
