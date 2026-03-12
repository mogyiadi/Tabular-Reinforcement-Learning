#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code

        G = 0
        rev_rewards = reversed(range(len(rewards)))
        for t in rev_rewards:
            G = rewards[t] + self.gamma * G
            s = states[t]
            a = actions[t]
            self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Monte Carlo RL algorithm here!
    t_total = 0

    while t_total < n_timesteps:
        s = env.reset()

        states = [s]
        actions = []
        rewards = []

        for t_ep in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)

            states.append(s_next)
            actions.append(a)
            rewards.append(r)

            if t_total % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_timesteps.append(t_total)
                eval_returns.append(mean_return)
                
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

            t_total += 1
            s = s_next

            if done or t_total >= n_timesteps:
                break

        pi.update(states,actions,rewards)
       
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()
