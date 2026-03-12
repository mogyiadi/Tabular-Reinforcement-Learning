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

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''


        T_ep = len(rewards)
        
        for t in range(T_ep):
            m = min(n, T_ep - t)
        
            G = 0
            for i in range(m):
                G += (self.gamma ** i) * rewards[t + i]
                
            is_terminal_at_horizon = done and (t + m == T_ep)
            
            if not is_terminal_at_horizon:
                s_next = states[t + m]
                G += (self.gamma ** m) * np.max(self.Q_sa[s_next])
                
            s = states[t]
            a = actions[t]
            self.Q_sa[s, a] = self.Q_sa[s, a] + self.learning_rate * (G - self.Q_sa[s, a])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
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
               env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True, step_pause=0.1)
               
            t_total += 1
            s = s_next
            
            if done or t_total >= n_timesteps:
                break
                
        pi.update(states, actions, rewards, done, n)

    return np.array(eval_returns), np.array(eval_timesteps) 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
