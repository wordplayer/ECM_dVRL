import time
import random
import numpy as np
import matplotlib.pyplot as plt

class Tabular:
    def __init__(self, env, state_space = None, action_space = None):
        """
        This class runs the tabular Q-learning method to solve a problem
        with a discrete state-action space. It can be run with any form of
        Gym environment (available or custom). 
        
        If one plans to use an existing Gym environment, you need only pass 
        the instance of the environment. For a custom one, please pass the 
        chosen state-action space as instances
        """
        
        self.env = env
        if not state_space:
            state_space = env.observation_space
        if not action_space:
            action_space = env.action_space
            
        self.state_space = np.array(state_space)
        self.action_space = np.array(action_space)
        self.n_states = self.state_space.shape[0]
        self.n_actions = self.action_space.shape[0]
        
        self.training_rewards = []
        self.epsilons = []
        self.achieved_final_goals = []
        self.episode_policy = []
        
    def run(self, episodes, max_steps = 100, state_is_numerical_value = True,
            access_key = None, data_needs_rounding_off = False, round_to_digits = 3,
            alpha = 0.7, discount_factor = 0.618, epsilon = 1, max_epsilon = 1, min_epsilon = 0.01,
            decay = 0.01):
        
        Q = np.zeros((self.n_states, self.n_actions))
        #print (Q.shape)
        step_rewards = []
        for episode in range(episodes):
            print (f'Starting Episode {episode + 1} ', end = '\r')
            state = None
            if state_is_numerical_value:
                state = self.env.reset()
            else:
                state = self.env.reset()[access_key]
            if data_needs_rounding_off:
                state = np.round(state, round_to_digits)
            total_training_rewards = 0
            policy = []
            last_achieved_goal = None
            
            s = np.where((self.state_space == state).all(axis = 1))
            a = None
            s_new = None
            
            start = time.time()
            for step in range(max_steps):
                exp_exp_tradeoff = random.uniform(0,1)
                action = None
                if exp_exp_tradeoff < epsilon:
                    a = random.randrange(self.n_actions)
                else:
                    a = np.argmax(Q[s, :])
                action = self.action_space[a]
                policy.append(action)
                new_state, reward, done, info = self.env.step(action)
                #step_rewards.append(reward)
                if not state_is_numerical_value:
                    # Modification for dVRL alone
                    last_achieved_goal = new_state['achieved_goal']
                    new_state = new_state[access_key]
                if data_needs_rounding_off:
                    new_state = np.round(new_state, round_to_digits)
                
                try:
                    if not state_is_numerical_value:
                        s_new = np.where((self.state_space == new_state).all(axis = 1))
                    else:
                        s = state
                        a = action
                        s_new = new_state    
                    Q[s, a] = Q[s, a] + alpha * (reward + discount_factor * np.max(Q[s_new, :]) - Q[s, a])
                    state = new_state
                    s = s_new
                    total_training_rewards += reward
                except ValueError as e:
                    #print ('Exceeded state limits, Terminating', end = '\r')
                    done = True
                    
                if done:
                    #print (total_training_rewards, end = '\r')
                    break
            #print (f'Previous Episode completed in {time.time() - start} seconds', end = '\r')
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            self.training_rewards.append(total_training_rewards)
            self.achieved_final_goals.append(last_achieved_goal)
            self.episode_policy.append(policy)
            self.epsilons.append(epsilon)
            #print (f'Training score over time: {sum(self.training_rewards)/episodes}', end = '\r')
            
    def visualize(self, start_episode = 1, end_episode = None):
        plt.plot(np.arange(1, len(self.training_rewards[start_episode - 1:]) + 1, 1), self.training_rewards[start_episode - 1:])
        plt.xlabel('Episode')
        plt.ylabel('Reward (-Distance^2)')
        plt.show()
        
    def close_sim(self):
        self.env.pr.shutdown()

