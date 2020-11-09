import time
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
        logging.basicConfig(filename = 'tabular_new.log', format = '%(message)s', level = logging.CRITICAL)
        Q = np.zeros((self.n_states, self.n_actions))
        print (Q.shape)
        #step_rewards = []
        for episode in range(episodes):
            logging.critical('Logging reset time')
            start = time.time()
            print (f'Starting Episode {episode + 1} ', end = '\r')
            if state_is_numerical_value:
                state = self.env.reset()
            else:
                state = self.env.reset()[access_key]
            logging.critical(time.time() - start)
            if data_needs_rounding_off:
                state = np.round(state, round_to_digits)
            total_training_rewards = 0
            policy = []
            last_achieved_goal = None
            s = (self.state_space == state).all(axis = 1)
            
            logging.critical('Logging episode time')
            start = time.time()
            for __ in range(max_steps):
                exp_exp_tradeoff = random.uniform(0,1)
                a = None
                #logging.info('Logging action retrieval time')
                if exp_exp_tradeoff < epsilon:
                    a = random.randrange(self.n_actions)
                else:
                    logging.info(Q[s, :])
                    a = np.argmax(Q[s, :])
                    logging.info(Q[s, a])
                policy.append(self.action_space[a])
                #logging.info(str(time.time() -  start))

                logging.info('Logging Simulator update time')
                new_state, reward, done, __ = self.env.step(self.action_space[a])
                logging.info(str(time.time() - start))

                #step_rewards.append(reward)
                if not state_is_numerical_value:
                    # Modification for dVRL alone
                    last_achieved_goal = new_state['achieved_goal']
                    new_state = new_state[access_key]
                if data_needs_rounding_off:
                    new_state = np.round(new_state, round_to_digits)
                try:
                    #logging.info('Logging Q-update time') 
                    s_new = np.where((self.state_space == new_state).all(axis = 1))
                    Q[s, a] = Q[s, a] + alpha * (reward + discount_factor * np.max(Q[s_new, :]) - Q[s, a])
                    #logging.info(str(time.time() - start))
                    state = new_state
                    s = s_new
                    total_training_rewards += reward
                except ValueError:
                    #print ('Exceeded state limits, Terminating', end = '\r')
                    done = True
                    
                if done:
                    #print (total_training_rewards, end = '\r')
                    break
            logging.critical(time.time()-start)
            print (f'Previous Episode completed in {time.time() - start} seconds', end = '\r')
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            self.training_rewards.append(total_training_rewards)
            self.achieved_final_goals.append(last_achieved_goal)
            self.episode_policy.append(policy)
            self.epsilons.append(epsilon)
            #print (f'Training score over time: {sum(self.training_rewards)/episodes}', end = '\r')
            
    def visualize(self, analysis = 'reward', start_episode = 1, end_episode = None):
        fig = plt.figure(figsize = (15, 9))

        if analysis == "reward": 
            self.get_loss_plot(start_episode) 
        elif analysis == "goal locations": 
            self.get_goal_locations_plot(fig, start_episode)

    def distance_from_goal(self):
        distances = np.zeros(len(self.achieved_final_goals))
        for it, goal in enumerate(self.achieved_final_goals):
            distances[it] = np.linalg.norm(goal - self.env.desired_goal)
        return distances

    def get_loss_plot(self, start_episode = 1, end_episode = None):
        plt.plot(np.arange(1, len(self.training_rewards[start_episode - 1:]) + 1, 1), self.training_rewards[start_episode - 1:])
        plt.xlabel('Episode')
        plt.ylabel('Reward (-Distance^2)')
        plt.show()

    def get_goal_locations_plot(self, fig, start_episode = 1, end_episode = None):
        ax = plt.axes(projection = '3d')
        desired_goal = self.env.desired_goal
        achieved_goals = np.array(self.achieved_final_goals)
        distances = self.distance_from_goal()

        ax.scatter3D(desired_goal[0], desired_goal[1], desired_goal[2], c = 'red')
        point_locs = ax.scatter3D(achieved_goals[:, 0], achieved_goals[:, 1], achieved_goals[:, 2], c = distances, cmap = 'viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.colorbar(point_locs, ax = ax)

    def close_sim(self):
        self.env.pr.shutdown()

