""" Proximal Policy Optimization (PPO): https://openai.com/blog/openai-baselines-ppo/
    train PPO-Agent
"""


# ------------------------------ Import Libraries -----------------------------
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import Snake
import parameters


'''
# policy folders
folder_name = "PPO_switchsteering"
folder_name2 = "SolvedPPO_switchsteering"

try:
    os.mkdir(folder_name)
except OSError as e:
    print(e)

try:
    os.mkdir(folder_name2)
except OSError as e:
    print(e)
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------- PyTorch Implementation of PPO -----------------------
class Memory:
    """
    A memory space for PPO agent to pool the collected experience

    Args:
        actions (list): The actions that the agent took.
        states (list): The states that the agent observed.
        logprobs (list): The log probabilities of the chosen action.
        rewards (list): The rewards that the agent received from the environment after taking an action.
        is_terminals (list): The episode-terminal flag that the agent received after taking an action. Done or terminal.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    """
    Model (network) class, using actor-critic approach `actor-critic <https://arxiv.org/abs/1602.01783>`

    Actor-Critic has two models: the Actor and the Critic.

    The Actor corresponds to the policy π and it chooses the action and update the policy network.
    The actor network returns a probability for each action.

    The critic network estimates the value of the state-action pair. Based on these value functions, the critic evaluates the actions made by the actor.
    The critic network returns the estimated value of each action, given a state

    Args:
        state_dim (int): state space dimension
        action_dim (int): agent's action space dimension
        n_latent_var (int): number of variables in hidden layer
    """
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 20),
            nn.Tanh(),
            nn.Linear(20, 50),
            nn.Tanh(),
            nn.Linear(50, action_dim),
            nn.Softmax(dim=-1)
        )

        '''
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        '''

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.Tanh(),
            nn.Linear(200, 20),
            nn.Tanh(),
            nn.Linear(20, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        """
        Passes the state observed into action_layer (actor network) to determine the action
        that the agent should take.
        
        Args:
            state (list): a list contatining the state observations

        Returns: 
            action (int): a number that indicates the action to be taken for gym environment
            log_prob (tensor): a tensor that contains the log probability of the action taken
        """
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    """
    Main PPO Class `PPO OpenAI <https://openai.com/blog/openai-baselines-ppo/>` `PPO paper <https://arxiv.org/abs/1707.06347>`

    Args:
        lr (float): learning rate
        betas (tuple): beta factor
        gamma (float): discount factor
        eps_clip (float): clip parameter for PPO
        K_epochs (int): update policy for K epochs
    """
    def __init__(self,
                 state_dim, action_dim, n_latent_var, lr,
                 betas, gamma, K_epochs, eps_clip
                 ):
        self.params = parameters.init_parameters()

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        """
        This function updates PPO Policy. 

        Monte Carlo estimation is used to estimate state rewards. Then the rewards are normalized and converted to tensor.
        Then the Policy will be optimized for K epochs by Evaluating old actions and values, Finding Surrogate Loss and taking gradient step.
        See: `PPO paper <https://arxiv.org/abs/1707.06347>`

        And at the end the weights will be updated.
        """
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.params['k_epochs']):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())



class Agent_PPO:
    def __init__(self, is_training_run):
        

        self.params = parameters.init_parameters()

        env_name = 'snake-v0'
        self.env = gym.make(env_name)

        #env.make_window()
        state_dim = self.env.observation_space.shape[0]          # state space dimention
        action_dim = self.env.action_space.n           # agent's action space dimention

        # Load Model
        self.memory = Memory()
        self.ppo = PPO(state_dim, action_dim, self.params['n_latent_var'] ,
                self.params['learning_rate'], self.params['betas'], self.params['gamma'], self.params['k_epochs'], self.params['eps_clip'])

        if not is_training_run:
            trained_model = './saved_models_ppo/model_ppo.pth'
            print('loaded model: ',trained_model)
            self.ppo.policy_old.load_state_dict(torch.load(trained_model))

        # not needed logging
        '''
        # logging variables
        running_score = 0
        avg_length = 0
        timestep = 0
        rewardList = []

        # log files
        log_dir = "PPO_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + env_name + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        # create new log file for each run
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')
        '''


    def train_step(self, world_env, counter_games, timestep):    
        # Running policy_old:

        action_index = self.ppo.policy_old.act(self.state, self.memory)
        action = [0,0,0]
        action[action_index]=1
        self.state, reward, done, info = self.env.step(action)

        # Saving reward and is_terminal:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if timestep % self.params['ppo_update_timestep'] == 0:
            self.ppo.update(self.memory)
            self.memory.clear_memory()
            timestep = 0
            model_path = os.path.join('saved_models_ppo', 'model_ppo.pth')
            torch.save(self.ppo.policy.state_dict(), model_path)

        return reward, done, info['score']

    def test_step(self, world_env, counter_games, timestep):
        action_index = self.ppo.policy_old.act(self.state, self.memory)
        action = [0,0,0]
        action[action_index]=1
        self.state, reward, done, info = self.env.step(action)

        '''
        # Saving reward and is_terminal:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if timestep % self.params['ppo_update_timestep'] == 0:
            self.ppo.update(self.memory)
            self.memory.clear_memory()
            #timestep = 0

        #if (done):
            # self.state = self.reset()
        '''
        return reward, done, info['score']


    def reset(self):
        self.state = self.env.reset()
