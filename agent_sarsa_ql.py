import random
from random import randint
import torch
import numpy as np
import parameters
import gym
import torch.nn as nn
import torch.nn.functional as F
import collections
from collections import deque
import time


# code from https://github.com/rogerlucena/snake-ai

class Agent_Sarsa_Ql(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.params = parameters.init_parameters()
        torch.set_grad_enabled(True)

        env_name = 'snake-v0'
        self.env = gym.make(env_name)

        self._network()
        self.gamma = self.params['gamma'] = 0.99
        self.memory = collections.deque(maxlen=self.params['memory_size'])
        self.saved_models_path = ''
        if(self.params['method'] == 'qlearning'):
            self.saved_models_path = 'saved_models_qlearning/model.h5'
        elif(self.params['method'] == 'sarsa'):
            self.saved_models_path = 'saved_models_sarsa/model.h5'
    
    def _network(self):
        self.f1 = nn.Linear(11, 200)
        self.f2 = nn.Linear(200, 20)
        self.f3 = nn.Linear(20, 50)
        self.f4 = nn.Linear(50, 3)

        # weights
        if not self.params['train']:
            if(self.params['method'] == 'qlearning'):
                self.saved_models_path = 'saved_models_qlearning/model.h5'
            elif(self.params['method'] == 'sarsa'):
                self.saved_models_path = 'saved_models_sarsa/model.h5'
            self.model = self.load_state_dict(torch.load(self.saved_models_path))
            print("weights loaded")


    def _get_epsilon_greedy_action(self, state_old):
        """
        Return the epsilon-greedy action for state_old.
        """
        if random.uniform(0, 1) < self.epsilon:
            # return a random action
            final_move = np.eye(3)[randint(0,2)]
        else:
            # choose the best action based on the old state
            with torch.no_grad():
                state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(self.params['device'])
                prediction = self(state_old_tensor)
                final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
        return final_move


    
    def _remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def _replay_mem(self, memory, batch_size):
        """
        Replay memory.
        """
        

        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory

        
        for state, action, reward, next_state, done in minibatch:
            self._train_short_memory(state, action, reward, next_state, done)

    def get_target(self, reward, next_state):
        """
        Return the appropriate TD target depending on the type of the
        agent (Q-Learning, SARSA or Expected-SARSA).
        """
        # next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(self.params['device'])
        next_state_tensor = torch.tensor(next_state.reshape((1, 11)), dtype=torch.float32).to(self.params['device'])
        q_values_next_state = self.forward(next_state_tensor[0])
        if self.params['method'] == 'qlearning':
            target = reward + self.gamma * torch.max(q_values_next_state) # Q-Learning is off-policy
        elif self.params['method'] == 'sarsa':
            next_action = self.get_epsilon_greedy_action(next_state) # SARSA is on-policy
            q_value_next_state_action = q_values_next_state[np.argmax(next_action)]
            target = reward + self.gamma * q_value_next_state_action
        else:
            raise ValueError('agent_type in get_target should necessarily be one of the supported agent types')
        return target

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

    def get_epsilon_greedy_action(self, state_old):
        """
        Return the epsilon-greedy action for state_old.
        """
        if random.uniform(0, 1) < self.epsilon:
            # return a random action
            final_move = np.eye(3)[randint(0,2)]
        else:
            # choose the best action based on the old state
            with torch.no_grad():
                state_old_tensor = torch.tensor(state_old.reshape((1, 11)), dtype=torch.float32).to(self.params['device'])
                prediction = self(state_old_tensor)
                final_move = np.eye(3)[np.argmax(prediction.detach().cpu().numpy()[0])]
        return final_move

    def _train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        target = reward
        state_tensor = torch.tensor(state.reshape((1, 11)), dtype=torch.float32, requires_grad=True).to(self.params['device'])
        # something here is very slow
        if not done:
            target = self.get_target(reward, next_state)
        output = self.forward(state_tensor)
        # ^^^slow

        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

    def train_step(self, world_env, counter_games, counter_steps):
        self.epsilon = 1 - (counter_games * self.params['epsilon_decay_linear'])

        state_old = self.env.get_state()

        final_move = self._get_epsilon_greedy_action(state_old)

        state_new, reward, done, info = self.env.step(final_move)
        score = info['score']

        #train


        self._train_short_memory(state_old,final_move,reward,state_new,done)
        self._remember(state_old,final_move,reward,state_new,done)
        
        self._replay_mem(self.memory, self.params['batch_size'])

        # do this at the end of an episode
        if(done):
            # train
            model_weights = self.state_dict()
            torch.save(model_weights, self.saved_models_path)
        
        return reward, done, score

    def test_step(self, world_env, counter_games, counter_steps):
        self.epsilon = 0.00

        state_old = self.env.get_state()

        final_move = self._get_epsilon_greedy_action(state_old)

        state_new, reward, done, info = self.env.step(final_move)
        score = info['score']
        
        return reward, done, score


    def reset(self):
        self.state = self.env.reset()


