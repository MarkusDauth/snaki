import torch 
import random 
import numpy as np
from collections import deque
from model_qlearning import Linear_QNet,QTrainer


# added by Markus
from Snake.envs.snake_env import SnakeEnv
import parameters


MAX_MEMORY = 100_000
# BATCH_SIZE = 1000

class Agent_QLearning:
    def __init__(self):
        self.params = parameters.init_parameters()
        
        self.gamma = self.params['gamma']
        self.memory = deque(maxlen=self.params['memory_size']) # popleft()
        self.model = Linear_QNet(11,200,20,50,3) 
        self.trainer = QTrainer(self.model,lr=self.params['learning_rate'] ,gamma=self.gamma)    

    def _remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def _train_long_memory(self):
        if (len(self.memory) > self.params['batch_size']):
            mini_sample = random.sample(self.memory,self.params['batch_size'])
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def _train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def _get_train_action(self,state, counter_games):
        # random moves: tradeoff explotation / exploitation
        # self.epsilon = 80 - self.n_game
        self.epsilon = 1 - (counter_games * self.params['epsilon_decay_linear'])
        final_move = [0,0,0]
        if random.uniform(0,1) < self.epsilon:
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float).cuda()
            prediction = self.model(state0).cuda() # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

    def _get_test_action(self,state):
        final_move = [0,0,0]
        state0 = torch.tensor(state,dtype=torch.float).cuda()
        prediction = self.model(state0).cuda() # prediction by model 
        move = torch.argmax(prediction).item()
        final_move[move]=1 
        return final_move

    def train_step(self, world_env, counter_games, counter_steps):
        # Get Old state
        state_old = world_env.get_state()

        # get move
        final_move = self._get_train_action(state_old, counter_games)

        # perform move and get new state
        state_new, reward, done, info = world_env.step(final_move)
        score = info['score']

        # train short memory
        self._train_short_memory(state_old,final_move,reward,state_new,done)

        #self
        self._remember(state_old,final_move,reward,state_new,done)

        if(done):
            world_env.reset()
            self._train_long_memory()
            self.model.save()
            
        return reward, done, score

    def test_step(self, world_env, counter_games, counter_steps):
        state_old = world_env.get_state()
        final_move = self._get_test_action(state_old)

        observation, reward, done, info = world_env.step(final_move)
        score = info['score']

        if(done):
            world_env.reset()
        return reward, done, score

    def reset(self):
        pass



        