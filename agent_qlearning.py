import torch 
import random 
import numpy as np
from collections import deque
from model_qlearning import Linear_QNet,QTrainer


# added by Markus
from snake_world_env import Snake_World_Env
import parameters


MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent_QLearning:
    def __init__(self):
        self.params = parameters.init_parameters()
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=self.params['learning_rate'] ,gamma=self.gamma)    

    def _remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def _train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
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

    def train_step(self, world_env, counter_games):
        # Get Old state
        state_old = world_env.get_state()

        # get move
        final_move = self._get_train_action(state_old, counter_games)

        # perform move and get new state
        reward, done, score = world_env.play_step(final_move)
        state_new = world_env.get_state()

        # train short memory
        self._train_short_memory(state_old,final_move,reward,state_new,done)

        #self
        self._remember(state_old,final_move,reward,state_new,done)

        if(done):
            world_env.reset()
            self._train_long_memory()
            self.model.save()
            
        return reward, done, score

    def test_step(self, world_env):
        state_old = world_env.get_state()
        final_move = self._get_test_action(state_old)
        reward, done, score = world_env.play_step(final_move)
        if(done):
            world_env.reset()
        return reward, done, score



        