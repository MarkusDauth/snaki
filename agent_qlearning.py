import torch 
import random 
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model_qlearning import Linear_QNet,QTrainer
from Helper import plot

# added by Markus
from env_snake import World


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float).cuda()
            prediction = self.model(state0).cuda() # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_mean_every_n_scores = []
    total_score = 0
    record = 0
    mean_every_n_score = 0
    mean_every_n_score_helper = 0
    agent = Agent()
    world = World()
    while True:
        # Get Old state
        state_old = world.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = world.play_step(final_move)
        state_new = world.get_state()

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            # Train long memory,plot result
            world.reset()
            agent.n_game += 1
            agent.train_long_memory()
            
            # new High score 
            if(score > record): 
                record = score
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            agent.model.save()
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            
            # mean every 20 games
            mean_every_n_score_helper = mean_every_n_score_helper + score
            if(agent.n_game % 20 == 0):
                mean_every_n_score = mean_every_n_score_helper / 20
                mean_every_n_score_helper = 0
            plot_mean_every_n_scores.append(mean_every_n_score)

            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores,plot_mean_every_n_scores)


if(__name__=="__main__"):
    train()