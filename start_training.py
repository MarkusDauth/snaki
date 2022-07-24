from agent_qlearning import Agent_QLearning
from agent_manhattan import Agent_Manhattan
from snake_world_env import Snake_World_Env
import parameters
import time
import matplotlib.pyplot as plt 
from IPython import display
import sys
from snake_game_ai import SnakeGameAI

plt.ion()

class Training_Setup:
    def __init__(self):
        self.params = parameters.init_parameters()

        if(self.params['method'] == 'qlearning'):
            self.agent = Agent_QLearning()
        elif(self.params['method'] == 'manhattan'):
            if (self.params['train']):
                print('error: manhattan can not be trained')
                sys.exit()
            else:
                self.agent = Agent_Manhattan()
        else:
            print('wrong parameter: method')
            sys.exit()

    def start_training(self):
        self.counter_games = 0
        self.counter_steps = 0
        plot_scores = []
        plot_mean_scores = []
        plot_mean_every_n_scores = []
        total_score = 0
        record = 0
        mean_every_n_score = 0 # used for running average
        mean_every_n_score_helper = 0 # used for running average
        game = SnakeGameAI()
        world_env = Snake_World_Env(game)
        start_time = time.time()

        if(self.params['train']):
            print('Starting TRAINING with '+self.params["method"])
            max_episodes = self.params['train_episodes']
        else:
            print('Starting TESTING with '+self.params["method"])
            max_episodes = self.params['test_episodes']

        while (self.counter_games < max_episodes):
            # Episode start
            if(self.params['train']):
                reward, done, score = self.agent.train_step(world_env, self.counter_games) # Agent does one step
            else:
                reward, done, score = self.agent.test_step(world_env) # Agent does one step

            self.counter_steps += 1
            if (done):
                # episode done
                self.counter_games += 1
                
                # the rest is visualization of the training
                # new High score 
                if(score > record): 
                    record = score
                
                plot_scores.append(score)
                total_score+=score
                mean_score = total_score / self.counter_games
                
                # mean every 20 games
                mean_every_n_score_helper = mean_every_n_score_helper + score
                if(self.counter_games % 20 == 0):
                    mean_every_n_score = mean_every_n_score_helper / 20
                    mean_every_n_score_helper = 0
                plot_mean_every_n_scores.append(mean_every_n_score)
                plot_mean_scores.append(mean_score)
                
                episode_end_time = time.time()
                episode_run_time = episode_end_time - start_time

                print(f'Game: {self.counter_games}\t\tScore: {score}\t\ttavg_score: {mean_score:.2f}\tRecord: {record}\tReward: {reward}\ttime: {episode_run_time:.2f}')

        # done training
        self.plot(plot_scores,plot_mean_scores,plot_mean_every_n_scores)
        input("DONE!")


    def plot(self, scores, mean_scores, mean_every_n_scores):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title("Training...")
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.plot(mean_every_n_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1,scores[-1],str(scores[-1]))
        plt.text(len(mean_scores)-1,mean_scores[-1],str(mean_scores[-1]))
        plt.text(len(mean_every_n_scores)-1,mean_every_n_scores[-1],str(mean_every_n_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)


if(__name__=="__main__"):
    setup = Training_Setup()
    setup.start_training()
    



