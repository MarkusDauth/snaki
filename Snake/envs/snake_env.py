from snake_game_ai import SnakeGameAI,Direction,Point,BLOCK_SIZE,Direction
import numpy as np
from enum import Enum
import gym
from gym import spaces

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        self.game = SnakeGameAI()
        self.reset()
        
        self.observation_space = spaces.MultiBinary(11)
        self.action_space = spaces.Discrete(3)

    def get_game(self):
        'this is needed for manhattan'
        return self.game

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self):
        head = self.game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.game.cardinal_direction == Direction.LEFT
        dir_r = self.game.cardinal_direction == Direction.RIGHT
        dir_u = self.game.cardinal_direction == Direction.UP
        dir_d = self.game.cardinal_direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and self.game.is_collision(point_u))or
            (dir_d and self.game.is_collision(point_d))or
            (dir_l and self.game.is_collision(point_l))or
            (dir_r and self.game.is_collision(point_r)),

            # Danger right
            (dir_u and self.game.is_collision(point_r))or
            (dir_d and self.game.is_collision(point_l))or
            (dir_u and self.game.is_collision(point_u))or
            (dir_d and self.game.is_collision(point_d)),

            #Danger Left
            (dir_u and self.game.is_collision(point_r))or
            (dir_d and self.game.is_collision(point_l))or
            (dir_r and self.game.is_collision(point_u))or
            (dir_l and self.game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            self.game.food.x < self.game.head.x, # food is in left
            self.game.food.x > self.game.head.x, # food is in right
            self.game.food.y < self.game.head.y, # food is up
            self.game.food.y > self.game.head.y  # food is down
        ]

        return np.array(state,dtype=int)

    def step(self,action):
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn 
        # [0,0,1] -> Left Turn

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # print(action)
        # print(type(self.game.get_cardinal_direction()))
        # print(type(clock_wise[0]))

        idx = clock_wise.index(self.game.get_cardinal_direction())
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn

        ate_apple, done, score = self.game.play_step(new_dir)
        
        # calculate reward
        reward = 0
        if (ate_apple):
            reward += 1
        if (done):
            reward -= 1

        observation = self.get_state()
        
        info = {'score':score}
        return observation, reward, done, info
    
    def reset(self):
        self.game.reset()
        return self.get_state()

