import torch 
import random 
import numpy as np
from collections import deque
from model_qlearning import Linear_QNet,QTrainer
from snake_game_ai import SnakeGameAI,Direction,Point,BLOCK_SIZE,Direction

# This agent uses the game world directly

class Agent_Manhattan:
    def __init__(self):
        pass

    def _get_test_action(self, game):
        # Action taken from snake_gameai.py
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn 
        # [0,0,1] -> Left Turn

        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.cardinal_direction == Direction.LEFT
        dir_r = game.cardinal_direction == Direction.RIGHT
        dir_u = game.cardinal_direction == Direction.UP
        dir_d = game.cardinal_direction == Direction.DOWN

        '''
        danger_is_straight = ((dir_u and game.is_collision(point_u)) or
        (dir_d and game.is_collision(point_d))or
        (dir_l and game.is_collision(point_l))or
        (dir_r and game.is_collision(point_r)))

        danger_is_right = ((dir_u and game.is_collision(point_r))or
        (dir_d and game.is_collision(point_l))or
        (dir_u and game.is_collision(point_u))or
        (dir_d and game.is_collision(point_d)))

        danger_is_left = ((dir_u and game.is_collision(point_r))or
        (dir_d and game.is_collision(point_l))or
        (dir_r and game.is_collision(point_u))or
        (dir_l and game.is_collision(point_d)))
        '''
        # food is left
        if (game.food.x < game.head.x):
            if(dir_l and not game.is_collision(point_l)):
                return [1,0,0]
            if(dir_u and not game.is_collision(point_l)):
                return [0,0,1]
            if(dir_d and not game.is_collision(point_l)):
                return [0,1,0]
        # food is in right
        elif (game.food.x > game.head.x):
            if(dir_r and not game.is_collision(point_r)):
                return [1,0,0]
            if(dir_u and not game.is_collision(point_r)):
                return [0,1,0]
            if(dir_d and not game.is_collision(point_r)):
                return [0,0,1]
        # food is up
        elif (game.food.y < game.head.y):
            if(dir_u and not game.is_collision(point_u)):
                return [1,0,0]
            if(dir_r and not game.is_collision(point_u)):
                return [0,0,1]
            if(dir_l and not game.is_collision(point_u)):
                return [0,1,0]
        # food is down
        elif (game.food.y > game.head.y):
            if(dir_d and not game.is_collision(point_d)):
                return [1,0,0]
            if(dir_r and not game.is_collision(point_d)):
                return [0,1,0]
            if(dir_l and not game.is_collision(point_d)):
                return [0,0,1]
        # way is blocked or food is behind snake
        if(dir_u and not game.is_collision(point_r)):
            return [0,1,0]
        if(dir_u and not game.is_collision(point_l)):
            return [0,0,1]
        if(dir_d and not game.is_collision(point_l)):
            return [0,1,0]
        if(dir_d and not game.is_collision(point_r)):
            return [0,0,1]
        if(dir_l and not game.is_collision(point_u)):
            return [0,1,0]
        if(dir_l and not game.is_collision(point_d)):
            return [0,0,1]
        if(dir_r and not game.is_collision(point_d)):
            return [0,1,0]
        if(dir_r and not game.is_collision(point_u)):
            return [0,0,1]        

        return [1,0,0]


    def test_step(self, world_env):
        # Get Old state
        # state = world_env.get_state()
        game = world_env.get_game()

        # get move
        final_move = self._get_test_action(game)

        # perform move and get new state
        reward, done, score = world_env.play_step(final_move)

        if(done):
            world_env.reset()

            
        return reward, done, score