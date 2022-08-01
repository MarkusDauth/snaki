import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
import parameters
pygame.init()

#font = pygame.font.Font('arial.ttf',25)
font = pygame.font.SysFont('arial.ttf',25)

# Reset 
# Play(action) -> Direction
# Game_Iteration
# is_collision

params = parameters.init_parameters()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
 
Point = namedtuple('Point','x , y')

FIELD_SIZE = params['game_field_size'] # must be even and atleast 6

BLOCK_SIZE=20
SPEED = 40
WHITE = (255,255,255)
RED = (200,0,0)
BLUE1 = (0,0,255)
BLUE2 = (0,100,255)
BLACK = (0,0,0)

class SnakeGameAI:
    def __init__(self,w=BLOCK_SIZE*FIELD_SIZE,h=BLOCK_SIZE*FIELD_SIZE):
        self.w=w
        self.h=h
        #init display
        self.display = pygame.display.set_mode((self.w,self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        #init game state
        self.reset()

    def reset(self):
        self.cardinal_direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE,self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()
        self.frame_iteration = 0
      

    def _place__food(self):
        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)
        if(self.food in self.snake):
            self._place__food()


    def play_step(self,cardinal_direction):
        self.frame_iteration+=1
        # 1. Collect the user input
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
        # 2. Move
        self._move(cardinal_direction)
        self.snake.insert(0,self.head)

        # 3. Check if game Over
        ate_apple = False
        game_over = False 
        if(self.is_collision() or self.frame_iteration > params['early_episode_end_steps']*(len(self.snake)-2) ):
            game_over=True
            return ate_apple, game_over,self.score
        # 4. Place new Food or just move
        if(self.head == self.food):
            self.score+=1
            ate_apple=True
            self._place__food()
            
        else:
            self.snake.pop()
        
        #5. Update UI and clock
        if(params['show_gui']):
            self._update_ui()
            self.clock.tick(SPEED)
        
        #6. Return game Over and Display Score
        
        return ate_apple,game_over,self.score

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display,BLUE1,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
            pygame.draw.rect(self.display,BLUE2,pygame.Rect(pt.x+4,pt.y+4,12,12))
        pygame.draw.rect(self.display,RED,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.score),True,WHITE)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def _move(self, cardinal_direction):
        self.cardinal_direction = cardinal_direction

        x = self.head.x
        y = self.head.y
        if(self.cardinal_direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.cardinal_direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.cardinal_direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.cardinal_direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)

        

    def is_collision(self,pt=None):
        if(pt is None):
            pt = self.head
        #hit boundary
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.snake[1:]):
            return True
        return False

    def get_cardinal_direction(self):
        return self.cardinal_direction
