#coding=utf-8
import numpy as np
import cv2
from time import sleep
import pygame
global draw,mutex

MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]
BLACK= [0,0,0]
WHITE = [255,255,255]
SCREEN_SIZE = [320, 320]
BAR_SIZE = [40, 10]
BALL_SIZE = [20, 20]

class Game(object):
   
    def __init__(self):
#         pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
#         pygame.display.set_caption('Simple Game')
        self.reset()
#  
    def reset(self):
        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2
        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2
 
        self.ball_dir_x = -2  # -1 = left 1 = right  
        self.ball_dir_y = -2  # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
 
        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        image = cv2.cvtColor(cv2.resize(screen_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        self.s_t = np.stack((image, image, image, image), axis=2)
#         self.s_t=image
        self.reward=0
        self.terminal=False
        
    def process(self, action):
        self.screen.fill((0,0,0))
#         print(action)
        if action == MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 20
        elif action == MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 20
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
 
 
#         for event in pygame.event.get(): 
#             pass
#             if event.type == QUIT:
#                 pygame.quit()
#                 sys.exit()
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)
 
        self.ball_pos.left += self.ball_dir_x * 0.5
        self.ball_pos.bottom += self.ball_dir_y * 1
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)
 
        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1]-BAR_SIZE[1]+1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1
 
        reward = 0
#         self.bar_pos.top <= self.ball_pos.bottom and 
        if  self.bar_pos.top <= self.ball_pos.bottom and(self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            reward = 1  # ���н���
#         self.bar_pos.top <= self.ball_pos.bottom and 
        elif self.bar_pos.top <= self.ball_pos.bottom and  (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            reward = -1# û���гͷ�
            self.terminal=True
 
            # �����Ϸ��������
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        image = cv2.cvtColor(cv2.resize(screen_image, (84, 84)), cv2.COLOR_BGR2GRAY)
        image=np.reshape(image, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:,:,1:],image, axis = 2)  
        self.reward=reward
        pygame.display.update()
        
    def update(self):
        self.s_t=self.s_t1
    


