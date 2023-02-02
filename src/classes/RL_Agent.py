import torch
import numpy as np
import random
import os
from collections import deque
import copy
import warnings
warnings.filterwarnings('ignore')

from environment.GameAI import Direction, Point, BlockType
from utils.model import Linear_QNet
from utils.trainer import QTrainer
from utils.reader import *
from conf.config import *

def distance(pointA,pointB):
    dx = pointA.x-pointB.x
    dy = pointA.y-pointB.y
    return np.sqrt(dx**2+dy**2)

class Agent:
    nGame = 0
    epsilon = 0
    gamma = 0
    mem = deque(maxlen=MAX_MEM) # Pop left when exceeded
    model = None
    lr = None
    trainer = None
    nDirection = None
    rayDistance = None
    
    def __init__(self, nDirection=4, rayDistance=False):
        assert(nDirection==4 or nDirection==8)

        nInput = 15
        if (rayDistance):
            nInput = 15 + nDirection*3
        self.model = Linear_QNet(nInput,512,3)

        self.lr = LR
        self.nGame = 0
        self.gamma = GAMMA
        self.trainer = QTrainer(self.model,self.lr,self.gamma)
        self.nDirection = nDirection
        self.rayDistance = rayDistance

    def save(self, epoch, score, total_scores, title=None, path='model'):
        if (not os.path.exists(path)): os.makedirs(path)
        ckpt = {
            'epoch': epoch,
            'score': score,
            'total_scores': total_scores,
            'model': self.model.state_dict(),
        }
        if (title): name = f"{path}/{title}"
        else: name = f"{path}/model_{epoch}.bin"
        torch.save(ckpt, name)

    def load(self, path='model', verbose=True):
        files = os.listdir(path)
        files = [f[6:-4] for f in files]
        last_epoch = -1
        for f in files:
            try:
                epoch = int(f)
                if (epoch > last_epoch): last_epoch = epoch
            except: pass
        if (last_epoch==-1): name = f'{path}/model_best.bin'
        else: name = f'{path}/model_{last_epoch}.bin'
        ckpt = torch.load(name)

        if (verbose):
            print("="*10,"Load model","="*10)
            print(f"File name: {name}")
            print(f"Epoch: {ckpt['epoch']}")
            print(f"Score: {ckpt['score']}")
            print(f"Total score: {ckpt['total_scores']}")
            print("="*32)
        self.model.load_state_dict(ckpt['model'])
    
        return (ckpt['epoch'], ckpt['score'], ckpt['total_scores'])

    def getState(self,game):
        head = game.snake[0]

        if (self.nDirection==8):
            dx = [1,1,0,-1,-1,-1, 0, 1]
            dy = [0,1,1, 1, 0,-1,-1,-1]
        else: 
            dx = [1,0,-1, 0]
            dy = [0,1, 0,-1]

        point = []
        wallPoint = []
        applePoint = []
        snakePoint = []
        for i in range(len(dx)):
            point.append(Point(head.x+dx[i],head.y+dy[i]))
            
            # Calculate ray distance
            cur = Point(head.x,head.y)
            infty = 2*np.sqrt(game.height**2+game.width**2)
            applePoint.append(infty)
            snakePoint.append(infty)
            wallPoint.append(infty)
            while (game.inside(cur)):
                cur = Point(cur.x+dx[i], cur.y+dy[i])
                collideBlock = game.collision(cur)

                distanceFromHead = distance(head, cur)
                if (collideBlock==BlockType.APPLE):
                    applePoint[i]=min(applePoint[i], distanceFromHead)
                if (collideBlock==BlockType.SNAKE):
                    snakePoint[i]=min(snakePoint[i], distanceFromHead)
                if (collideBlock==BlockType.WALL):
                    wallPoint[i]=min(wallPoint[i], distanceFromHead)

        prevTwice = -2
        if (len(game.snake)==1):
            prevTwice = -1
        
        ray_state = [
            wallPoint[i] for i in range(len(dx))
        ] + [
            applePoint[i] for i in range(len(dx))
        ] + [
            snakePoint[i] for i in range(len(dx))
        ]

        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)

        basic_state = [
            # Danger straight
            (dir_r and (game.collision(point[0])==BlockType.WALL or game.collision(point[0])==BlockType.SNAKE)) or 
            (dir_d and (game.collision(point[1])==BlockType.WALL or game.collision(point[1])==BlockType.SNAKE)) or 
            (dir_l and (game.collision(point[2])==BlockType.WALL or game.collision(point[2])==BlockType.SNAKE)) or 
            (dir_u and (game.collision(point[3])==BlockType.WALL or game.collision(point[3])==BlockType.SNAKE)),

            # Danger right
            (dir_r and (game.collision(point[1])==BlockType.WALL or game.collision(point[1])==BlockType.SNAKE)) or 
            (dir_d and (game.collision(point[2])==BlockType.WALL or game.collision(point[2])==BlockType.SNAKE)) or 
            (dir_l and (game.collision(point[3])==BlockType.WALL or game.collision(point[3])==BlockType.SNAKE)) or 
            (dir_u and (game.collision(point[0])==BlockType.WALL or game.collision(point[0])==BlockType.SNAKE)),
            
            # Danger left
            (dir_r and (game.collision(point[3])==BlockType.WALL or game.collision(point[3])==BlockType.SNAKE)) or 
            (dir_d and (game.collision(point[0])==BlockType.WALL or game.collision(point[0])==BlockType.SNAKE)) or 
            (dir_l and (game.collision(point[1])==BlockType.WALL or game.collision(point[1])==BlockType.SNAKE)) or 
            (dir_u and (game.collision(point[2])==BlockType.WALL or game.collision(point[2])==BlockType.SNAKE)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Tail direction
            game.snake[-1].x == game.snake[prevTwice].x-1,
            game.snake[-1].x == game.snake[prevTwice].x+1,
            game.snake[-1].y == game.snake[prevTwice].y-1,
            game.snake[-1].y == game.snake[prevTwice].y+1,

            # Food location 
            game.food.x < game.snakeHead.x,  # food left
            game.food.x > game.snakeHead.x,  # food right
            game.food.y < game.snakeHead.y,  # food up
            game.food.y > game.snakeHead.y   # food down
        ]

        if (self.rayDistance): state = ray_state + basic_state
        else:  state = basic_state

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.mem.append((state, action, reward, next_state, done))

    # Batch training
    def train_long_mem(self):
        if (len(self.mem)>BATCH_SIZE):
            samples = random.sample(self.mem,BATCH_SIZE)
        else:
            samples = self.mem

        states,actions,rewards,next_states,dones = zip(*samples)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    # Local training
    def train_short_mem(self, state, action, reward, next_state, done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def getAction(self, state):
        self.epsilon = 80 - self.nGame
        final_move = [0,0,0]
        move = -1
        if (random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
        else:
            tensor_state = torch.tensor(state,dtype=torch.float)
            prediction = self.model(tensor_state)
            move = torch.argmax(prediction).item()
        assert(move!=-1)
        final_move[move] = 1
        return final_move
