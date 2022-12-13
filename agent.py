import torch
import numpy as np
import random
from collections import deque
from GameAI import GameAI, Direction, Point, BlockType
from model import QTrainer, Linear_QNet
import plotter 
from reader import *
from config import *

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
    
    def __init__(self):
        self.model = Linear_QNet(15,512,3)
        self.lr = LR
        self.nGame = 0
        self.gamma = GAMMA
        self.trainer = QTrainer(self.model,self.lr,self.gamma)

    def getState(self,game):
        head = game.snake[0]
        
        wallPoint = []
        applePoint = []
        snakePoint = []

        # dx = [1,1,0,-1,-1,-1, 0, 1]
        # dy = [0,1,1, 1, 0,-1,-1,-1]
        dx = [1,0,-1, 0]
        dy = [0,1, 0,-1]

        point = []
        for i in range(len(dx)):
            point.append(Point(head.x+dx[i],head.y+dy[i]))
            # point = Point(head.x,head.y)
            # infty = 2*np.sqrt(game.height**2+game.width**2)
            # applePoint.append(infty)
            # snakePoint.append(infty)
            # wallPoint.append(infty)
            # while (game.inside(point)):
            #     point = Point(point.x+dx[i], point.y+dy[i])
            #     collideBlock = game.collision(point)
            #     # print(point, collideBlock)

            #     if (collideBlock==BlockType.APPLE):
            #         applePoint[i]=min(applePoint[i],distance(head,point))
            #     if (collideBlock==BlockType.SNAKE):
            #         snakePoint[i]=min(snakePoint[i],distance(head,point))
            #     if (collideBlock==BlockType.WALL):
            #         wallPoint[i]=min(wallPoint[i],distance(head,point))

        # assert(len(applePoint)==8)
        # print(applePoint)
        # print("{}\n{}\n{}\n".format(applePoint,snakePoint,wallPoint))

        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)

        prevTwice = -2
        if (len(game.snake)==1):
            prevTwice = -1

        state = [
            # Distance
            # wallPoint[0],
            # wallPoint[1],
            # wallPoint[2],
            # wallPoint[3],
            # wallPoint[4],
            # wallPoint[5],
            # wallPoint[6],
            # wallPoint[7],
            
            # applePoint[0],
            # applePoint[1],
            # applePoint[2],
            # applePoint[3],
            # applePoint[4],
            # applePoint[5],
            # applePoint[6],
            # applePoint[7],
            
            # snakePoint[0],
            # snakePoint[1],
            # snakePoint[2],
            # snakePoint[3],
            # snakePoint[4],
            # snakePoint[5],
            # snakePoint[6],
            # snakePoint[7],

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
            game.food.y > game.snakeHead.y  # food down
        ]

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

def train(n, k, m, foodList, blockList, x0, y0):
    scores = []
    mean_scores = []
    total_scores = 0
    record = 0
    agent = Agent()
    game = GameAI(n,n,foodList,blockList,x0,y0)
    print("nGame\tScore\tRecord")
    while (True):
        state = agent.getState(game)
        final_move = agent.getAction(state)

        reward, done, score = game.playStep(final_move)
        state_new = agent.getState(game)

        agent.train_short_mem(state,final_move,reward,state_new,done)
        agent.remember(state,final_move,reward,state_new,done)
        if (done):
            game.reset()
            agent.nGame += 1
            agent.train_long_mem()
            if (score>record):
                record = score
                agent.model.save()
            print("{}\t{}\t{}".format(agent.nGame,score,record))

            scores.append(score)
            total_scores += score
            mean_scores.append(total_scores/agent.nGame)
            plotter.plot(scores,mean_scores)

if (__name__=='__main__'):
    # n, k, m, x0, y0, foodList, blockList = readInput('Data/input1.txt')
    foodList = None
    blockList = None
    n = 20
    k = -1
    m = -1
    x0 = 10
    y0 = 10
    train(n, k, m, foodList, blockList, x0, y0)