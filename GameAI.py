import pygame
import random
from collections import namedtuple
from enum import Enum
from config import *
from color import *
import numpy as np

pygame.init()
font = pygame.font.Font(None,25)
Point = namedtuple('Point', 'x, y')

class TypeMove(Enum):
    RL = 1
    ASTAR = 2

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    IDLE = 5

class BlockType(Enum):
    NOTHING = 0
    WALL = 1
    SNAKE = 2
    APPLE = 3

class GameAI:

    width = None
    height = None
    clock = pygame.time.Clock()
    screen = None
    running = True

    foodList = []
    food = None

    blockList = []

    snakeHead = None
    snake = None
    direction = Direction.LEFT

    score = 0

    initFoodList = []
    initBlockList = []
    initX = 1
    initY = 1

    frameCount = 0

    def __init__(self, width=30, height=30, foodList=None, blockList=None, initX = 1, initY = 1):
        self.initBlockList=[]
        self.initFoodList=[]
        
        if (foodList is None):
            self.initFoodList = None
        else:
            for food in foodList:
                self.initFoodList.append(Point(food[0],food[1]))
        
        if (blockList is None):
            self.initBlockList = None
        else:
            for block in blockList:
                self.initBlockList.append(Point(block[0],block[1]))
        
        self.initX = initX
        self.initY = initY
           
        self.width = width
        self.height = height
        screen_size = (width*BLOCK_SIZE, height*BLOCK_SIZE)
        self.screen = pygame.display.set_mode(screen_size)
        self.reset()

    def reset(self):

        if (self.initBlockList is None):
            m = random.randint(1,self.width*self.height//100)
            self.blockList = []
            for _ in range(m):
                self.blockList.append(Point(random.randint(1,self.width),random.randint(1,self.height)))
        else:
            self.blockList = self.initBlockList
        if (self.initFoodList is None):
            k = random.randint(100,200)
            self.foodList = []
            for _ in range(k):
                new_point = Point(random.randint(1,self.width),random.randint(1,self.height))
                while (new_point in self.blockList):
                    new_point = Point(random.randint(1,self.width),random.randint(1,self.height))

                self.foodList.append(new_point)
        else:
            self.foodList = self.initFoodList
            

        self.snakeHead = Point(self.initX,self.initY)
        self.snake = [self.snakeHead]
        self.direction = Direction.LEFT
        self.score = 0
        self.frameCount = 0
        self.createFood()

    def drawBlock(self, point: Point, baseColor, subColor = None):
        base_x = (point.x-1)*BLOCK_SIZE
        base_y = (point.y-1)*BLOCK_SIZE
        if (subColor is None):
            subColor = baseColor
        pygame.draw.rect(self.screen, baseColor, pygame.Rect(base_x ,base_y ,BLOCK_SIZE ,BLOCK_SIZE))
        pygame.draw.rect(self.screen, subColor, pygame.Rect(base_x+4 , base_y+4, BLOCK_SIZE-2*4, BLOCK_SIZE-2*4))


    def drawUI(self):
        self.screen.fill(BLACK)
        
        for block in self.snake:
            self.drawBlock(block,BLUE,LIGHT_BLUE)

        for block in self.blockList:
            self.drawBlock(block,RED,LIGHT_RED)

        self.drawBlock(self.food,YELLOW,LIGHT_YELLOW)    

        text = font.render("Score: {}".format(self.score), True, WHITE)
        self.screen.blit(text, Point(0,0))
        text = font.render("Frame: {}".format(self.frameCount), True, WHITE)
        self.screen.blit(text, Point(0,20))

        # pygame.display.flip()
        pygame.display.update()

    def gameOver(self):
        self.running = False
        pygame.quit()
        quit()

    def handleEvent(self):
        for event in pygame.event.get():
            if (event.type==pygame.QUIT):
                self.gameOver()

    def createFood(self):
        if (len(self.foodList)<1):
            self.gameOver()
        else:
            nLose = 0
            self.food = self.foodList[0]
            self.foodList = self.foodList[1:]
            while (self.food in self.snake or self.food in self.blockList):
                nLose += 1
                self.food = self.foodList[0]
                self.foodList = self.foodList[1:]
            return nLose

    def inside(self, point=None):
        if (point is None):
            point = self.snakeHead
        c1 = (point.x > self.width)
        c2 = (point.x < 1)
        c3 = (point.y > self.height)
        c4 = (point.y < 1)
        if (c1 or c2 or c3 or c4):
            return False
        return True

    def collision(self, point=None):
        if (point is None):
            point = self.snakeHead
        
        if (self.inside(point)==False):
            return BlockType.WALL
        
        if (point in self.snake[1:]):
            return BlockType.SNAKE

        if (point in self.blockList):
            return BlockType.WALL
        
        if (point == self.food):
            return BlockType.APPLE

        return BlockType.NOTHING

    def updateObject(self, action, typeMove):
        game_end = False

        if (typeMove == TypeMove.RL):
            self.snakeMoveRL(action)
        else:
            self.snakeMoveAStar(action)
        self.snake.insert(0,self.snakeHead)

        reward = 0
        collideBlock = self.collision()
        if (collideBlock==BlockType.WALL or collideBlock==BlockType.SNAKE or self.frameCount > 100*len(self.snake)):
            reward = -10
            game_end = True
            return reward, game_end, self.score

        if (self.snakeHead==self.food):
            reward = 10
            self.score += 1
            nLose = self.createFood()
            # reward -= 10*nLose
        else:
            self.snake.pop()

        return reward, game_end, self.score

    def snakeMoveRL(self, action):
        x = self.snakeHead.x
        y = self.snakeHead.y

        # [forward,left,right]
        cw = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = cw.index(self.direction)
        if (np.array_equal(action,[1,0,0])):
            new_dir = cw[idx]
        elif (np.array_equal(action,[0,1,0])):
            new_dir = cw[(idx+1)%4]
        elif (np.array_equal(action,[0,0,1])):
            new_dir = cw[(idx-1+4)%4]
        self.direction = new_dir

        if (self.direction==Direction.UP):
            y -= 1
        elif (self.direction==Direction.DOWN):
            y += 1
        elif (self.direction==Direction.LEFT):
            x -= 1
        elif (self.direction==Direction.RIGHT):
            x += 1

        self.snakeHead = Point(x,y)

    def snakeMoveAStar(self, action):
        x = self.snakeHead.x
        y = self.snakeHead.y

        # [forward,left,right]
        cw = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        dx = [1, 0, -1,  0]
        dy = [0, 1,  0, -1]
        new_dir = -1
        for i in range(len(dx)):
            if (x+dx[i]==action.x and y+dy[i]==action.y):
                new_dir = cw[i]
        assert (new_dir!=-1)
        self.direction = new_dir

        if (self.direction==Direction.UP):
            y -= 1
        elif (self.direction==Direction.DOWN):
            y += 1
        elif (self.direction==Direction.LEFT):
            x -= 1
        elif (self.direction==Direction.RIGHT):
            x += 1

        self.snakeHead = Point(x,y)

    def playStep(self, action, typeMove=TypeMove.RL):
        self.frameCount += 1
        self.handleEvent()
        reward, game_end, score = self.updateObject(action,typeMove)
        if (game_end):
            return reward, game_end, score
        self.drawUI()
        self.clock.tick(SPEED)
        return reward, game_end, score

