import pygame
from collections import namedtuple
from enum import Enum
from conf.config import *
from conf.color import *
from utils.reader import *
from argparse import ArgumentParser

pygame.init()
font = pygame.font.Font(None,25)
Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    IDLE = 5

class Game:

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
    direction = Direction.IDLE

    score = 0

    def __init__(self, width=30, height=30, foodList=[], blockList=[], initX = 1, initY = 1):
        for food in foodList:
            self.foodList.append(Point(food[0],food[1]))
        for block in blockList:
            self.blockList.append(Point(block[0],block[1]))
        
        self.width = width
        self.height = height
        screen_size = (width*BLOCK_SIZE, height*BLOCK_SIZE)
        self.screen = pygame.display.set_mode(screen_size)

        self.snakeHead = Point(initX,initY)
        self.snake = [self.snakeHead]

        self.createFood()

    def gameOver(self):
        self.running = False
        pygame.quit()
        quit()

    def handleEvent(self):
        for event in pygame.event.get():
            if (event.type==pygame.QUIT):
                self.gameOver()
            elif (event.type==pygame.KEYDOWN):
                key = pygame.key.get_pressed()
                if (key[pygame.K_LEFT]):
                    self.direction = Direction.LEFT
                elif (key[pygame.K_RIGHT]):
                    self.direction = Direction.RIGHT
                elif (key[pygame.K_UP]):
                    self.direction = Direction.UP
                elif (key[pygame.K_DOWN]):
                    self.direction = Direction.DOWN

    def createFood(self):
        if (len(self.foodList)<1):
            self.gameOver()
        else:
            self.food = self.foodList[0]
            self.foodList = self.foodList[1:]
            if (self.food in self.snake):
                self.createFood() # Recursion

    def collision(self):
        c1 = (self.snakeHead.x > self.width)
        c2 = (self.snakeHead.x < 1)
        c3 = (self.snakeHead.y > self.height)
        c4 = (self.snakeHead.y < 1)
        if (c1 or c2 or c3 or c4):
            return True
        
        if (self.snakeHead in self.snake[1:]):
            return True

        if (self.snakeHead in self.blockList):
            return True

        return False

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
        pygame.display.flip()

    def updateObject(self):
        game_end = False

        self.snakeMove()
        self.snake.insert(0,self.snakeHead)

        if (self.collision()):
            game_end = True
            return game_end, self.score

        if (self.snakeHead==self.food):
            self.score += 1
            self.createFood()
        else:
            self.snake.pop()

        return game_end, self.score

    def snakeMove(self):
        x = self.snakeHead.x
        y = self.snakeHead.y
        if (self.direction==Direction.UP):
            y -= 1
        elif (self.direction==Direction.DOWN):
            y += 1
        elif (self.direction==Direction.LEFT):
            x -= 1
        elif (self.direction==Direction.RIGHT):
            x += 1

        self.snakeHead = Point(x,y)

    def run(self):
        while (self.running):
            self.handleEvent()
            if (self.direction!=Direction.IDLE):
                game_end, score = self.updateObject()
                if (game_end):
                    print("Score: {}".format(score))
                    self.gameOver()
            self.drawUI()
            self.clock.tick(SPEED)

def main(args):
    testNum = args.test_num
    n, k, m, x0, y0, foodList, blockList = readInput(f'data/input{testNum}.txt')
    game = Game(n, n, foodList, blockList, x0, y0)
    game.run()    

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('test_num', help="Test case number (1 to 10)")
    args = parser.parse_args()
    main(args)