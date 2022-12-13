from reader import *
from Game import Game

n, k, m, x0, y0, foodList, blockList = readInput('Data/input2.txt')
game = Game(n,n,foodList,blockList,x0,y0)
game.run()