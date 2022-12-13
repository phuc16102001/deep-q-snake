from test import *
from inference import *

for testNum in range(5,11):
    print("Running testcase {}... ".format(testNum),end="")
    left = 5
    highest = test(testNum)
    print(highest,end=" ")
    score = 0
    while (left>0 and score<=highest):
        n, k, m, x0, y0, foodList, blockList = readInput('Data/input{}.txt'.format(testNum))
        score = inferenceBoth(n,foodList,blockList,x0,y0,'Result/output_BOTH_{}.txt'.format(testNum))
        left -= 1
    print(score)
        