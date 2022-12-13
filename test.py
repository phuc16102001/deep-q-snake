from GameAI import *
from reader import *

def test(testNum):
    n, k, m, x0, y0, foodList, blockList = readInput('Data/input{}.txt'.format(testNum))
    points = readOutput('Result/output{}.txt'.format(testNum))
    game = GameAI(n, n, foodList, blockList, x0, y0)
    
    done = False
    score = 0
    k = 1
    for j in range(1,len(points)):
        reward, done, score = game.playStep(Point(points[j][0],points[j][1]),TypeMove.ASTAR)
    return score

if __name__=="__main__":
    totalScore = 0
    for testNum in range(1,11):
        score = test(testNum)
        print("Testcase {}: {}".format(testNum,score))
        totalScore += score
    print("Total score:",totalScore)