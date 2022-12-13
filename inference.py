from numpy import block
import AStar_agent
import agent as AI_agent
import torch
from GameAI import *
from reader import *
import os
import copy

def inferenceRL(n, foodList, blockList, x0, y0, fileName = None):
    if (fileName is not None):
        file = open(fileName,'w')

    agent = AI_agent.Agent()
    agent.model.load_state_dict(copy.deepcopy(torch.load(os.path.join('model','best.pth'))))

    game = GameAI(n,n,foodList,blockList,x0,y0)
    if (fileName is not None):
        file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    done = False
    score = 0
    while (done != True):
        state = agent.getState(game)
        final_move = [0,0,0]
        
        tensor_state = torch.tensor(state,dtype=torch.float)
        prediction = agent.model(tensor_state)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        reward, done, score = game.playStep(final_move)
        if (fileName is not None):
            file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    return score

# ========================= AStar

def inferenceAStar(n, foodList, blockList, x0, y0, fileName = None):
    if (fileName is not None):
        file = open(fileName,'w')
    
    game = GameAI(n,n,foodList,blockList,x0,y0)

    agent = AStar_agent.Agent()
    if (fileName is not None):
        file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    
    done = False
    score = 0
    while (done!=True):
        action = agent.getAction(game)
        if (action is None):
            done = True
            continue
        pos = action.pos()
        reward, done, score = game.playStep(Point(pos[0],pos[1]),TypeMove.ASTAR)
    
        if (fileName is not None):
            file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    return score

# ====================== BOTH ====================
            
def inferenceBoth(n, foodList, blockList, x0, y0, fileName = None):
    if (fileName is not None):
        file = open(fileName,'w')
    
    game = GameAI(n,n,foodList,blockList,x0,y0)

    agentAS = AStar_agent.Agent()
    agentRL = AI_agent.Agent()
    agentRL.model.load_state_dict(copy.deepcopy(torch.load(os.path.join('model','best.pth'))))

    if (fileName is not None):
        file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    
    done = False
    score = 0
    while (done!=True):
        actionAS = agentAS.getAction(game)
        state = agentRL.getState(game)
        if (actionAS is None):
            actionRL = agentRL.getAction(state)
            reward, done, score = game.playStep(actionRL)
        else:
            pos = actionAS.pos()
            reward, done, score = game.playStep(Point(pos[0],pos[1]),TypeMove.ASTAR)
    
        if (fileName is not None):
            file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    return score

# ============================= Random

           
def inferenceRand(n, foodList, blockList, x0, y0, fileName = None):
    if (fileName is not None):
        file = open(fileName,'w')
    
    game = GameAI(n,n,foodList,blockList,x0,y0)

    agentAS = AStar_agent.Agent()
    agentRL = AI_agent.Agent()
    agentRL.model.load_state_dict(copy.deepcopy(torch.load(os.path.join('model','best.pth'))))

    if (fileName is not None):
        file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    
    done = False
    score = 0
    while (done!=True):
        if (random.randint(0,2)==0):
            actionAS = agentAS.getAction(game)
            pos = actionAS.pos()
            reward, done, score = game.playStep(Point(pos[0],pos[1]),TypeMove.ASTAR)
        else:
            state = agentRL.getState(game)
            actionRL = agentRL.getAction(state)
            reward, done, score = game.playStep(actionRL)
    
        if (fileName is not None):
            file.write("{} {}\n".format(int(game.snakeHead.x),int(game.snakeHead.y)))
    return score

# ========================= Main test =============================

if __name__=="__main__":
    print("TEST\tAS\tRL\tBOTH\tRANDOM")
    scoreBOTH = 0
    while (scoreBOTH<=131):
        for testNumber in range(4,5):
            n, k, m, x0, y0, foodList, blockList = readInput('data/input{}.txt'.format(testNumber))
            # scoreAS = inferenceAStar(n,foodList,blockList,x0,y0,'Result/output_AS_{}.txt'.format(testNumber))
            # scoreRL = inferenceRL(n,foodList,blockList,x0,y0,'Result/output_RL_{}.txt'.format(testNumber))
            scoreBOTH = inferenceBoth(n,foodList,blockList,x0,y0,'output/output_BOTH_{}.txt'.format(testNumber))
            # scoreRAND = inferenceRand(n,foodList,blockList,x0,y0,'Result/output_RAND_{}.txt'.format(testNumber))
            print("{}\t{}\t{}\t{}\t{}".format(testNumber,0,0,scoreBOTH,0))