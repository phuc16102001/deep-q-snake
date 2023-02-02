import sys
sys.path.insert(0,'.')

import torch
import os
import copy
from argparse import ArgumentParser

import classes.AS_Agent as AS_Agent
import classes.RL_Agent as RL_Agent
from environment.GameAI import *
from utils.reader import *

def writePosToFile(file, x, y):
    file.write(f"{int(x)} {int(y)}\n")

def inference(
    n, 
    foodList, 
    blockList, 
    x0, 
    y0, 
    strategy = 'both', 
    modelPath = None, 
    outputPath = None
):
    assert(strategy in ['rl', 'as', 'both', 'rand'])

    if (outputPath is not None):
        file = open(outputPath,'w')
    
    game = GameAI(n,n,foodList,blockList,x0,y0)
    agentAS = AS_Agent.Agent()
    
    if (strategy != 'as'):
        assert(modelPath is not None)
        agentRL = RL_Agent.Agent()
        agentRL.load(modelPath, verbose=False)

    if (outputPath is not None):
        writePosToFile(file, game.snakeHead.x, game.snakeHead.y)
    
    done = False
    score = 0
    while (done!=True):
        moveType = None
        movePos = None

        if (strategy=='as'):
            action = agentAS.getAction(game)
            movePos = action
            moveType = TypeMove.ASTAR

        elif (strategy=='rl'):
            state = agentRL.getState(game)
            movePos = agentRL.getAction(state)
            moveType = TypeMove.RL

        elif (strategy=='both'):
            actionAS = agentAS.getAction(game)
            if (actionAS is None):
                state = agentRL.getState(game)
                movePos = agentRL.getAction(state)
                moveType = TypeMove.RL
            else:
                movePos = actionAS
                moveType = TypeMove.ASTAR
    
        elif (strategy=='rand'):
            if (random.randint(0,2)==0):
                actionAS = agentAS.getAction(game)
                movePos = actionAS
                moveType = TypeMove.ASTAR
            else:
                state = agentRL.getState(game)
                movePos = agentRL.getAction(state)
                moveType = TypeMove.RL

        if (movePos is None): 
            done = True
            continue
        _, done, score = game.playStep(movePos, moveType)

        if (outputPath is not None):
            writePosToFile(file, game.snakeHead.x, game.snakeHead.y)
    return score
    
def main(args):
    from_test = int(args.from_test)
    to_test = int(args.to_test)
    data_folder = args.data_folder
    output_folder = args.output_folder
    model_path = args.model_path

    lsStrategy = ['as','rl','both','rand']
    lsScore = [0, 0, 0, 0]
    print("TEST\tAS\tRL\tBOTH\tRANDOM")
    for testNumber in range(from_test, to_test+1):
        n, k, m, x0, y0, foodList, blockList = readInput(f'{data_folder}/input{testNumber}.txt')
        for i in range(len(lsStrategy)):
            lsScore[i] = inference(
                n, 
                foodList, 
                blockList, 
                x0, 
                y0, 
                strategy=lsStrategy[i],
                outputPath=f'{output_folder}/output_{lsStrategy[i]}_{testNumber}.txt',
                modelPath=model_path
            )
        print(f"{testNumber}\t{lsScore[0]}\t{lsScore[1]}\t{lsScore[2]}\t{lsScore[3]}")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--from_test',
        '-f',
        help='The starting test number (1 to 10)',
        required=True
    )
    parser.add_argument(
        '--to_test',
        '-t',
        help='The ending test number (1 to 10)',
        required=True
    )
    parser.add_argument(
        '--data_folder',
        '-d',
        help='The path to data folder',
        required=True
    )
    parser.add_argument(
        '--output_folder',
        '-o',
        help='The path to output folder',
        required=True
    )
    parser.add_argument(
        '--model_path',
        '-m',
        help='The path RL model',
        required=True
    )
    args = parser.parse_args()
    main(args)