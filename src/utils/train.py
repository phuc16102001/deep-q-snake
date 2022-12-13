import sys
sys.path.insert(0,'.')

from argparse import ArgumentParser

import classes.RL_Agent as RL_Agent
from environment.GameAI import GameAI
import utils.plotter as plotter
from utils.reader import readInput

def train(
    n, 
    k, 
    m, 
    foodList, 
    blockList, 
    x0, 
    y0,
    output_path='models/current_model.pth'
):
    scores = []
    mean_scores = []
    total_scores = 0
    record = 0
    agent = RL_Agent.Agent()
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
                agent.model.save(output_path)
            print("{}\t{}\t{}".format(agent.nGame,score,record))

            scores.append(score)
            total_scores += score
            mean_scores.append(total_scores/agent.nGame)
            plotter.plot(scores, mean_scores)

def main(args):
    input_path = args.input_path
    output_path = args.output_path

    n, k, m, x0, y0, foodList, blockList = readInput(input_path)
    train(n, k, m, foodList, blockList, x0, y0, output_path)

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--input_path', 
        help='The path to input map used to train',
        required=True
    )
    parser.add_argument(
        '--output_path', 
        help='The path to output the model (.pth)',
        required=True
    )
    args = parser.parse_args()
    main(args)
    