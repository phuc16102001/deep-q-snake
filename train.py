from agent import *

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

# n, k, m, x0, y0, foodList, blockList = readInput('Data/input1.txt')
foodList = None
blockList = None
n = 20
k = -1
m = -1
x0 = 10
y0 = 10
train(n, k, m, foodList, blockList, x0, y0)