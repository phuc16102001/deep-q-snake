from GameAI import GameAI

def extractPoint(line):
    line = line.split()
    assert(len(line)==2)
    return int(line[0]), int(line[1])

def readOutput(fileName):
    fout = open(fileName,'r')
    lines = fout.readlines()
    points = []
    for line in lines:
        point = extractPoint(line)
        points.append(point)
    return points
    
def readInput(fileName):
    fin = open(fileName,'r')
    line = fin.readline()
    line = line.split()
    assert(len(line)==3)
    
    n = int(line[0])
    k = int(line[1])
    m = int(line[2])

    line = fin.readline()
    x0, y0 = extractPoint(line)

    foodList = []
    for _ in range(k):
        line = fin.readline()
        foodList.append(extractPoint(line))

    blockList = []
    for _ in range(m):
        line = fin.readline()
        blockList.append(extractPoint(line))

    fin.close()
    return n, k, m, x0, y0, foodList, blockList
