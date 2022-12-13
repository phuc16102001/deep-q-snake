from classes.MyNode import *
from environment.GameAI import Point, BlockType

#=============================Heuristic function=============================
#Manhattan
def heuristic(pos,goal):
    (px,py)=pos.pos()
    (gx,gy)=goal.pos()
    return abs(px-gx)+abs(py-gy)

#=============================File function==================================

#Convert from format "(x;y)" to tuple(x,y)
#Input: raw = String in format "(x;y)"
#Output: Tuple(x,y)
def convertPoint(raw):
    raw = raw[1:len(raw)-1]
    raw = raw.split(';')
    for i in range(len(raw)):
        raw[i] = int(raw[i])
    return tuple(raw)

#=============================Utils function===================================
#The priority function to create heap
#Input: node = Node
#Output: Value of priority
def priority(node):
    return node.f

#Get all the neighboor node (adjacent, |deltA|<=m)
#Input: node = Node
#Output: list of Node 
def getNeighboor(nodeMat, node, game):
    dx = [-1, 0,0,1]
    dy = [ 0,-1,1,0]
    x = node.x
    y = node.y
    height = len(nodeMat)
    width = len(nodeMat[0])

    result = []
    for i in range(len(dx)):
        newX = x+dx[i]
        newY = y+dy[i]
        nodeType = game.collision(Point(newX,newY))
        if (not(newX<=0 or newY<=0 or newX>width or newY>height) and nodeType!=BlockType.WALL and nodeType!=BlockType.SNAKE):
            neighboor = nodeMat[newY][newX]
            result.append(neighboor)
    return result

#Create nodes matrix from an image
#Input: img = MyImg
#Output: list of list Node (matrix Node)
def createNode(game):
    nodeMat = []
    start = None
    end = None
    
    padding = [-1]*(game.width+2)
    nodeMat.append(padding)

    for y in range(1,game.height+1):
        nodeRow = [-1]
        for x in range(1,game.width+1):
            pos = Point(x,y)
            nodeType = game.collision(pos)
            node = Node(pos,nodeType)
            if (pos == game.food):
                start = node
            if (pos == game.snakeHead):
                end = node
            nodeRow.append(node)
        nodeMat.append(nodeRow)
    nodeMat.append(padding)

    return nodeMat, start, end

#Add a node to the queue
#Input: queue, parent, node, goal, hFunction = set, Node, Node, Node, Pointer to a float-return function
#Description: Add [node] to [queue] whose [parent] and [goal] with the heuristic function [hFunction]
def addNode(queue,parent,node,goal,hFunction):
    if (node in queue):
        oldG = node.g
        newG = parent.g+parent.distanceTo(node)
        if (newG<oldG):
            node.setParent(parent)
    else:
        node.setParent(parent)
        node.setH(hFunction(node,goal))
        queue.add(node)

#Running function
#Input: img, hFunc, startPos, endPos, m = MyImage, Pointer to float-return function, tuples, tuples. int
#Output: path, cost, nTouchNode
def run(game, hFunc):
    #Explored and queue set
    explored = set()
    queue = set()
    touch = set()

    #Copy into a new image
    nodeMat, startNode, goalNode = createNode(game)
    
    #Add the startNode to queue
    addNode(queue,startNode,startNode,goalNode,hFunc)
    touch.add(startNode)

    #If the queue is not empty
    while (len(queue)>0):
        #Pop the highest priority out (lowest Node.f)
        node = min(queue,key=priority)

        #If we found the goal (end searching)
        if (node==goalNode):
            #Traceback
            result = []
            while (node!=startNode):
                result.append(node)
                node=node.parent
            result.append(startNode)
            result = result[::-1]
            return (result,result[-1].g,touch)
        
        #Not goal
        queue.remove(node)                                              #Remove the poped
        explored.add(node)                                              #Add to explored set
        neighboor = getNeighboor(nodeMat,node,game)                        #Get list of node that can move
        for i in range(len(neighboor)):                                 #For each node
            if not(neighboor[i] in explored):                           #Not in explored      
                addNode(queue,node,neighboor[i],goalNode,hFunc)         #Add that neighboor to queue if it better
                touch.add(neighboor[i])
    return None, None, None

class Agent:
    def getAction(self,game):
        path, cost, touch = run(game, heuristic)
        if (path!=None):
            point = path[-2].pos()
            return Point(point[0], point[1])
        return None