import math

class Node:
    #========================Attribute=========================
    x = None        #x-coordinate of the node
    y = None        #y-coordinate of the node
    g = 0           #g(node) is the cost from the start
    h = 0           #h(node) is the heuristic cost to goal
    f = 0           #f(node)=g(node)+h(node)
    parent = None   #parent node of the current node
    nodeType = None

    #========================Method=========================
    #Construction for node
    #Input: pos=Tuples(x,y); a=greyscale(float/int)
    def __init__(self,pos,nodeType):
        self.x=pos.x
        self.y=pos.y
        self.nodeType=nodeType

    #Cast the node to string
    #Output: string("x y a g h f")
    def __str__(self):
        return (f"{self.x}, {self.y}, {self.a}, {self.g}, {self.h}, {self.f}")

    #Get the euclid distance (diagonal distance) between the current to another node
    #Input: node = Node
    #Output: euclid distance from self to node = (int/float)
    def euclidDistance(self,node):
        dx = self.x-node.x
        dy = self.y-node.y
        d = math.sqrt(dx**2+dy**2)
        return d
    
    #Get the cost from the current when move to the parameter node
    #Input: node = Node
    #Output: the cost for move from current to the parameter node
    def distanceTo(self,node):
        return self.euclidDistance(node)

    #Calculate the f(self)
    def calculateF(self):
        self.f = self.g+self.h

    #Set the parent node of current node
    def setParent(self,node):
        self.parent = node 
        self.g = self.parent.g + self.parent.distanceTo(self)
        self.calculateF()

    #Set the heuristic value for current node
    def setH(self,h):
        self.h=h
        self.calculateF()

    #Get the current position
    #Output: Tuple(x,y)
    def pos(self):
        return (self.x, self.y)