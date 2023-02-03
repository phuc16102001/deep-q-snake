import random

def gen_map(n, k, m):
    x0 = random.randint(1, n)
    y0 = random.randint(1, n)
    initPoint = (x0, y0)

    blockList = []
    while (len(blockList)<m):
        x = random.randint(1, n)
        y = random.randint(1, n)
        block = (x, y)
        c1 = (block not in blockList)
        c2 = (block != initPoint)
        if (c1 and c2): blockList.append(block)

    foodList = []
    while (len(foodList)<k):
        x = random.randint(1, n)
        y = random.randint(1, n)
        food = (x, y)
        c1 = (food not in blockList)
        c2 = (food != initPoint)
        if (c1 and c2): foodList.append(food)
    
    return n, k, m, x0, y0, foodList, blockList