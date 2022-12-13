from re import L
from black import out
import torch 
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as f
import os

class Linear_QNet(nn.Module):

    def __init__(self,input_size,hidden_size1,output_size):
    # def __init__(self,input_size,hidden_size1,hidden_size2,hidden_size3,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,output_size)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
    def save(self, fileName='model.pth'):
        modelFolder = 'model'
        if (not os.path.exists(modelFolder)):
            os.makedirs(modelFolder)
        fileName = os.path.join(modelFolder,fileName)
        torch.save(self.state_dict(), fileName)

class QTrainer:
    lr = None
    model = None
    gamma = None
    optimizer = None

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.optimizer = opt.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)

        if (len(state.shape)==1):
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
