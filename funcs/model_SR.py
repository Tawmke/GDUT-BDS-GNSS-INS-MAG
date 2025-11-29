import torch
import torch.nn as nn
from funcs.parameters import *
#Actor网络
class Actor(nn.Module):
    def __init__(self,N_S,N_A, actor_net):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(N_S,actor_net[0])
        self.fc2 = nn.Linear(actor_net[0],actor_net[1])
        self.sigma = nn.Linear(actor_net[1],N_A)
        self.mu = nn.Linear(actor_net[1],N_A)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        # self.set_init([self.fc1,self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal
    #初始化网络参数
    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))

        mu = self.mu(x)
        log_sigma = self.sigma(x)
        #log_sigma = torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return mu,sigma

    def choose_action(self,s):
        mu,sigma = self.forward(s)
        Pi = self.distribution(mu,sigma)
        action = Pi.sample().numpy()
        return action

#Critic网洛
class Critic(nn.Module):
    def __init__(self,N_S, critic_net):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(N_S,critic_net[0])
        self.fc2 = nn.Linear(critic_net[0],critic_net[1])
        self.fc3 = nn.Linear(critic_net[1],1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x1 = torch.tanh(self.fc1(s))
        x2 = torch.tanh(self.fc2(x1))
        values = self.fc3(x2)
        return values, x2, x1

class Criticorig(nn.Module):
    def __init__(self,N_S, critic_net):
        super(Criticorig,self).__init__()
        self.fc1 = nn.Linear(N_S,critic_net[0])
        self.fc2 = nn.Linear(critic_net[0],critic_net[1])
        self.fc3 = nn.Linear(critic_net[1],1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        # self.set_init([self.fc1, self.fc2, self.fc2])

    def set_init(self,layers):
        for layer in layers:
            nn.init.normal_(layer.weight,mean=0.,std=0.1)
            nn.init.constant_(layer.bias,0.)

    def forward(self,s):
        x1 = torch.tanh(self.fc1(s))
        x2 = torch.tanh(self.fc2(x1))
        values = self.fc3(x2)
        return values