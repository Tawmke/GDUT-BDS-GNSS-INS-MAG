import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Attention(nn.Module):
    def __init__(self, num_features):
        super(Attention, self).__init__()
        self.query = nn.Linear(num_features, 1, bias=False)
        self.attwts = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        # query from (seq_len,feature_dim) to (seq_len, 1)
        # softmax return a total 1 for each seq input (seq_len, 1)
        attwts = F.softmax(self.query(x), dim=0)
        self.attwts.weight.data=torch.mean(attwts)
        return attwts

class Attention1(nn.Module):
    def __init__(self, num_features):
        super(Attention1, self).__init__()
        self.query = nn.Linear(num_features, 1, bias=False)
        self.attwts = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        # attwts = F.softmax(self.query(x), dim=0)
        attwts = torch.sigmoid(self.query(x))
        self.attwts.weight.data=torch.mean(attwts)
        return attwts

class SelfAttention(nn.Module):
    def __init__(self, num_features, hidden_size=64):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(num_features, hidden_size)
        self.key = nn.Linear(num_features, hidden_size)
        self.value = nn.Linear(num_features, hidden_size)
        self.attwts = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = F.softmax(scores / math.sqrt(query.shape[-1]), dim=-1) # math is faster than np
        self.attwts.weight.data = torch.mean(weights)
        output = torch.matmul(weights, value)
        return output

class SelfAttentionW(nn.Module):
    def __init__(self, num_features, hidden_size=32):
        super(SelfAttentionW, self).__init__()
        self.query = nn.Linear(num_features, hidden_size)
        self.key = nn.Linear(num_features, hidden_size)
        self.value = nn.Linear(num_features, hidden_size)
        self.attwts = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        # value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = F.softmax(scores / math.sqrt(query.shape[-1]), dim=-1) # math is faster than np
        # output = torch.matmul(weights, value)
        self.attwts.weight.data = torch.mean(weights)
        return weights

class SelfAttentionW1(nn.Module):
    def __init__(self, num_features, hidden_size=32):
        super(SelfAttentionW1, self).__init__()
        self.query = nn.Linear(num_features, hidden_size)
        self.key = nn.Linear(num_features, hidden_size)
        self.value = nn.Linear(num_features, hidden_size)
        self.attwts = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        # value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        weights = torch.sigmoid(scores / np.sqrt(query.shape[-1])) # math is faster than np, but math have warning
        # output = torch.matmul(weights, value)
        self.attwts.weight.data = torch.mean(weights)
        return weights

class CustomATF(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomATF, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        self.attention1 = Attention(dim_input1)
        self.attention2 = Attention(dim_input2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w1=self.attention1(observations['gnss'])
        w2=self.attention2(observations['pos'])
        out=torch.cat((observations['gnss']*w1,observations['pos']*w2),dim=-1)
        return out

class CustomATF1_gnssW(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #注意力自适应权重
        super(CustomATF1_gnssW, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["weight"].shape[-1]
        self.attention1 = Attention1(dim_input1)
        self.attention2 = Attention1(dim_input2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w1=self.attention1(observations['gnss'])
        w2=self.attention2(observations['weight'])
        out=torch.cat((observations['gnss']*w1,observations['weight']*w2),dim=-1)
        return out

