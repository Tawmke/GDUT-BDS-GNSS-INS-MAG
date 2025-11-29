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

class CustomATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #注意力自适应权重
        super(CustomATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        self.attention1 = Attention1(dim_input1)
        self.attention2 = Attention1(dim_input2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w1=self.attention1(observations['gnss'])
        w2=self.attention2(observations['pos'])
        out=torch.cat((observations['gnss']*w1,observations['pos']*w2),dim=-1)
        return out

class CustomATF1_AKFRL(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256): #注意力自适应权重
        super(CustomATF1_AKFRL, self).__init__(observation_space, features_dim)
        dim_input1 = observation_space["gnss"].shape[-1]
        dim_input2 = observation_space["pos"].shape[-1]
        dim_input3 = observation_space["Q_noise"].shape[-1]
        dim_input4 = observation_space["R_noise"].shape[-1]
        self.attention1 = Attention1(dim_input1)
        self.attention2 = Attention1(dim_input2)
        self.attention3 = Attention1(dim_input3)
        self.attention4 = Attention1(dim_input4)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w1 = self.attention1(observations['gnss'])
        w2 = self.attention2(observations['pos'])
        w3 = self.attention3(observations['Q_noise'])
        w4 = self.attention4(observations['R_noise'])
        out=torch.cat((observations['gnss']*w1,observations['pos']*w2,
                       observations['Q_noise']*w3,observations['R_noise']*w4),dim=-1)
        return out

class CustomsATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomsATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.fc1=nn.Sequential(nn.Linear(dim_input1,hidden_size),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(dim_input2,hidden_size),nn.ReLU())
        self.attention1 = Attention1(hidden_size)
        self.attwts1 = nn.Linear(1, 1, bias=False)
        self.attwts2 = nn.Linear(1, 1, bias=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x1=self.fc1(observations['gnss'])
        x2=self.fc2(observations['pos'])
        w1=self.attention1(x1)
        w2=self.attention1(x2)
        self.attwts1.weight.data = torch.mean(w1)
        self.attwts2.weight.data = torch.mean(w2)
        out=torch.cat((torch.matmul(w1,observations['gnss']),
                       torch.matmul(w2,observations['pos'])),dim=-1)
        return out

class CustomSATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomSATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        self.attention11 = SelfAttentionW1(dim_input1)
        self.attention12 = SelfAttentionW1(dim_input2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos'])),dim=-1)
        return out

class CustomSATFre1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomSATFre1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        self.attention1 = SelfAttentionW1(dim_input1)
        self.attention2 = SelfAttentionW1(dim_input2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention1(observations['gnss'])
        w12=self.attention2(observations['pos'])
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos'])),dim=-1)
        return out

class CustomsSATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomsSATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.fc1=nn.Sequential(nn.Linear(dim_input1,hidden_size),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(dim_input2,hidden_size),nn.ReLU())
        self.attention1 = SelfAttentionW1(hidden_size)
        self.attwts1 = nn.Linear(1, 1, bias=False)
        self.attwts2 = nn.Linear(1, 1, bias=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x1=self.fc1(observations['gnss'])
        x2=self.fc2(observations['pos'])
        w1=self.attention1(x1)
        w2=self.attention1(x2)
        self.attwts1.weight.data = torch.mean(w1)
        self.attwts2.weight.data = torch.mean(w2)
        out=torch.cat((torch.matmul(w1,observations['gnss']),
                       torch.matmul(w2,observations['pos'])),dim=-1)
        return out

class CustomCSATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCSATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.attention11 = SelfAttentionW1(dim_input1)
        self.attention12 = SelfAttentionW1(dim_input2)
        self.attention13 = SelfAttentionW1(hidden_size)
        self.fc3=nn.Sequential(nn.Linear(dim_input1+dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        x31=torch.cat((observations['gnss'],observations['pos']),dim=-1)
        x32=self.fc3(x31)
        w3=self.attention13(x32)
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos']),
                       torch.matmul(w3,x32)),dim=-1)
        return out

class CustomCoSATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCoSATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.attention11 = SelfAttentionW1(dim_input1)
        self.attention12 = SelfAttentionW1(dim_input2)
        self.fc31=nn.Sequential(nn.Linear(dim_input1,hidden_size),nn.ReLU())
        self.fc32=nn.Sequential(nn.Linear(dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        wall=torch.cat((w11,w12),dim=-1)
        walls=torch.softmax(wall, dim=-1)
        x31=self.fc31(observations['gnss'])
        x32=self.fc32(observations['pos'])
        out=torch.cat((observations['gnss'],observations['pos'],
                       torch.matmul(walls[:,:,:1],x31)+
                       torch.matmul(walls[:,:,1:],x32)),dim=-1)
        return out

class CustomCATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.attention11 = Attention1(dim_input1)
        self.attention12 = Attention1(dim_input2)
        self.attention13 = Attention1(hidden_size)
        self.fc3=nn.Sequential(nn.Linear(dim_input1+dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        x31=torch.cat((observations['gnss'],observations['pos']),dim=-1)
        x32=self.fc3(x31)
        w3=self.attention13(x32)
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos']),
                       torch.matmul(w3,x32)),dim=-1)
        return out

class CustomCoATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCoATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=64
        self.attention11 = Attention1(dim_input1)
        self.attention12 = Attention1(dim_input2)
        self.fc31=nn.Sequential(nn.Linear(dim_input1,hidden_size),nn.ReLU())
        self.fc32=nn.Sequential(nn.Linear(dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        wall=torch.cat((w11,w12),dim=-1)
        walls=torch.softmax(wall, dim=-1)
        x31=self.fc31(observations['gnss'])
        x32=self.fc32(observations['pos'])
        out=torch.cat((observations['gnss'],observations['pos'],
                       torch.matmul(walls[:,:,:1],x31)+
                       torch.matmul(walls[:,:,1:],x32)),dim=-1)
        return out

class CustomCfATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCfATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=32
        self.attention11 = Attention1(dim_input1)
        self.attention12 = Attention1(dim_input2)
        self.fc3=nn.Sequential(nn.Linear(dim_input1+dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        x31=torch.cat((observations['gnss'],observations['pos']),dim=-1)
        x32=self.fc3(x31)
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos']),
                       x32),dim=-1)
        return out

class CustomCpATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCpATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size=32
        self.attention11 = Attention1(dim_input1)
        self.attention12 = Attention1(dim_input2)
        self.attention13 = Attention1(hidden_size)
        self.attention14 = Attention1(hidden_size)
        self.fc1=nn.Sequential(nn.Linear(dim_input1,hidden_size),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(dim_input2,hidden_size),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        w11=self.attention11(observations['gnss'])
        w12=self.attention12(observations['pos'])
        x21=self.fc1(observations['gnss'])
        x22=self.fc2(observations['pos'])
        w13=self.attention13(x21)
        w14=self.attention14(x22)
        out=torch.cat((torch.matmul(w11,observations['gnss']),
                       torch.matmul(w12,observations['pos']),
                       torch.matmul(w13,x21)+torch.matmul(w14,x22)),dim=-1)
        return out

class CustomFATF1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomFATF1, self).__init__(observation_space, features_dim)
        dim_input1=observation_space["gnss"].shape[-1]
        dim_input2=observation_space["pos"].shape[-1]
        hidden_size1=64
        hidden_size2=32
        self.attention13 = Attention1(hidden_size1)
        self.attention14 = Attention1(hidden_size2)
        self.fc1=nn.Sequential(nn.Linear(dim_input1,hidden_size1),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(dim_input2,hidden_size2),nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x21=self.fc1(observations['gnss'])
        x22=self.fc2(observations['pos'])
        w13=self.attention13(x21)
        w14=self.attention14(x22)
        out=torch.cat((torch.matmul(w13,x21),torch.matmul(w14,x22)),dim=-1)
        return out