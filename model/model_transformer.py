import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(num_seeds, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = nn.MultiheadAttention(dim, num_heads)

    def forward(self, X, src_key_padding_mask=None):
        Q = self.S.repeat(1, X.size(1), 1)
        out, _ = self.mab(Q, X, X, key_padding_mask=src_key_padding_mask)
        return out

class Net_Snapshot(torch.nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=64, num_heads=4):
        super(Net_Snapshot, self).__init__()
#         self.enc = nn.Sequential(
#                 SAB(dim_input, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads))
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=4, dim_feedforward=2*dim_hidden, dropout=0.0)
        self.feat_in = nn.Sequential(
                        nn.Linear(dim_input, dim_hidden),
                    )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 nn.Linear(dim_hidden, dim_output))
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
                    nn.Linear(dim_hidden, dim_output)
                    )

    def forward(self, x, pad_mask=None):
        x = self.feat_in(x)
        x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.pool(x, src_key_padding_mask=pad_mask)
        x = self.dec(x)
        out = self.feat_out(x)
        return torch.squeeze(out, dim=0)


########################################################
# DeepSets (src: https://github.com/yassersouri/pytorch-deep-sets)
class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        # sum up the representations
        x = torch.sum(x, dim=0, keepdim=False)
        # compute the output
        out = self.rho.forward(x)
        return out

class SmallPhi(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepSetModel(nn.Module): # linear relu linear bottleneck
    def __init__(self, input_size: int, output_size: int = 1, hidden_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        phi = SmallPhi(self.input_size, self.hidden_size)
        rho = SmallPhi(self.hidden_size, self.output_size)
        self.net = InvariantModel(phi, rho)

    def forward(self, x, pad_mask=None):
        out = self.net.forward(x)
        return out

class CustomATN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomATN, self).__init__(observation_space, features_dim)
        dim_input=observation_space.shape[-1]
        num_outputs=1
        self.transformer = Net_Snapshot(dim_input, num_outputs, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # device = torch.device("cuda:0")
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        mask = torch.any(observations != 0, dim=2)
        x = observations[:,mask[0,:],:]
        #  the shape of x should be (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        x1 = self.transformer(x)
        return x1

class DeepSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(DeepSetExtractor, self).__init__(observation_space, features_dim)
        dim_input=observation_space.shape[-1]
        hidden_size=10
        self.transformer = DeepSetModel(dim_input, features_dim, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # device = torch.device("cuda:0")
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        mask = torch.any(observations != 0, dim=2)
        x = observations[:,mask[0,:],:]
        x = x.permute(1, 0, 2)
        x1 = self.transformer(x)
        return x1

class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        # Define the transformer network
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=observation_space.shape[-1], nhead=4),
            num_layers=2
        )

        # Define the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape observations to match the transformer input shape
        observations = observations.permute(1, 0, 2)
        features = self.transformer(observations)
        features = features[-1].flatten(start_dim=1)
        features = self.fc(features)
        return features

class CustomATN1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomATN1, self).__init__(observation_space, features_dim)
        dim_input=observation_space.shape[-1]
        num_outputs=1
        dim_hidden = 16
        num_heads = 4
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2*dim_hidden, dropout=0.0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2*dim_hidden, dropout=0.0)
        self.feat_in = nn.Sequential(
                        nn.Linear(dim_input, dim_hidden),
                    )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 nn.Linear(dim_hidden, dim_output))
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
                    nn.Linear(dim_hidden, features_dim)
                    )
        # self.transformer = Net_Snapshot(dim_input, num_outputs, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mask = torch.any(observations != 0, dim=2)
        x = observations[:,mask[0,:],:]
        #  the shape of x should be (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        batch_size = x.shape[1]
        # if batch_size>1:
        #     batchcnt=1
        # out1 = self.transformer(x)
        pad_mask = None
        x1 = self.feat_in(x)
        x2 = self.enc(x1, src_key_padding_mask=pad_mask)
        x31 = x2.permute(1, 0, 2).reshape(batch_size,-1)
        padsize=self.features_dim-x31.shape[1]
        pad = nn.ZeroPad2d(padding=(0, padsize, 0, 0))
        out = pad(x31)
        # x3 = self.pool(x2, src_key_padding_mask=pad_mask)
        # x4 = self.dec(x3)
        # out = torch.squeeze(self.feat_out(x4), dim=0)
        # out = torch.squeeze(x3, dim=0)
        return out
class CustomATN2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomATN2, self).__init__(observation_space, features_dim)
        dim_input=observation_space.shape[-1]
        num_outputs=1
        dim_hidden = 8
        num_heads = 4
        encoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2*dim_hidden, dropout=0.0)
        decoder_layer = nn.TransformerEncoderLayer(dim_hidden, nhead=num_heads, dim_feedforward=2*dim_hidden, dropout=0.0)
        self.feat_in = nn.Sequential(
                        nn.Linear(dim_input, dim_hidden),
                    )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.dec = nn.Sequential(
#                 PMA(dim_hidden, num_heads, num_outputs),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 SAB(dim_hidden, dim_hidden, num_heads),
#                 nn.Linear(dim_hidden, dim_output))
        self.pool = PMA(dim_hidden, num_heads, num_outputs)
        self.dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.feat_out = nn.Sequential(
                    nn.Linear(dim_hidden, features_dim)
                    )
        # self.transformer = Net_Snapshot(dim_input, num_outputs, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mask = torch.any(observations != 0, dim=2)
        x = observations[:,mask[0,:],:]
        #  the shape of x should be (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)
        batch_size = x.shape[1]
        # if batch_size>1:
        #     batchcnt=1
        # out1 = self.transformer(x)
        pad_mask = None
        x1 = self.feat_in(x)
        x2 = self.enc(x1, src_key_padding_mask=pad_mask)
        x31 = x2.permute(1, 0, 2).reshape(batch_size,-1)
        padsize=self.features_dim-x31.shape[1]
        pad = nn.ZeroPad2d(padding=(0, padsize, 0, 0))
        out = pad(x31)
        # x3 = self.pool(x2, src_key_padding_mask=pad_mask)
        # x4 = self.dec(x3)
        # out = torch.squeeze(self.feat_out(x4), dim=0)
        # out = torch.squeeze(x3, dim=0)
        return out