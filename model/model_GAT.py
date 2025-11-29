import gym
import numpy as np
import torch
import torch as th
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, TransformerConv, BatchNorm
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, num_heads: int = 4):
        super().__init__(observation_space, features_dim=64)

        if not hasattr(observation_space, 'graph'):
            raise ValueError("GATFeatureExtractor requires an environment with a graph attribute")

        num_nodes = observation_space.graph.num_nodes
        node_dim = observation_space.graph.node_attr_dim

        self.conv1 = GATConv(node_dim, 32, heads=num_heads)
        self.conv2 = GATConv(32 * num_heads, 64, heads=num_heads)

        self.mlp = nn.Sequential(
            nn.Linear(num_nodes * 64, 256),
            nn.ReLU(),
            nn.Linear(256, self.features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x, edge_index = observations.graph.x, observations.graph.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        features = self.mlp(x)

        return features

class CustomGAT(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGAT, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.num_heads = 4
        self.conv1 = GATConv(num_features, 16, heads=self.num_heads)
        self.conv2 = GATConv(16*self.num_heads, 32, heads=self.num_heads)
        self.fc1 = nn.Linear(self.num_heads * 32, 256)
        # self.graph_network = GNNetwork(num_features, features_dim)
        # self.gat=GraphAttentionNetwork(observation_space["gnss"].shape[-1], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = torch.device("cuda:0")
        # batch_len=len(observations['visible_satnum'])
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # edge_size=np.int(observations['edge_size'].cpu().numpy().reshape([1,])[0])
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        # edge_index=torch.tensor((observations['edge_index'].cpu().numpy()[0,:,:edge_size]))\
        #     .type(torch.LongTensor).to(device)

        # padding zero for batch input
        batch_size=len(observations['visible_satnum'])
        if batch_size==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            mask[0,0]=True # all zeros input should output zero feature added 0306
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
            # rollout_buffer has all zeros observations
            mask_x[:,0]=True # all zeros input should output zero feature added 0306
            x = [observations['gnss'][i, mask_x[i, :], :] for i in range(batch_size)]
            num_nodes=[x[i].shape[0] for i in range(batch_size)]

            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            # edge_index_o = [observations['edge_index'][i, :, mask_e[i, :]] for i in range(batch_size)]
            edge_index = [observations['edge_index'][i, :, mask_e[i, :]]+sum(num_nodes[:i])
                          for i in range(batch_size)]
            # create batch index tensor
            batch_idx = torch.cat([torch.full((num_nodes[i],), i) for i in range(batch_size)]).to(device)
            # flatten inputs and edge index for batch processing
            x1 = torch.cat(x, dim=0).to(device)
            edge_index1 = torch.cat(edge_index, dim=1).type(torch.LongTensor).to(device)

        # x2 = self.graph_network(x1,edge_index1)
        x2 = self.conv1(x1, edge_index1)
        x2 = F.relu(x2)
        x3 = self.conv2(x2, edge_index1)
        x3 = F.relu(x3)
        x4 = self.fc1(x3)
        out = global_mean_pool (x4, batch_idx)
        return out

class CustomGAT1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGAT1, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.num_heads = 2
        self.conv1 = GATConv(num_features, 16, heads=self.num_heads)
        self.conv2 = GATConv(16*self.num_heads, 32, heads=self.num_heads)
        self.fc1 = nn.Linear(self.num_heads * 32, features_dim)
        # self.graph_network = GNNetwork(num_features, features_dim)
        # self.gat=GraphAttentionNetwork(observation_space["gnss"].shape[-1], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = torch.device("cuda:0")
        # batch_len=len(observations['visible_satnum'])
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # edge_size=np.int(observations['edge_size'].cpu().numpy().reshape([1,])[0])
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        # edge_index=torch.tensor((observations['edge_index'].cpu().numpy()[0,:,:edge_size]))\
        #     .type(torch.LongTensor).to(device)

        # padding zero for batch input
        batch_size=len(observations['visible_satnum'])
        if batch_size==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            mask[0,0]=True # all zeros input should output zero feature added 0306
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
            # rollout_buffer has all zeros observations
            mask_x[:,0]=True # all zeros input should output zero feature added 0306
            x = [observations['gnss'][i, mask_x[i, :], :] for i in range(batch_size)]
            num_nodes=[x[i].shape[0] for i in range(batch_size)]

            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            # edge_index_o = [observations['edge_index'][i, :, mask_e[i, :]] for i in range(batch_size)]
            edge_index = [observations['edge_index'][i, :, mask_e[i, :]]+sum(num_nodes[:i])
                          for i in range(batch_size)]
            # create batch index tensor
            batch_idx = torch.cat([torch.full((num_nodes[i],), i) for i in range(batch_size)]).to(device)
            # flatten inputs and edge index for batch processing
            x1 = torch.cat(x, dim=0).to(device)
            edge_index1 = torch.cat(edge_index, dim=1).type(torch.LongTensor).to(device)

        # x2 = self.graph_network(x1,edge_index1)
        x2 = self.conv1(x1, edge_index1)
        x2 = F.relu(x2)
        x3 = self.conv2(x2, edge_index1)
        x3 = F.relu(x3)
        x4 = self.fc1(x3)
        out = global_mean_pool (x4, batch_idx)
        return out

class CustomGTNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGTNN, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.num_heads = 4
        self.fc = nn.Linear(num_features, 16)
        self.conv1 = TransformerConv(16, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn1 = BatchNorm(32)
        self.conv2 = TransformerConv(32, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn2 = BatchNorm(32)
        self.conv3 = TransformerConv(32, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn3 = BatchNorm(32)
        self.conv4 = TransformerConv(64, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn4 = BatchNorm(32)
        self.conv5 = TransformerConv(64, features_dim, heads=1, concat=True, beta=False, dropout=0.)
        # self.graph_network = GNNetwork(num_features, features_dim)
        # self.gat=GraphAttentionNetwork(observation_space["gnss"].shape[-1], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = torch.device("cuda:0")
        # batch_len=len(observations['visible_satnum'])
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # edge_size=np.int(observations['edge_size'].cpu().numpy().reshape([1,])[0])
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        # edge_index=torch.tensor((observations['edge_index'].cpu().numpy()[0,:,:edge_size]))\
        #     .type(torch.LongTensor).to(device)

        # padding zero for batch input
        batch_size=len(observations['visible_satnum'])
        if batch_size==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            mask[0,0]=True # all zeros input should output zero feature added 0306
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
            # rollout_buffer has all zeros observations
            mask_x[:,0]=True # all zeros input should output zero feature added 0306
            x = [observations['gnss'][i, mask_x[i, :], :] for i in range(batch_size)]
            num_nodes=[x[i].shape[0] for i in range(batch_size)]

            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            # edge_index_o = [observations['edge_index'][i, :, mask_e[i, :]] for i in range(batch_size)]
            edge_index = [observations['edge_index'][i, :, mask_e[i, :]]+sum(num_nodes[:i])
                          for i in range(batch_size)]
            # create batch index tensor
            batch_idx = torch.cat([torch.full((num_nodes[i],), i) for i in range(batch_size)]).to(device)
            # flatten inputs and edge index for batch processing
            x1 = torch.cat(x, dim=0).to(device)
            edge_index1 = torch.cat(edge_index, dim=1).type(torch.LongTensor).to(device)


        x1 = self.fc(x1)
        x1 = self.conv1(x1, edge_index1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index1)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(torch.cat([x1, x3], -1), edge_index1)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        x5 = self.conv5(torch.cat([x2, x4], -1), edge_index1)
        out = global_mean_pool (x5, batch_idx)
        return out

class CustomGTNN1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGTNN1, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.num_heads = 4
        self.fc = nn.Linear(num_features, 16)
        self.conv1 = TransformerConv(16, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn1 = BatchNorm(32)
        self.conv2 = TransformerConv(32, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn2 = BatchNorm(32)
        self.conv3 = TransformerConv(32, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn3 = BatchNorm(32)
        self.conv4 = TransformerConv(64, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn4 = BatchNorm(32)
        self.conv5 = TransformerConv(64, features_dim, heads=1, concat=True, beta=False, dropout=0.)
        # self.graph_network = GNNetwork(num_features, features_dim)
        # self.gat=GraphAttentionNetwork(observation_space["gnss"].shape[-1], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = torch.device("cuda:0")
        # batch_len=len(observations['visible_satnum'])
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # edge_size=np.int(observations['edge_size'].cpu().numpy().reshape([1,])[0])
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        # edge_index=torch.tensor((observations['edge_index'].cpu().numpy()[0,:,:edge_size]))\
        #     .type(torch.LongTensor).to(device)

        # padding zero for batch input
        batch_size=len(observations['visible_satnum'])
        if batch_size==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            mask[0,0]=True # all zeros input should output zero feature added 0306
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
            # rollout_buffer has all zeros observations
            mask_x[:,0]=True # all zeros input should output zero feature added 0306
            x = [observations['gnss'][i, mask_x[i, :], :] for i in range(batch_size)]
            num_nodes=[x[i].shape[0] for i in range(batch_size)]

            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            # edge_index_o = [observations['edge_index'][i, :, mask_e[i, :]] for i in range(batch_size)]
            edge_index = [observations['edge_index'][i, :, mask_e[i, :]]+sum(num_nodes[:i])
                          for i in range(batch_size)]
            # create batch index tensor
            batch_idx = torch.cat([torch.full((num_nodes[i],), i) for i in range(batch_size)]).to(device)
            # flatten inputs and edge index for batch processing
            x1 = torch.cat(x, dim=0).to(device)
            edge_index1 = torch.cat(edge_index, dim=1).type(torch.LongTensor).to(device)


        x1 = self.fc(x1)
        x1 = self.conv1(x1, edge_index1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index1)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(torch.cat([x1, x3], -1), edge_index1)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        out = self.conv5(torch.cat([x2, x4], -1), edge_index1)
        # seems not applicable because of policies.evaluate_actions
        # log_prob = distribution.log_prob(actions)
        # where actions scale do not match feature and latent dimension
        # out = global_mean_pool (x5, batch_idx)
        return out

class CustomGTNNcat(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGTNN, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.fc = nn.Linear(num_features, 16)
        self.conv1 = TransformerConv(16, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn1 = BatchNorm(32)
        self.conv2 = TransformerConv(32, 16, heads=2, concat=True, beta=False, dropout=0.3)
        self.bn2 = BatchNorm(32)
        self.conv3 = TransformerConv(32, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn3 = BatchNorm(32)
        self.conv4 = TransformerConv(64, 32, heads=1, concat=True, beta=False, dropout=0.3)
        self.bn4 = BatchNorm(32)
        self.hiddensize = 16
        self.conv5 = TransformerConv(64, self.hiddensize, heads=1, concat=True, beta=False, dropout=0.)
        # self.graph_network = GNNetwork(num_features, features_dim)
        # self.gat=GraphAttentionNetwork(observation_space["gnss"].shape[-1], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = torch.device("cuda:0")
        # batch_len=len(observations['visible_satnum'])
        # visible_satnum=np.int(observations['visible_satnum'].cpu().numpy().reshape([1,])[0])
        # edge_size=np.int(observations['edge_size'].cpu().numpy().reshape([1,])[0])
        # x=torch.tensor(observations['gnss'].cpu().numpy()[0,:visible_satnum,:]).to(device)
        # edge_index=torch.tensor((observations['edge_index'].cpu().numpy()[0,:,:edge_size]))\
        #     .type(torch.LongTensor).to(device)

        # padding zero for batch input
        batch_size=len(observations['visible_satnum'])
        if batch_size==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            mask[0,0]=True # all zeros input should output zero feature added 0306
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
            # rollout_buffer has all zeros observations
            mask_x[:,0]=True # all zeros input should output zero feature added 0306
            x = [observations['gnss'][i, mask_x[i, :], :] for i in range(batch_size)]
            num_nodes=[x[i].shape[0] for i in range(batch_size)]

            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            # edge_index_o = [observations['edge_index'][i, :, mask_e[i, :]] for i in range(batch_size)]
            edge_index = [observations['edge_index'][i, :, mask_e[i, :]]+sum(num_nodes[:i])
                          for i in range(batch_size)]
            # create batch index tensor
            batch_idx = torch.cat([torch.full((num_nodes[i],), i) for i in range(batch_size)]).to(device)
            # flatten inputs and edge index for batch processing
            x1 = torch.cat(x, dim=0).to(device)
            edge_index1 = torch.cat(edge_index, dim=1).type(torch.LongTensor).to(device)


        x1 = self.fc(x1)
        x1 = self.conv1(x1, edge_index1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index1)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(torch.cat([x1, x3], -1), edge_index1)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)

        xx = self.conv5(torch.cat([x2, x4], -1), edge_index1)

        batch_size = len(torch.unique(batch_idx))
        max_nodes_num = np.int(self.features_dim/self.hiddensize)
        new_shape = (batch_size, max_nodes_num*xx.shape[-1])
        new_x = torch.zeros(new_shape, dtype=xx.dtype, device=xx.device)
        # xx1 = [ [,torch.zeros(features_dim, dtype=xx.dtype, device=xx.device)] for i in range(batch_size)]
        for i in range(batch_size):
            xtmp=xx[batch_idx==i,:].reshape(1,-1)
            new_x[i,:xtmp.shape[-1]]=xtmp
        # out = global_mean_pool (x5, batch_idx)
        return new_x