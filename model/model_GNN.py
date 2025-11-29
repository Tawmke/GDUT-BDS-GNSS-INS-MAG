import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TransformerConv
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import numpy as np

class GCNFeatureExtractor(nn.Module):
    """
    Graph Convolutional Network (GCN)-based feature extractor with dictionary inputs for PPO.
    """

    def __init__(self, observation_space, gcn_hidden_size=64, gcn_num_layers=2):
        super().__init__()

        self.observation_space = observation_space

        # Compute the input dimensionality of the GCN
        self.input_dim = get_flattened_obs_dim(observation_space)

        # Create the GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(gcn_num_layers):
            input_size = self.input_dim if i == 0 else gcn_hidden_size
            output_size = gcn_hidden_size
            self.gcn_layers.append(GCNConv(input_size, output_size))

        # Create the output layer
        self.output_layer = nn.Linear(gcn_hidden_size, gcn_hidden_size)

    def forward(self, observations):
        # Convert the dictionary observations to a tensor
        obs_tensor = torch.cat([observations[key] for key in observations], dim=-1)

        # Reshape the observations to (batch_size, num_nodes, input_dim)
        batch_size = obs_tensor.shape[0]
        num_nodes = 1  # Since this is a single-agent environment
        obs_tensor = obs_tensor.view(batch_size, num_nodes, self.input_dim)

        # Apply the GCN layers
        for gcn_layer in self.gcn_layers:
            obs_tensor = gcn_layer(obs_tensor)

        # Flatten the output and apply the output layer
        obs_tensor = obs_tensor.view(batch_size, -1)
        obs_tensor = self.output_layer(obs_tensor)

        return obs_tensor

class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_features, features_dim):
        super(GraphAttentionNetwork, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GATConv(16, features_dim, heads=4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GNNetwork(nn.Module):
    def __init__(self, num_features, features_dim):
        super(GNNetwork, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, features_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TemporalCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(TemporalCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class GATFeatureExtractor(nn.Module):
    def __init__(self, observation_space, hidden_size=64):
        super(GATFeatureExtractor, self).__init__()
        self.observation_space = observation_space
        self.hidden_size = hidden_size

        # We assume that the observation space is a Box space with shape (num_nodes, num_features)
        num_nodes, num_features = observation_space.shape

        # Define the GAT layers
        self.gat1 = GATConv(num_features, hidden_size, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_size * 8, hidden_size, dropout=0.6)

        # Compute the output size of the GAT layers
        x = torch.zeros((1, num_nodes, num_features))
        x = self.gat1(x)
        x = self.gat2(x)
        self.output_size = x.size(-1)

    def forward(self, x):
        # Convert the observation x to a PyTorch Geometric Data object
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        x = x.transpose(1, 2)

        # Apply the GAT layers
        x = F.elu(self.gat1(x))
        x = F.elu(self.gat2(x))

        # Reshape the output tensor to (batch_size, output_size)
        x = x.view(x.size(0), -1)

        return x

class PPOFeatureExtractor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PPOFeatureExtractor, self).__init__()
        self.graph_network = GraphAttentionNetwork(observation_space["graph"].num_features, 64)
        self.temporal_network = TemporalCNN(observation_space["temporal"].shape[0], 64)

    def forward(self, observation):
        x1 = self.graph_network(observation["graph"].x, observation["graph"].edge_index)
        x2 = self.temporal_network(observation["temporal"].unsqueeze(0))
        x = torch.cat((x1, x2), dim=1)
        return x

class CustomGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGNN, self).__init__(observation_space, features_dim)
        self.graph_network = GNNetwork(observation_space["gnss"].shape[-1], features_dim)
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
        if len(observations['visible_satnum'])==1:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            x = observations['gnss'][:, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index = observations['edge_index'][:, :, mask[0, :]]
            batch = torch.tensor(np.zeros([x.shape[1],], dtype=int)).to(device)
        else:
            mask = torch.any(observations['gnss'] != 0, dim=2)
            max_row_idx = torch.argmax(torch.sum(mask, dim=1))
            x = observations['gnss'][:, mask[max_row_idx, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            max_row_idx = torch.argmax(torch.sum(mask, dim=1))
            edge_index = observations['edge_index'][:, :, mask[max_row_idx, :]]
            batch = torch.tensor(np.zeros([x.shape[1],], dtype=int)).to(device)

        x1 = self.graph_network(x,edge_index)
        # x2 = self.gat(x,edge_index)
        x2 = global_mean_pool (x1, batch)
        return x2

class CustomGNN1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGNN1, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, features_dim)
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
            x1 = observations['gnss'][0, mask[0, :], :]
            mask = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask[0, :]].type(torch.LongTensor).to(device)
            batch_idx = torch.tensor(np.zeros([x1.shape[0],], dtype=int)).to(device)
        else:
            mask_x = torch.any(observations['gnss'] != 0, dim=2)
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
        x2 = F.dropout(x2, training=self.training)
        x2 = self.conv2(x2, edge_index1)
        x3 = global_mean_pool (x2, batch_idx)
        return x3

class CustomGNN2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGNN2, self).__init__(observation_space, features_dim)
        num_features=observation_space["gnss"].shape[-1]
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, features_dim)
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
        x2 = F.elu(x2)
        x2 = self.conv2(x2, edge_index1)
        x3 = global_mean_pool (x2, batch_idx)
        return x3

class CustomGNNcat(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomGNNcat, self).__init__(observation_space, features_dim)
        self.num_features=observation_space["gnss"].shape[-1]
        self.conv1 = GCNConv(self.num_features, 8)
        self.hiddensize=16
        self.conv2 = GCNConv(8, self.hiddensize)
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
            mask_e = torch.any(observations['edge_index'] != 0, dim=1)
            edge_index1 = observations['edge_index'][0, :, mask_e[0, :]].type(torch.LongTensor).to(device)
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
        x3 = F.elu(x2)
        xx = self.conv2(x3, edge_index1)
        # x5 = global_mean_pool (xx, batch_idx)
        batch_size = len(torch.unique(batch_idx))
        max_nodes_num = np.int(self.features_dim/self.hiddensize)
        new_shape = (batch_size, max_nodes_num*xx.shape[-1])
        new_x = torch.zeros(new_shape, dtype=xx.dtype, device=xx.device)
        # xx1 = [ [,torch.zeros(features_dim, dtype=xx.dtype, device=xx.device)] for i in range(batch_size)]
        for i in range(batch_size):
            xtmp=xx[batch_idx==i,:].reshape(1,-1)
            new_x[i,:xtmp.shape[-1]]=xtmp
        return new_x