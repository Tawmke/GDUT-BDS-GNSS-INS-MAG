from typing import Iterable, List, Optional, Union, Tuple, Dict
import os.path as osp
import torch
import torch.nn as nn
import model.common.utils as ptu
from torch.nn.modules import module

def to_list(
    x,
    length=None,
    pad_last_ietm=True,
    pad_item=None
):
    # None to []
    x = [] if x is None else x
    # item to [item] if type(item) is not a list
    x = x if type(x) is list else [x]
    # convert to a list with a given length.
    # if pad_last_ietm is True, pad the list with the the last item.
    # else pad the list with pad_item
    if length == 0:
        return []
    if length is not None:
        pad_item = x[-1] if pad_last_ietm else pad_item
        while len(x) < length:
            x.append(pad_item)
        assert len(x) == length
    return x


class MyLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            with_bias: bool = True,
            init_bias_constant: Union[int, float] = 0,
            connection: str = "simple",  # densenet, resnet, simple
            init_func_name: str = 'orthogonal_',
            init_kwargs: dict = {},
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()

        if connection == "densenet":
            self.final_out_features = out_features + in_features
        else:
            self.final_out_features = out_features
            if connection == "resnet":
                assert out_features == in_features
            elif connection == "simple":
                pass
            else:
                raise NotImplementedError
        self.connection = connection

        self.init_func_name = init_func_name
        self.init_kwargs = init_kwargs

        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = with_bias
        self.init_bias_constant = init_bias_constant

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self._get_parameters()
        self.reset_parameters()

    def _get_parameters(self) -> None:
        self.weight, self.bias = self._creat_weight_and_bias()

    def _creat_weight_and_bias(self) -> Tuple[nn.Parameter, Optional[nn.Parameter]]:
        weight = nn.Parameter(
            torch.empty(
                (self.in_features, self.out_features),
                **self.factory_kwargs
            )
        )
        if self.with_bias:
            bias = nn.Parameter(
                torch.empty((1, self.out_features), **self.factory_kwargs)
            )
        else:
            bias = None
        return weight, bias

    def reset_parameters(self) -> None:
        self._reset_weight_and_bias(self.weight, self.bias)

    def _reset_weight_and_bias(
            self,
            weight: nn.Parameter,
            bias: Optional[nn.Parameter],
            init_func_name: Optional[str] = None,
            init_kwargs: dict = {}
    ) -> None:
        if init_func_name is None:
            init_func_name = self.init_func_name
            init_kwargs = self.init_kwargs
        init_func = eval("nn.init." + init_func_name)
        init_func(weight.T, **init_kwargs)

        if bias is not None:
            if self.init_bias_constant is None:
                fan_in = self.in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
            else:
                nn.init.constant_(bias, self.init_bias_constant)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, connection={}'.format(
            self.in_features, self.out_features, self.with_bias, self.connection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 2:
            x = x.unsqueeze(0)

        if self.with_bias:
            output = x.matmul(self.weight) + self.bias
        else:
            output = x.matmul(self.weight)

        if self.connection == "densenet":
            output = torch.cat([output, x], -1)
        elif self.connection == "resnet":
            output = x + output

        return output

    def get_weight_decay(self, weight_decay: Union[int, float] = 0) -> torch.Tensor:
        return (self.weight ** 2).sum() * weight_decay * 0.5


class MyEnsembleLinear(MyLinear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            ensemble_size: int,
            **linear_kwargs
    ) -> None:
        self.ensemble_size = ensemble_size
        super().__init__(in_features, out_features, **linear_kwargs)

    def _get_parameters(self) -> None:
        self.weights, self.biases = [], []
        for i in range(self.ensemble_size):
            weight, bias = self._creat_weight_and_bias()
            weight_name, bias_name = 'weight_net%d' % i, 'bias_net%d' % i
            self.weights.append(weight)
            self.biases.append(bias)
            setattr(self, weight_name, weight)
            setattr(self, bias_name, bias)

    def reset_parameters(self) -> None:
        for w, s in zip(self.weights, self.biases):
            self._reset_weight_and_bias(w, s)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, ensemble_size={}, bias={}, connection={}'.format(
            self.in_features, self.out_features, self.ensemble_size, self.with_bias, self.connection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        while x.dim() < 3:
            x = x.unsqueeze(0)
        w = torch.stack(self.weights, 0)
        if self.with_bias:
            b = torch.stack(self.biases, 0)
            output = x.matmul(w) + b
        else:
            output = x.matmul(w)
        if self.connection == "densenet":
            if x.dim() == 3 and x.shape[0] == 1:
                x = x.repeat(self.ensemble_size, 1, 1)
            output = torch.cat([output, x], -1)
        elif self.connection == "resnet":
            output = output + x

        return output

    def get_weight_decay(self, weight_decay: Union[int, float] = 0) -> torch.Tensor:
        decays = []
        for w in self.weights:
            decays.append((w ** 2).sum() * weight_decay * 0.5)
        return sum(decays)


def get_fc(
    in_features: int,
    out_features: int,
    ensemble_size: Optional[int],
    is_last: bool = False,
    **linear_kwargs
) -> Union[MyLinear, MyEnsembleLinear]:
    if is_last: # last layer should not be dense or res
        linear_kwargs['connection'] = "simple"

    if ensemble_size is None:
        fc = MyLinear(
            in_features,
            out_features,
            **linear_kwargs
        )
    else:
        fc = MyEnsembleLinear(
            in_features,
            out_features,
            ensemble_size,
            **linear_kwargs
        )
    return fc

def build_mlp(
        layer_size: List[int],
        ensemble_size: Optional[int],
        activation: Union[str, List[str]] = "relu",
        output_activation: str = 'identity',
        **linear_kwargs: dict,
) -> Tuple[nn.Module, List[int]]:
    num_fc = len(layer_size) - 1
    act_name = to_list(activation, num_fc - 1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]
    module_list = []
    final_layer_size = [layer_size[0]]  # for densenet
    in_features = layer_size[0]
    assert len(act_func) == num_fc
    for i in range(num_fc):
        if i == num_fc - 1:
            if "connection" in linear_kwargs:
                linear_kwargs["connection"] = "simple"
        fc = get_fc(in_features, layer_size[i + 1], ensemble_size, i == num_fc, **linear_kwargs)
        module_list.append(fc)
        module_list.append(act_func[i])
        in_features = fc.final_out_features
        final_layer_size.append(in_features)  # for densenet
    return nn.Sequential(*module_list), final_layer_size


# A more simple implementation that directly uses nn.Linear.
# However, it does not support EnsembleLinear.
def build_mlp_v2(
        layer_size: List[int],
        ensemble_size: Optional[int],
        activation: Union[str, List[str]] = "relu",
        output_activation: str = 'identity',
        **linear_kwargs: dict,
) -> Tuple[nn.Module, List[int]]:
    assert ensemble_size is None
    num_fc = len(layer_size) - 1
    act_name = to_list(activation, num_fc - 1)
    act_name.append(output_activation)
    act_func = [ptu.get_activation(act) for act in act_name]

    module_list = []
    assert len(act_func) == num_fc
    for i in range(num_fc):
        fc = nn.Linear(layer_size[i], layer_size[i + 1], **linear_kwargs)
        module_list.append(fc)
        module_list.append(act_func[i])
    return nn.Sequential(*module_list), layer_size

class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_layers: List[int],
            ensemble_size: Optional[int] = None,
            activation: Union[str, List[str]] = 'relu',
            output_activation: str = 'identity',
            module_name: str = 'mlp',
            v1_or_v2: str = 'v1',
            **linear_kwargs
    ) -> None:
        # If ensemble is n
        # Given a tensor with shape (n,a,b) output (n,a,c)
        # Given a tensor with shape (a,b) output (n,a,c).
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.ensemble_size = ensemble_size
        self.module_name = module_name

        # get activation functions
        self.num_fc = len(hidden_layers) + 1
        layer_size = [input_size] + hidden_layers + [output_size]
        self.v1_or_v2 = v1_or_v2
        if v1_or_v2 == "v2":
            assert ensemble_size is None
        build_func = build_mlp if self.v1_or_v2 == "v1" else build_mlp_v2
        self.net, self.layer_size = build_func(
            layer_size,
            ensemble_size,
            activation,
            output_activation,
            **linear_kwargs
        )
        self.min_output_dim = 2 if self.ensemble_size is None else 3

    def _forward_v1(self, x: torch.Tensor) -> torch.Tensor:
        if self.ensemble_size is None:
            max_output_dim = x.dim()
        else:
            max_output_dim = x.dim() + 1
        output = self.net(x)
        while output.dim() > max_output_dim:
            output = output.squeeze(0)
        return output

    def _forward_v2(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v1_or_v2 == "v1":
            return self._forward_v1(x)
        else:
            return self._forward_v2(x)

    def get_snapshot(self, key_must_have: str = '') -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        state_dict = self.state_dict()
        if key_must_have == '':
            new_state_dict = state_dict
        else:
            for k, v in state_dict.items():
                if key_must_have in k:
                    new_state_dict[k] = v
        return new_state_dict

    def load_snapshot(self, loaded_state_dict: dict, key_must_have: str = '') -> None:
        state_dict = self.state_dict()
        if key_must_have == '':
            state_dict = loaded_state_dict
        else:
            for k, v in loaded_state_dict.items():
                if key_must_have in k:
                    state_dict[k] = v
        self.load_state_dict(state_dict)

    def save(self, save_dir: str, net_id: Optional[int] = None) -> None:
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(save_dir, '%s.pt' % self.module_name)
        else:
            assert self.v1_or_v2 == "v1"
            net_name = 'net%d' % net_id
            file_path = osp.join(save_dir, '%s_%s.pt' % (self.module_name, net_name))
        state_dict = self.get_snapshot(net_name)
        torch.save(state_dict, file_path)

    def load(self, load_dir: str, net_id=None) -> None:
        if self.ensemble_size is None or net_id is None:
            net_name = ''
            file_path = osp.join(load_dir, '%s.pt' % self.module_name)
        else:
            assert self.v1_or_v2 == "v1"
            net_name = 'net%d' % net_id
            file_path = osp.join(load_dir, '%s_%s.pt' % (self.module_name, net_name))
            if not osp.exists(file_path):
                file_path = osp.join(load_dir, '%s.pt' % self.module_name)
        if osp.exists(file_path):
            success = True
            loaded_state_dict = torch.load(file_path)
            self.load_snapshot(loaded_state_dict, net_name)
        else:
            success = False
        return success

    def get_weight_decay(self, weight_decays: Union[int, float, List[Union[int, float]]] = 0) -> torch.Tensor:
        assert self.v1_or_v2 == "v1"
        weight_decays = to_list(weight_decays, len(self.layer_size) - 1)
        fcs = [fc for fc in self.net if hasattr(fc, "get_weight_decay")]
        assert len(fcs) == len(weight_decays)
        weight_decay_tensors = []
        for weight_decay, fc in zip(weight_decays, fcs):
            weight_decay_tensors.append(fc.get_weight_decay(weight_decay))
        return sum(weight_decay_tensors)

class Prediction(nn.Module):
    def __init__(self, dim_input, dim_discretize, dim_state, normalizer="batch", trainable=True):
        super(Prediction,self).__init__()
        self.output_dim = dim_discretize * dim_state
        self.normalizer = normalizer
        self.pred_layer = nn.Linear(dim_input, 1024) # input:policy.lstm_output_dim
        self.out_layer_re = nn.Linear(1024, self.output_dim)
        self.out_layer_im = nn.Linear(1024, self.output_dim)
        self.BatchNorm1d = nn.BatchNorm1d(1024)
        self.LayerNorm = nn.LayerNorm(1024)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = self.pred_layer(inputs)
        if self.normalizer == 'batch':
            x = self.BatchNorm1d(x)
        elif self.normalizer == 'layer':
            x = self.LayerNorm(x)
        x = self.relu(x)
        return self.out_layer_re(x), self.out_layer_im(x)

class Projection(nn.Module):
    def __init__(self,input_dim,output_dim=256, normalizer="batch"):
        super(Projection, self).__init__()
        self.output_dim = output_dim
        self.normalizer = normalizer
        self.dense1 = nn.Linear(input_dim, output_dim*2)
        self.dense2 = nn.Linear(output_dim*2, output_dim)
        self.BatchNorm1d = nn.BatchNorm1d(output_dim*2)
        self.LayerNorm = nn.LayerNorm(output_dim*2,eps=1e-5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    def forward(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if self.normalizer == 'batch':
            x = self.BatchNorm1d(x)  # training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
        elif self.normalizer == 'layer':
            x = self.LayerNorm(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

class Projection2(nn.Module):
    def __init__(self,input_dim,output_dim=256, normalizer="batch"):
        super(Projection2, self).__init__()
        self.output_dim = output_dim
        self.normalizer = normalizer
        self.dense1 = nn.Linear(input_dim, output_dim*2)
        self.dense2 = nn.Linear(output_dim*2, output_dim)
        self.BatchNorm1d = nn.BatchNorm1d(output_dim*2)
        self.LayerNorm = nn.LayerNorm(output_dim*2,eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.dense1(inputs)
        if self.normalizer == 'batch':
            x = self.BatchNorm1d(x)  # training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
        elif self.normalizer == 'layer':
            x = self.LayerNorm(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x