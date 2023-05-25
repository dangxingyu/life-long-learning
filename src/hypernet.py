import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaMLP
import copy

class LowRankLinear(nn.Module):
    def __init__(self, rank, in_features, out_features, bias=False):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.ln1 = nn.Linear(in_features, rank, bias=False)
        self.ln2 = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x):
        return self.ln2(self.ln1(x))

class AuxilaryLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rank: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.aux_gate_proj = LowRankLinear(rank, hidden_size, intermediate_size, bias=False)
        
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.aux_down_proj = LowRankLinear(rank, intermediate_size, hidden_size, bias=False)
        
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.aux_up_proj = LowRankLinear(rank, hidden_size, intermediate_size, bias=False)
        
        self.act_fn = ACT2FN[hidden_act]
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        def down_proj(x):
            return self.down_proj(x) + self.aux_down_proj(x)
        def up_proj(x):
            return self.up_proj(x) + self.aux_up_proj(x)
        def gate_proj(x):
            return self.gate_proj(x) + self.aux_gate_proj(x)
        return down_proj(self.act_fn(gate_proj(x)) * up_proj(x))

    def load_from_llama_mlp(self, mlp: LlamaMLP):
        self.gate_proj.weight.data = mlp.gate_proj.weight.data.clone().detach()
        self.down_proj.weight.data = mlp.down_proj.weight.data.clone().detach()
        self.up_proj.weight.data = mlp.up_proj.weight.data.clone().detach()

    def load_from_hypernet_output(self, x):
        # x: [rank, hidden_size*3+intermediate_size*3]
        cur_len = 0
        self.aux_gate_proj.ln1.weight.data = x[:, cur_len:cur_len+self.hidden_size]
        cur_len += self.hidden_size
        self.aux_gate_proj.ln2.weight.data = x[:, cur_len:cur_len+self.intermediate_size].T
        cur_len += self.intermediate_size
        self.aux_down_proj.ln1.weight.data = x[:, cur_len:cur_len+self.intermediate_size]
        cur_len += self.intermediate_size
        self.aux_down_proj.ln2.weight.data = x[:, cur_len:cur_len+self.hidden_size].T
        cur_len += self.hidden_size
        self.aux_up_proj.ln1.weight.data = x[:, cur_len:cur_len+self.hidden_size]
        cur_len += self.hidden_size
        self.aux_up_proj.ln2.weight.data = x[:, cur_len:cur_len+self.intermediate_size].T
        cur_len += self.intermediate_size
        assert cur_len == x.shape[1]

# TODO: HyperNetForLlama
class HyperNetForLlama(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_selected_weights: int,  # down sampled by the similarity with the input key vector
                 output_rank: int,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.down_sample_size = num_selected_weights
        self.output_rank = output_rank

        input_size = num_selected_weights * 3 * hidden_size + hidden_size + hidden_size
        output_size = output_rank * (hidden_size * 3 + intermediate_size * 3)
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)


    def forward(self, key, value, ffn_weights):
        if len(key.shape) == 1:
            key = key.unsqueeze(0)
        if len(value.shape) == 1:
            value = value.unsqueeze(0)
        x = torch.cat([key, value, ffn_weights], dim=0)

class HyperNetForOPT(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_selected_neurons: int,
                 hypernet_hidden_size: int,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_selected_neurons = num_selected_neurons
        self.hypernet_hidden_size = hypernet_hidden_size

        input_size = num_selected_neurons * 2 * hidden_size + hidden_size + hidden_size
        output_size = hidden_size * 2 * num_selected_neurons
        self.linear1 = nn.Linear(input_size, hypernet_hidden_size)
        self.linear2 = nn.Linear(hypernet_hidden_size, output_size)

    def forward(self, key, value, ffn_weights, *args, **kwargs):
        if len(key.shape) == 1:
            key = key.unsqueeze(0)
        if len(value.shape) == 1:
            value = value.unsqueeze(0)
        x = torch.cat([key, value, ffn_weights], dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x