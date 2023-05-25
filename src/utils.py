import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from typing import List

from hypernet import HyperNet, AuxilaryLlamaMLP, LowRankLinear
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--num-selected-neurons", type=int, default=8)

    args = parser.parse_args()
    return args

def convert_modules(model, module_type, new_module_type):
    """
    Convert all modules of type `module_type` to `new_module_type`
    """
    for name, module in model.named_children():
        if isinstance(module, module_type):
            setattr(model, name, new_module_type(module))
        else:
            convert_modules(module, module_type, new_module_type)

def add_auxilary_head_to_model(model):
    if isinstance(model, LlamaForCausalLM):
        # convert all llamaMLP to AuxilaryLlamaMLP
        convert_modules(model, LlamaMLP, AuxilaryLlamaMLP)

def get_layers(model):
    if isinstance(model, LlamaForCausalLM):
        return model.model.layers
    if isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers

def get_serialized_inputs_for_hypernet(layer, key, value, args):
    """
    Get serialized inputs of the hypernet for the given layer
    """
    if isinstance(layer, OPTDecoderLayer):
        fc1_weight = layer.fc1.weight.data
        fc2_weight = layer.fc2.weight.data.T
        score = fc1_weight @ key
        similar_indices = torch.argsort(score, descending=True)[:args.num_selected_neurons]
        fc1_weight = fc1_weight[similar_indices]
        fc2_weight = fc2_weight[similar_indices]

        return key, value, torch.cat([fc1_weight, fc2_weight], dim=0), similar_indices
    

def deserialize_outputs_for_hypernet(layer, output):
    """
    Deserialize the outputs of the hypernet for the given layer
    """
    if isinstance(layer, OPTDecoderLayer):
        fc1_weight, fc2_weight = torch.split(output, [layer.fc1.weight.shape[0], layer.fc2.weight.shape[1]])
        return fc1_weight, fc2_weight.T

def apply_outputs_for_hypernet(layer, output, similar_indices):
    """
    Apply the output weights for the given layer
    """
    if isinstance(layer, OPTDecoderLayer):
        fc1_weight, fc2_weight = deserialize_outputs_for_hypernet(layer, output)
        layer.fc1.weight.data[similar_indices] += fc1_weight
        layer.fc2.weight.data.T[similar_indices] += fc2_weight

def update_model_through_hypernet(model, hypernet, args, inputs):
    """
    Update the model through the hypernet
    """
    # TODO: update the model through the hypernet
    # with torch.no_grad():
    #     outputs = model(**inputs, return_dict=True)
    #     key = outputs.last_hidden_state[:, -1]
    #     value = outputs.last_hidden_state[:, -1]
    for layer in get_layers(model):
        if isinstance(layer, OPTDecoderLayer):
            serialized_inputs = get_serialized_inputs_for_hypernet(layer, key, value, args)
            output = hypernet(serialized_inputs)
            apply_outputs_for_hypernet(layer, output, serialized_inputs[-1])