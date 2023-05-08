import torch
from torch.utils import data
import torch.nn as nn
from models.modules.quantize import QConv2d,QLinear


def search_replace_measure(model,measure,name_model=''):
    for i,m in enumerate(model.children()):
        modules_names=[key for key in model._modules.keys()]
        layer_name=name_model+'.'+modules_names[i] if name_model !='' else name_model+modules_names[i]
        m.name=layer_name
        if is_q_module(m):
            print("Layer {}, switch from measure = {} to measure = {}.".format(
                layer_name, m.measure, measure))
            m.quantize_input.measure=measure
            m.quantize_weight.measure=measure
        search_replace_measure(m,measure,layer_name)
    return model


def is_q_module(m):
    return isinstance(m, QConv2d) or isinstance(m, QLinear)
