import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_weight(model: nn.Module, weight_name: str):
    try:
        return model.get_parameter(weight_name)
    except AttributeError:
        return model.get_buffer(weight_name)


def replace_packed_module_name(weight_name: str, source: str, target: str) -> str | None:
    prefix = f"{source}."
    if weight_name.startswith(prefix):
        return weight_name.replace(prefix, f"{target}.", 1)
    source = f".{source}."
    if source in weight_name:
        return weight_name.replace(source, f".{target}.", 1)
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    ignored_weights = getattr(model, "ignored_weights", ())
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if any(weight_name.startswith(prefix) for prefix in ignored_weights):
                    continue
                for k in packed_modules_mapping:
                    v, shard_id = packed_modules_mapping[k]
                    param_name = replace_packed_module_name(weight_name, k, v)
                    if param_name is not None:
                        param = get_weight(model, param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = get_weight(model, weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
