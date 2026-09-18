import torch
from torch import nn

from manas.training.utils import unwrap


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    device = next(model.parameters()).device
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(device)
            module.lora = lora
            original_forward = module.forward

            def forward_with_lora(x, base=original_forward, adapter=lora):
                return base(x) + adapter(x)

            module.forward = forward_with_lora
    return model


def lora_parameters(model):
    return [p for name, p in model.named_parameters() if ".lora." in name]


def lora_state_dict(model):
    state = {}
    for name, module in unwrap(model).named_modules():
        if hasattr(module, "lora"):
            for key, value in module.lora.state_dict().items():
                state[f"{name}.lora.{key}"] = value.detach().cpu().half()
    return state


def save_lora(model, path):
    torch.save(lora_state_dict(model), path)


def load_lora(model, path):
    device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    for name, module in unwrap(model).named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            module.lora.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    return model


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw = unwrap(model)
    state = {k: v.detach().cpu().half() for k, v in raw.state_dict().items() if ".lora." not in k}
    for name, module in raw.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            delta = module.lora.B.weight.data @ module.lora.A.weight.data
            state[f"{name}.weight"] = (module.weight.data + delta).detach().cpu().half()
    torch.save(state, save_path)
    return save_path
