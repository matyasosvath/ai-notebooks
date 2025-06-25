import torch
from thop import profile


def compute_flops(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: torch.device
    ):
    model = model.bfloat16()
    model.to(device)
    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs
    print(f"Model has {flops:.1e} FLOPS and {params / 1e6:.2f} M parameters.")
    return flops, params


def compute_model_params(model: torch.nn.Module):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params / 1e6:.2f} M")
    return params