import time
import torch
from thop import profile

import constants


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


def get_gpu_model(flops_per_second_dict):
    device_name = torch.cuda.get_device_name(0)
    for model in flops_per_second_dict.keys():
        if model in device_name:
            return model
    raise ValueError(
        f"GPU model '{device_name}' not found in "
        "the provided FLOPS per second dictionary:"
        f" {flops_per_second_dict.keys()}."
        )


def compute_mfu( # Model FLOP Utilization (MFU)
        model: torch.nn.Module,
        vocab_size: int,
        context_length: int,
        device: torch.device
    ):

    gpu_model = get_gpu_model(constants.FLOPS_PER_SECOND)

    min_batch_size = 1
    max_batch_size = None
    max_possible_batch_size = 4096

    while min_batch_size <= max_possible_batch_size:
        batch_size = (min_batch_size + max_possible_batch_size) // 2
        try:
            input_tensor = torch.randint(
                0, vocab_size,
                (batch_size, context_length),
                device=device
            )

            model = model.bfloat16().to(device)
            model.train()

            # Start timing
            torch.cuda.synchronize()
            start_time = time.time()

            # Forward & backward pass
            output = model(input_tensor)
            loss = output.sum()  # Compute a dummy loss
            loss.backward()

            # End timing
            torch.cuda.synchronize()
            end_time = time.time()

            total_time_seconds = end_time - start_time

            # Calculate FLOPs for forward pass
            macs, params = profile(model, inputs=(input_tensor,), verbose=False)
            flops_forward = 2 * macs  # Assuming one MAC equals two FLOPs

            # Estimate FLOPs for backward pass (typically 2x forward FLOPs)
            flops_backward = 2 * flops_forward

            # Total FLOPs for forward + backward passes
            total_flops = flops_forward + flops_backward  # Or total_flops = flops_forward * 3

            data_type = next(model.parameters()).dtype
            max_flops_per_second = constants.FLOPS_PER_SECOND[gpu_model].get(data_type, 0)

            # Compute tokens per second
            tokens_processed = batch_size * context_length
            tokens_per_second = tokens_processed / total_time_seconds

            # Compute FLOPs per token
            flops_per_token = total_flops / tokens_processed

            # Compute theoretical max tokens per second
            if flops_per_token > 0:
                theoretical_max_tokens_per_second = max_flops_per_second / flops_per_token
            else:
                theoretical_max_tokens_per_second = 0  # Avoid division by zero

            # Compute MFU
            if theoretical_max_tokens_per_second > 0:
                mfu = tokens_per_second / theoretical_max_tokens_per_second
            else:
                mfu = 0  # Avoid division by zero

            print(f"  Batch size {batch_size}: Tokens/sec: {tokens_per_second:.2f}, MFU: {mfu:.4f}")

            # If successful, try a larger batch size
            min_batch_size = batch_size + 1
            max_batch_size = batch_size

            # Clean up
            del model, input_tensor, output, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Try smaller batch size
                max_possible_batch_size = batch_size - 1

                # Clean up
                try:
                    del model, input_tensor
                    torch.cuda.empty_cache()
                except NameError:
                    pass
            else:
                raise e