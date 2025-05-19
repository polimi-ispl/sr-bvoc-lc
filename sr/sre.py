import torch


def signal_to_reconstruction_error(output, original):
    # Signal to Reconstruction Ratio Error (SRE)
    n = original.numel()
    E_X2 = torch.mean(original ** 2)
    error = torch.norm(output - original) ** 2 / n
    SRE = 10 * torch.log10(E_X2 / error)
    return SRE