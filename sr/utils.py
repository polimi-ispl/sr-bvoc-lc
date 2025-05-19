import torch
import numpy as np
import torch.nn.functional as F
import random
from datetime import datetime
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio, error_relative_global_dimensionless_synthesis, universal_image_quality_index, spectral_angle_mapper
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error
from sr.sre import signal_to_reconstruction_error
from sr.scc import spatial_correlation_coefficient

BASE_ROOT_PATH = '/nas/home/...' # TODO: Set the base root path
BASE_DATASET_PATH = '/nas/home/...' # TODO: Set the base root path
BASE_OUTPUT_DIR_PATH = '/nas/home/...' # TODO: Set the base root path

MIN, TINY, MAX, EPS = torch.finfo(torch.float32).min, torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).max, torch.finfo(torch.float32).eps

def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)


def emission_consistency_loss(sr_output, lr_original, overlapping_factor, patch_size_lr, upscaling_factor):
    '''
    Compute the consistency loss between two tensors, ensuring that the mean of values within corresponding patches in
    the original low-resolution tensor "lr_original" is approx. equal to the mean of values within
    corresponding patches in the super-resolved tensor "sr_output".
    The consistency loss is calculated by comparing patches of the same spatial area in both tensors. The size of these
    patches is determined by the parameter "patch_size", which applies to both dimensions of the low-resolution tensor.
    This patch size is then upscaled by the specified upscaling factor to determine the size of patches in the
    super-resolved tensor.
    The number of patches used for comparison is fixed and computed based on the dimensions of the low-resolution
    tensor "lr_original". The overlapping factor, defined by the parameter "overlapping_factor", determines the extent
    of overlap between adjacent patches in both dimensions of the low-resolution tensor, thus the stride.
    For example, if upscaling from a 3x3 to a 6x6 tensor (thus using a 2x upscaling factor) with a patch size of 2x2
    (on the low-resolution tensor) and an overlapping factor of 1, resulting in a 4x4 patch size on the super-resolved
    tensor, there will be 4 patches in both the lr_original and sr_output tensors. The function ensures that the mean of
    values within each 4x4 patch of the super-resolved tensor is approx. equal to the mean of values
    within the corresponding 2x2 patch of the low-resolution tensor.
    :param sr_output: The super-resolved tensor.
    :param lr_original: The original low-resolution tensor.
    :param overlapping_factor: The factor determining the extent of overlap between adjacent patches.
    :param patch_size_lr: The size of patches (observation window) in the low-resolution tensor.
    :param upscaling_factor: The factor by which the patch size is upscaled to match the super-resolved tensor.
    :return: The consistency loss.
    '''
    # Compute the patch size on the super-resolved tensor
    patch_size_sr = patch_size_lr * upscaling_factor
    # Compute the stride of the overlapping patches
    stride = patch_size_lr - overlapping_factor
    # Compute the number of patches in the x and y dimensions
    num_patches_x = (lr_original.shape[-1] - patch_size_lr + 1) // stride
    num_patches_y = (lr_original.shape[-2] - patch_size_lr + 1) // stride
    # Loss initialization
    consistency_loss = 0

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Extract the subpatches from the original tensor
            subpatch_original = lr_original[:, :, i * stride:i * stride + patch_size_lr,
                             j * stride:j * stride + patch_size_lr]
            # Calculate corresponding indices for the super-resolved tensor
            sr_i_start = i * stride * upscaling_factor
            sr_j_start = j * stride * upscaling_factor
            # Extract the patch from the super-resolved tensor using the calculated indices
            subpatch_output = sr_output[:, :, sr_i_start:sr_i_start + patch_size_sr, sr_j_start:sr_j_start + patch_size_sr]
            # Compute the consistency loss for the current patch.
            # The loss is the absolute difference between the mean values of the patches
            consistency_loss += torch.abs(torch.mean(subpatch_original) - torch.mean(subpatch_output))

    # Normalize the consistency loss by the number of patches
    consistency_loss /= num_patches_x * num_patches_y
    return consistency_loss

def tensor2img(tensor, max_val=1):
    tensor = torch.clamp(tensor, 0, max_val)  # Ensure values are between 0 and max_val
    tensor = (tensor * 255.0).round()
    return tensor

def compute_metrics(output, original, metrics, ten2img=False, mean=False):
    results = {}
    output[output < 0.0] = 0.0  # Clip negative SR values to 0

    if ten2img:
        # Need to convert to uint8 for SSIM
        original = tensor2img(original)
        output = tensor2img(output)

    if type(original) != type(output):
        print(f'Type original_output_exp: {type(output)}')
        print(f'Type original_hr: {type(original)}')

    # Compute metrics for each single image in the batch
    if 'SSIM' in metrics:
        a = [structural_similarity_index_measure(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['SSIM'] = a
    if 'PSNR' in metrics:
        a = [peak_signal_noise_ratio(output[i], original[i]).item() for i in range(output.size(0))]
        results['PSNR'] = a
    if 'MSE' in metrics:
        a = [mean_squared_error(output[i], original[i]).item() for i in range(output.size(0))]
        results['MSE'] = a
    if 'NMSE' in metrics:
        # Log NMSE, dB
        a = [(10 * torch.log10(mean_squared_error(output[i], original[i]) / torch.mean(original[i] ** 2))).item() for i in range(output.size(0))]
        results['NMSE'] = a
    if 'MAE' in metrics:
        a = [mean_absolute_error(output[i], original[i]).item() for i in range(output.size(0))]
        results['MAE'] = a
    if 'MaxAE' in metrics:
        a = [torch.max(torch.abs(output[i] - original[i])).item() for i in range(output.size(0))]
        results['MaxAE'] = a
    if 'ERGAS' in metrics:
        a = [error_relative_global_dimensionless_synthesis(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['ERGAS'] = a
    if 'UIQ' in metrics:
        a = [universal_image_quality_index(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['UIQ'] = a
    if 'SCC' in metrics:
        a = [spatial_correlation_coefficient(output[i], original[i]).item() for i in range(output.size(0))]
        results['SCC'] = a
    # if 'SAM' in metrics:
    #     a = [spectral_angle_mapper(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
    #     results['SAM'] = a
    if 'SRE' in metrics:
        a = [signal_to_reconstruction_error(torch.unsqueeze(output[i], dim=0), torch.unsqueeze(original[i], dim=0)).item() for i in range(output.size(0))]
        results['SRE'] = a

    if mean:
        for key in results.keys():
            results[key] = np.mean(results[key])

    return results

def custom_criterion(output, target, additional_losses=False, upscaling_factor=2):
    mse_loss = F.mse_loss(output, target)  # L2 loss
    if additional_losses:
        loss_rescale_factors = [2, 0.000001]
        em_penalty = torch.mean(torch.relu(output - 1)) + torch.mean(torch.relu(-output))  # Penalize values above 1 and below 0
        em_consistency = emission_consistency_loss(output, target, overlapping_factor=1, patch_size_lr=2, upscaling_factor=upscaling_factor)  # Consistency loss for emission SR consistency
        return mse_loss, loss_rescale_factors[0] * em_penalty, loss_rescale_factors[1] * em_consistency
    else:
        return mse_loss, torch.tensor(0.0), torch.tensor(0.0)

def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"NaN or Inf found in {name} tensor.")


def gradient_norm(model):
    # Filter out parameters with None gradients
    parameters = [p for p in model.parameters() if p.grad is not None]
    # Compute the norm of gradients for each parameter and sum them
    gradient_norm = sum(p.grad.data.norm(2).item() for p in parameters)
    return gradient_norm


def check_gradients(model):
    problematic_params = []
    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad) | torch.isinf(param.grad)).any():
            problematic_params.append(name)
    if problematic_params:
        problematic_params_str = ', '.join(problematic_params)
        return True, problematic_params_str  # Gradient contains NaN or Inf values
    else:
        return False, ''  # Gradients are okay


def shorten_datetime(datetime_obj):
    # Extract date and time components
    date_part = datetime_obj.strftime("%Y%m%d")
    time_part = datetime_obj.strftime("%H%M%S")
    # Combine date and time components with hyphen
    formatted_datetime = f"{date_part}-{time_part}"
    return formatted_datetime
