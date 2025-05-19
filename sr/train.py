import sys
print('Python %s on %s' % (sys.version, sys.platform))
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import BVOCDatasetSR
from utils import set_backend, set_seed, weights_init, compute_metrics, \
    emission_consistency_loss, check_gradients, gradient_norm, shorten_datetime, custom_criterion, check_for_nan_inf
from original_san.san import SAN
from two_channel_san.san import SAN as TwoChannelSAN
from tree_channel_san.san import SAN as TreeChannelSAN
from original_hat.hat import HAT

set_backend()
set_seed(10)

BASE_OUTPUT_DIR_PATH = '<usr_dir>/runs/'
BASE_DATASET_PATH = '<usr_dir>/datasets/'

########################################
# Params ArgumentParser()              #
########################################
parser = argparse.ArgumentParser(description='BVOC Super-Resolution Training')

# Network
parser.add_argument('--model_name', type=str, default='SAN', help='Name of the Super-Resolution network')
parser.add_argument('--pretrained', action='store_true', help='Use a pretrained model')
parser.add_argument('--pretrained_model_path', type=str, default='', help='Path to the pretrained model weights')

# Data Specs
parser.add_argument('--hr_patch_size', type=int, default=30, help='Size of high-resolution patches')
parser.add_argument('--dataset_folder', type=str, default='p30pix_s1deg_z10_topdown_isoprene',
                    help='Specific folder for the dataset')
# parser.add_argument('--qt_type', type=str, default='HR_1e3qua_fullsub', help='Type of QT to use for the dataset')
parser.add_argument('--partition_file', type=str, default='partition_index_t70_v15_te15_gs10_n200000_seed10.csv', help='Partition file for the dataset')
parser.add_argument('--channels', type=int, default=1, help='number of channels for the SAN head')
parser.add_argument('--fusion_flag', type=str, default='', help='ag or forest')

# Training Specs
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--n_epochs', type=int, default=100000, help='Number of epochs to train')
parser.add_argument('--upscaling_factor', type=int, default=2, help='Upscaling factor for the super-resolution')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_patience', type=int, default=10,
                    help='Learning rate scheduler patience, after which the learning rate is reduced. Works only with on-validation scheduler type.')
parser.add_argument('--es_patience', type=int, default=50,
                    help='Early stopping patience, after which the training stops')
parser.add_argument('--prefetch_factor', type=int, default=20, help='Prefetch factor for the dataloader')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--train_batches_percentage', type=float, default=1.0,
                    help='Percentage of train batches, after which the validation starts. 1.0=all the train batches are considered before the validation starts')
parser.add_argument('--val_batches_percentage', type=float, default=1.0,
                    help='Percentage of validation batches, after which the next epochs starts, 1.0=all the validation batches are considered before the next epoch starts')
parser.add_argument('--warmup', type=int, default=3, help='Warmup epochs')
parser.add_argument('--es_min_delta', type=float, default=0.05e-5, help='Minimum delta for early stopping')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers of the dataloader')
parser.add_argument('--scheduler_type', type=str, default='on-validation',
                    help='Type of lr scheduler. Options: global (CosineAnnealingWarmRestarts), on-validation (ReduceLROnPlateau)')

# Training Regularization
parser.add_argument('--clip_gradient', action='store_true', help='Clip the gradient to prevent grad explosion')
parser.add_argument('--opt_weight_decay', type=float, default=0.0,
                    help='L2 regularization of the loss function. This penalizes large weights in the model, which indirectly discourages large output values')
parser.add_argument('--additional_losses', action='store_true', help='Use additional losses')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='Alpha value, for the Emission Penalty (penalize values above 1 and below 0)')
parser.add_argument('--beta', type=float, default=0.0,
                    help='Beta value, fot the Emission Consistency (for LR-SR emission consistency)')

args = parser.parse_args()
print(args)

########################################
# Paths and Folders                    #
########################################
start_time = datetime.now()
dataset_name = os.path.basename(args.dataset_folder)
partition_name = os.path.basename(args.partition_file).split('.')[0]

if args.channels == 1:
    partition_name += '_ch1'
elif args.channels == 2:
    partition_name += '_ch2'
elif args.channels == 3:
    partition_name += '_ch3'

output_folder_name = f'{shorten_datetime(start_time)}_{partition_name}_{args.fusion_flag}'
output_dir_path = os.path.join(BASE_OUTPUT_DIR_PATH, dataset_name, f'x{args.upscaling_factor}',
                               f'{args.model_name}', output_folder_name)  # divided by dataset
tmp_run_name = f'{shorten_datetime(start_time)}_{partition_name}_{args.fusion_flag}'
writer = SummaryWriter(log_dir=os.path.join(BASE_OUTPUT_DIR_PATH, 'logs', tmp_run_name))
os.makedirs(output_dir_path, exist_ok=True)
print(
    f'==> Output dir: {output_dir_path}\nRun name: {output_folder_name}\nTensorboard dir: {os.path.join(BASE_OUTPUT_DIR_PATH, "logs", tmp_run_name)} <==')

########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))

# Networks
network = []
if args.model_name == 'SAN':
    network_kwargs = {
        'scale': args.upscaling_factor,
        'channels': args.channels,
    }
    network = SAN(**network_kwargs)
elif args.model_name == 'HAT':  
    additional_network_kwargs = {'compress_ratio': 3, # 12
                                'squeeze_factor': 30, # 12
                                'depths': (6, 6, 6, 6), # (3, 3, 3)
                                'embed_dim': 96,
                                'num_heads': (6, 6, 6, 6), # (3, 3, 3)
                                'mlp_ratio': 4., # 2
                                }
    network_kwargs = {'img_size': args.hr_patch_size // args.upscaling_factor,
                        'in_chans': args.channels,
                        'upscale': args.upscaling_factor,
                        'window_size': args.hr_patch_size // args.upscaling_factor // 3, # 5
                        'patch_size': 1, 
                        }
    network_kwargs.update(additional_network_kwargs)
    network = HAT(**network_kwargs)
    print('Using HAT')
else:
    print('SR Network not implemented !!!')

if args.pretrained:
    network.load_state_dict(torch.load(args.pretrained_model_path))
network.cuda()

# Losses
criterion = torch.nn.MSELoss()

# Optimizer & LR scheduler
optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)

if args.scheduler_type == 'global':
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-07)
elif args.scheduler_type == 'on-validation':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                              patience=args.lr_patience, min_lr=1e-07)

# Datasets and DataLoaders
dataset_kwargs = {'root': os.path.join(BASE_DATASET_PATH, args.dataset_folder),
                  'partition_file': os.path.join(BASE_DATASET_PATH, args.dataset_folder, args.partition_file),
                  'quantile_transform': True,
                  'channels': args.channels,
                  'fusion_flag': args.fusion_flag,
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     'prefetch_factor': args.prefetch_factor,
                     }

train_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='train'), shuffle=True, drop_last=True,
                                **dataloader_kwargs)
val_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode='val'), shuffle=False, drop_last=True,
                            **dataloader_kwargs)

# TODO add logger
# train_logger = Logger(n_epochs=args.n_epochs, batches_epoch=len(train_dataloader), mode='train')
# val_logger = Logger(n_epochs=args.n_epochs, batches_epoch=len(val_dataloader), mode='val')

# Info
print(f'\nBatches: \tTR {len(train_dataloader)} | VL: {len(val_dataloader)}'
      f'\nN. Maps: \tTR{len(train_dataloader) * args.batch_size} | VL{len(val_dataloader) * args.batch_size}')

########################################
# Training                             #
########################################
metrics = ['SSIM', 'PSNR', 'MSE', 'NMSE', 'MAE']
best_val_loss = np.inf
best_epoch = 1
batches_before_valid = round(len(train_dataloader) * args.train_batches_percentage)
batches_before_break_valid = round(len(val_dataloader) * args.val_batches_percentage)
tr_len = min(len(train_dataloader), batches_before_valid)
vl_len = min(len(val_dataloader), batches_before_break_valid)
tb_img_idx = 5  # TensorBoard selected image's index of the current batch

print(f'\n{start_time}\tTrain starts !')
print(f'Considered batches TR: {batches_before_valid} | VL: {batches_before_break_valid}')
for epoch in range(1, args.n_epochs):
    # ———————————————— Epoch Starts ————————————————
    ###################################### T R A I N ######################################
    total_train_loss, total_val_loss = 0, 0
    total_train_mse, total_val_mse = 0, 0
    total_train_em_penalty, total_val_em_penalty = 0, 0
    total_train_em_consistency, total_val_em_consistency = 0, 0
    network.train()
    for i, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
        optimizer.zero_grad()
        # (b, c, h, w)
        hr_img = batch['HR'].cuda()
        lr_img = batch['LR'].cuda()
        output = network(lr_img)

        # If the output has multiple channels, only use the first channel
        if output.shape[1] > 1:
            output = output[:, 0:1, :, :]

        # Check for NaN and Inf
        # check_for_nan_inf(hr_img, 'HR')
        # check_for_nan_inf(lr_img, 'LR')
        # check_for_nan_inf(output, 'SR')

        # Complete loss
        # mse_loss, em_penalty, em_consistency = custom_criterion(output, hr_img, additional_losses=args.additional_losses, upscaling_factor=args.upscaling_factor)
        # train_loss = mse_loss + args.alpha * em_penalty + args.beta * em_consistency

        # Loss
        train_loss = criterion(output, hr_img)

        train_loss.backward()
        if args.clip_gradient:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5)  # Clip the gradient to prevent grad explosion
        optimizer.step()

        total_train_loss += train_loss.item()
        # total_train_mse += mse_loss.item()
        # total_train_em_penalty += em_penalty.item()
        # total_train_em_consistency += em_consistency.item()

        # Progress report (TensorBoard + Log)
        x_axis_train = epoch * tr_len
        writer.add_scalar('BatchLoss/Train', train_loss.item(), x_axis_train)
        writer.add_image('Maps/Train/HR', hr_img[tb_img_idx], x_axis_train)
        writer.add_image('Maps/Train/LR', lr_img[tb_img_idx], x_axis_train)
        writer.add_image('Maps/Train/SR', output[tb_img_idx], x_axis_train)
        writer.add_text('Stats/Train', '\n'.join(
            f'{name}| Min {img[tb_img_idx].min().item():.4f} / Max {img[tb_img_idx].max().item():.4f}\n' for name, img
            in zip(['LR', 'HR', 'SR'], [lr_img, hr_img, output])), x_axis_train)
        if not torch.isnan(output[tb_img_idx]).any() and not torch.isinf(output[tb_img_idx]).any():
            writer.add_histogram('Histograms/Train/SR', output[tb_img_idx], x_axis_train)
            writer.add_histogram('Histograms/Train/HR', hr_img[tb_img_idx], x_axis_train)
        # Gradient monitorning
        writer.add_scalar('Gradients', gradient_norm(network), x_axis_train)
        problematic, problematic_params = check_gradients(network)
        if problematic:
            writer.add_text('ProblematicParams', problematic_params, x_axis_train)
            print(f'Problematic gradients found in {problematic_params}')

        # Early validation, when a certain percentage of train batches is processed
        if i == batches_before_valid:
            break  # exit from the inner for loop (TR BATCHES), start the validation

    ###################################### V A L I D A T I O N ######################################
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch}/{args.n_epochs}', unit='batch')):
            hr_img = batch['HR'].cuda()
            lr_img = batch['LR'].cuda()
            output = network(lr_img)

            # If the output has multiple channels, only use the first channel
            if output.shape[1] > 1:
                output = output[:, 0:1, :, :]

            # Check for NaN and Inf
            # check_for_nan_inf(hr_img, 'HR')
            # check_for_nan_inf(lr_img, 'LR')
            # check_for_nan_inf(output, 'SR')

            # Complete loss
            # mse_loss, em_penalty, em_consistency = custom_criterion(output, hr_img, additional_losses=args.additional_losses, upscaling_factor=args.upscaling_factor)
            # val_loss = mse_loss + args.alpha * em_penalty + args.beta * em_consistency

            # Loss
            
            val_loss = criterion(output, hr_img)

            total_val_loss += val_loss.item()
            # total_val_mse += mse_loss.item()
            # total_val_em_penalty += em_penalty.item()
            # total_val_em_consistency += em_consistency.item()

            # Progress report (TensorBoard)
            x_axis_val = epoch * vl_len
            writer.add_scalar('BatchLoss/Val', val_loss.item(), x_axis_val)
            writer.add_image('Maps/Val/HR', hr_img[tb_img_idx], x_axis_val)
            writer.add_image('Maps/Val/LR', lr_img[tb_img_idx], x_axis_val)
            writer.add_image('Maps/Val/SR', output[tb_img_idx], x_axis_val)
            writer.add_text('Stats/Val', '\n'.join(
                f'{name}| Min {img[tb_img_idx].min().item():.4f} / Max {img[tb_img_idx].max().item():.4f}\n' for
                name, img in zip(['LR', 'HR', 'SR'], [lr_img, hr_img, output])), x_axis_val)
            if not torch.isnan(output[tb_img_idx]).any() and not torch.isinf(output[tb_img_idx]).any():
                writer.add_histogram('Histograms/Val/SR', output[tb_img_idx], x_axis_val)
                writer.add_histogram('Histograms/Val/HR', hr_img[tb_img_idx], x_axis_val)

            # Short validation, when a certain percentage of val batches is processed
            if i == batches_before_break_valid:
                break  # exit from the inner for loop (VL BATCHES), pass to the next epoch

    # Epoch Averaged Losses
    avg_train_loss, avg_val_loss = total_train_loss / tr_len, total_val_loss / vl_len
    # avg_train_mse, avg_val_mse = total_train_mse / tr_len, total_val_mse / vl_len
    # avg_train_em_penalty, avg_val_em_penalty = total_train_em_penalty / tr_len, total_val_em_penalty / vl_len
    # avg_train_em_consistency, avg_val_em_consistency = total_train_em_consistency / tr_len, total_val_em_consistency / vl_len

    # Epoch Report (TensorBoard + Log)
    results = compute_metrics(output, hr_img, metrics=metrics, mean=True)  # Compute metrics on the LAST batch ONLY, thus no mean
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)
    writer.add_scalar('Loss/Train/Total', avg_train_loss, epoch)
    writer.add_scalar('Loss/Val/Total', avg_val_loss, epoch)
    writer.add_scalar('Metrics/Val/SSIM', results['SSIM'], epoch)
    writer.add_scalar('Metrics/Val/NMSE', results['NMSE'], epoch)
    writer.add_scalar('Metrics/Val/PSNR', results['PSNR'], epoch)


    # if args.additional_losses:
    #     writer.add_scalar('Loss/Train/MSE', avg_train_mse, epoch)
    #     writer.add_scalar('Loss/Train/Em_Penalty', avg_train_em_penalty, epoch)
    #     writer.add_scalar('Loss/Train/Em_Consistency', avg_train_em_consistency, epoch)
    #     writer.add_scalar('Loss/Val/MSE', avg_val_mse, epoch)
    #     writer.add_scalar('Loss/Val/Em_Penalty', avg_val_em_penalty, epoch)
    #     writer.add_scalar('Loss/Val/Em_Consistency', avg_val_em_consistency, epoch)

    # print(f'Epoch {epoch}/{args.n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
    #       f'\nMSE | Em_Penalty | Em_Consistency: TR {avg_train_mse:.6f} | {avg_train_em_penalty:.6f} | {avg_train_em_consistency:.6f} '
    #       f'| VL {avg_val_mse:.6f} | {avg_val_em_penalty:.6f} | {avg_val_em_consistency:.6f}'
    #       f'\nSSIM: {results["SSIM"]:.4f} | NMSE: {results["NMSE"]:.4f}dB | MAE: {results["MAE"]:.4f}')

    print(f'Epoch {epoch}/{args.n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
          f'\nSSIM: {results["SSIM"]:.4f} | NMSE: {results["NMSE"]:.4f}dB | MAE: {results["MAE"]:.4f}')

    print(f'Learning Rate: {current_lr:.8f}')
    # ———————————————— Epoch Ends ————————————————

    # Save the best model
    if avg_val_loss < best_val_loss - args.es_min_delta:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        best_SSIM = results['SSIM']
        best_NMSE = results['NMSE']
        # save BEST models checkpoints
        torch.save(network.state_dict(), os.path.join(output_dir_path, f'{args.model_name}_best.pth'))
        print(f'Best model saved at epoch {epoch} with loss {best_val_loss:.8f}!')
    else:  # Early stopping, after initial warmup
        if (epoch - best_epoch) > args.es_patience and epoch > args.warmup:
            last_loss = avg_val_loss
            # Save LAST models checkpoints
            torch.save(network.state_dict(), os.path.join(output_dir_path, f'{args.model_name}_last.pth'))
            print(f'Early stopping at epoch {epoch} with loss {last_loss:.8f}!'
                  f'\nBest model saved at epoch {best_epoch} with loss {best_val_loss:.8f}!')
            break  # to exit from the main for loop (EPOCHS)
        else:
            print(f'No improvement. Early stopping in {epoch - best_epoch}/{args.es_patience}')

    # Update learning rate
    if args.scheduler_type == 'global':
        lr_scheduler.step()
    elif args.scheduler_type == 'on-validation':
        lr_scheduler.step(avg_val_loss)

###################################
writer.close()
end_time = datetime.now()
print(f'{end_time}\tTrain ends !\nElapsed time: {end_time - start_time}')
print(f'Best loss {best_val_loss:.4f} and SSIM {best_SSIM:.4f}, NMSE {best_NMSE:.4f}dB at epoch {best_epoch}')
print(f'==> {output_folder_name} <==')
