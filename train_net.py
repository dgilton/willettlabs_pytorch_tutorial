import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dbpn_v1 import Net as DenseDBPN
from src.data import get_training_set
from core_training import training_loop
from functools import partial

print = partial(print, flush=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--use_pretrained', type=bool, default=False)
parser.add_argument('--prefix', default='tpami_residual_filter8', help='Location to save checkpoint models')
parser.add_argument('--savepath',
                    default="/share/data/vision-greg2/users/gilton/dbpn_model.pt")

opt = parser.parse_args()


save_location = opt.savepath

gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii))
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!")

print(os.getenv('CUDA_VISIBLE_DEVICES'))
gpu_ids = [int(x) for x in gpu_ids]
# device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]))

use_cuda = torch.cuda.is_available()

torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size,
                             opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')

model = DenseDBPN(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(opt.nEpochs / 2), gamma=0.1)

if os.path.exists(save_location) and opt.use_pretrained:
    if use_cuda:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')
    model.load_state_dict(saved_dict['model_state_dict'])
    optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    scheduler.load_state_dict(saved_dict['scheduler_state_dict'])

if use_dataparallel:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
criterion = nn.L1Loss()

if use_cuda:
    model = model.to(device=device)
    criterion = criterion.to(device=device)


training_loop(model, loss_function=criterion, optimizer=optimizer, scheduler=None,
              train_dataloader = training_data_loader, test_dataloader = None,
              save_every_n_epochs=5, save_location=save_location, device=device,
              n_epochs=opt.nEpochs, use_dataparallel = use_dataparallel)