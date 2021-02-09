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
from src import misc_utils
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


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

# image_location = "/Users/dgilton/PycharmProjects/willettlabs_pytorch_tutorial/data/6046.jpg"
image_location = "/Users/dgilton/PycharmProjects/willettlabs_pytorch_tutorial/data/mandrill.tiff"
save_location = "/Users/dgilton/PycharmProjects/willettlabs_pytorch_tutorial/DBPNLL_8x.pt"

model = DenseDBPN(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)

if os.path.exists(save_location):
    if use_cuda:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')
    model.load_state_dict(saved_dict)

if use_dataparallel:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

if use_cuda:
    model = model.to(device=device)

#########################################
## Loading image
#########################################

pil_img = misc_utils.load_img(image_location)
np_img = np.asarray(pil_img) / 255.0
tensor_img = torch.from_numpy(np_img)
tensor_img = torch.unsqueeze(tensor_img, 0).permute((0,3,1,2))

def convert_to_tensor(img):
    return torch.from_numpy(np.asarray(img)).unsqueeze(0).permute((0,3,1,2)) / 255.0

downsampled_img = misc_utils.rescale_img(pil_img, 1.0 / 8.0)
bicubic_upscale = convert_to_tensor(misc_utils.rescale_img(downsampled_img, 8.0))


downsampled_tensor = convert_to_tensor(downsampled_img)

reconstruction = bicubic_upscale + model(downsampled_tensor)

numpy_reconstruction = reconstruction.detach().cpu().numpy()[0,:,:,:]
numpy_reconstruction = np.transpose(numpy_reconstruction, (1,2,0))

plt.imshow(numpy_reconstruction)
plt.show()

