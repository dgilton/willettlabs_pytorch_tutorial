import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from PIL import Image
import re
import torch.utils.data as datautils

def save_tensor_as_color_img(img_tensor, filename):
    np_array = img_tensor.cpu().detach().numpy()
    imageio.save(filename, np_array)

def save_batch_as_color_imgs(tensor_batch, batch_size, ii, folder_name, names):
    numpy_array = tensor_batch.cpu().detach().numpy()
    bwhc_array = np.transpose(numpy_array,(0,2,3,1))
    img_array = (np.clip(bwhc_array,0,1) + 1.0) *  255

    img_array = img_array.astype(np.uint8)
    for kk in range(batch_size):
        img_number = batch_size*ii + kk
        filename = folder_name + str(img_number) + "_" + str(names[kk]) + ".png"
        imageio.imwrite(filename, img_array[kk,...])


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def directory_filelist(target_directory):
    file_list = [f for f in sorted(os.listdir(target_directory))
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list

def load_img(file_name):
    with open(file_name,'rb') as f:
        img = Image.open(f).convert("RGB")
    return img

class FolderDataset(datautils.Dataset):
    def __init__(self, target_directory, transform=None):
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]
        self.transform = transform

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        return data

def rescale_img(input_img, scale):
    input_size = input_img.size
    output_size = tuple([int(dim * scale) for dim in input_size])
    output_img = input_img.resize(output_size, resample=Image.BICUBIC)
    return output_img