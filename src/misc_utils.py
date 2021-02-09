import matplotlib.pyplot as plt
import numpy as np
import imageio

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