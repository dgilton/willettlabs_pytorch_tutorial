import torch
from src import pytorch_ssim
import numpy as np
from src import misc_utils

def testing_loop(learned_net, test_dataloader, optimizer, output_folder,
                 device='cpu'):
    ind_mse_loss = torch.nn.MSELoss(reduction='none')
    mse_loss = torch.nn.MSELoss()
    loss_accumulator = []
    ssim_accumulator = []

    ssim_loss = pytorch_ssim.SSIM(window_size=15)

    for ii, sample_batch in enumerate(test_dataloader):
        batch_size = sample_batch[0].shape[0]
        optimizer.zero_grad()
        net_input = sample_batch[0].to(device=device)
        target = sample_batch[1].to(device=device)
        bicubic_interp = sample_batch[2].to(device=device)

        net_output = learned_net(net_input) + bicubic_interp
        mse = mse_loss(net_output, target)
        ssim = ssim_loss(net_output, target)


        loss_logger = mse.cpu().detach().numpy()
        ssim_logger = ssim.cpu().detach().numpy()

        name_logger = -10 * np.log10(
            torch.mean(ind_mse_loss(net_output, sample_batch), dim=(1, 2, 3)).cpu().detach().numpy())

        misc_utils.save_batch_as_color_imgs(net_output, batch_size, ii, output_folder, name_logger)

        loss_accumulator.append(loss_logger)
        ssim_accumulator.append(ssim_logger)

    loss_array = np.asarray(loss_accumulator)
    loss_mse = np.mean(loss_array)
    PSNR = -10 * np.log10(loss_mse)
    percentiles = np.percentile(loss_array, [25, 50, 75])
    percentiles = -10.0 * np.log10(percentiles)

    print("MEAN TEST PSNR: " + str(PSNR), flush=True)
    print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
          ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)
    percentiles = np.percentile(np.asarray(ssim_accumulator), [25, 50, 75])
    print("MEAN TEST SSIM: " + str(np.mean(np.asarray(ssim_accumulator))), flush=True)
    print("TEST SSIM QUARTILES AND MEDIAN: " + str(percentiles[0]) +
          ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)