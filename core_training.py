import torch

def training_loop(learned_net, train_dataloader, test_dataloader, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    for epoch in range(start_epoch, n_epochs):

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            net_input = sample_batch[0].to(device=device)
            target = sample_batch[1].to(device=device)
            bicubic_interp = sample_batch[2].to(device=device)

            net_output = learned_net(net_input) + bicubic_interp
            loss = loss_function(net_output, target)
            loss.backward()
            optimizer.step()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)
        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                net_state = learned_net.module.state_dict()
            else:
                net_state = learned_net.state_dict()
            torch.save({'model_state_dict': net_state,
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)