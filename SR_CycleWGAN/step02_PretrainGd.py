# coding=utf-8

########################################################################################################################
# step02_PretrainG.py
########################################################################################################################
from sys import argv
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torchvision.datasets as dset
import torchvision.transforms as transforms
import Model
import Data
import tools


if __name__ == '__main__':
    script, _dataroot, _select_rows, _select_cols, _NGPU, _B_EPOCHS, _N_EPOCHS = argv

    ## set the hyper parameters
    manualSeed = 998
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    DEBUG = True
    N_GPU = int(_NGPU)  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = int(_B_EPOCHS), int(_N_EPOCHS)  # train the model for n epochs
    learn_rate = 0.0005  # set the learning rate
    image_H, image_W = 128 * 8, 128 * 14
    minibatch_size = 1  # set the minibatch size
    MAX_MINIBATCH_NUM = int(1e10)
    select_rows, select_cols = int(_select_rows), int(_select_cols)

    ## set the data set
    dataroot = _dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=select_rows, col=select_cols, shuffle=True)
    minibatch_count = min(MAX_MINIBATCH_NUM, len(dataLoader))

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and N_GPU > 0 else "cpu")

    # show some data samples
    if DEBUG:
        print("Show some images .... ")
        n = random.randint(0, len(dataLoader))
        ILR, IHR = dataLoader[n]
        tools.showNineGrid_3x3(ILR[0], ILR[1], ILR[2], ILR[3], ILR[4], ILR[5], ILR[6], ILR[7], ILR[8])
        tools.showNineGrid_3x3(IHR[0], IHR[1], IHR[2], IHR[3], IHR[4], IHR[5], IHR[6], IHR[7], IHR[8])

    ## create the model and initialize weights
    Gd = Model.GeneratorD()  # create a generator
    Gd.apply(tools.weights_init)  # initialize weights for generator

    # Setup Adam optimizers for both G and D
    optimizerGd = optim.Adam(Gd.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    # Setup the loss function
    MSE = torch.nn.MSELoss()

    ## push models to GPUs
    Gd = Gd.to(device)
    if device.type == 'cuda' and N_GPU > 1:
        Gd = nn.DataParallel(Gd, list(range(N_GPU)))

    print("Start to pretrain Generator .... ")
    alpha1, alpha2 = 0.01, 0.001
    AVE_LOSS_S, AVE_LOSS_L = tools.EXPMA(alpha1), tools.EXPMA(alpha2)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update G network: minimize the MSE Loss
            Gd.zero_grad()  # set the generator gradient to zero
            ILR, IHR = dataLoader[minibatch_id]
            ILR, IHR = ILR.to(device), IHR.to(device)
            RLR = Gd(IHR)
            loss = MSE(RLR, ILR)
            loss.backward()  # back propogation
            # Update D parameters
            optimizerGd.step()

            V_AVE_LOSS_L, V_AVE_LOSS_S = AVE_LOSS_L.expma(loss.item()), AVE_LOSS_S.expma(loss.item())
            message = "Epoch: %5d/%05d, MinibatchID: %5d/%05d, V_AVE_LOSS: % 12.10f[S=% 12.10f]" % (
            epoch, N_EPOCHS, minibatch_id, len(dataLoader), V_AVE_LOSS_L, V_AVE_LOSS_S)
            print(message)

            if minibatch_id % 500 == 0:
                # save model every 1000 iteration
                model_Gd_file = open(r"./model/model_PretrainGd_CPU_%03d.pkl" % epoch, "wb")
                pickle.dump(Gd.to('cpu'), model_Gd_file)
                Gd.to(device)
                model_Gd_file.close()

        # save model every epoch
        model_Gd_file = open(r"./model/model_PretrainGd_CPU_%03d.pkl" % epoch, "wb")
        pickle.dump(Gd.to('cpu'), model_Gd_file)
        Gd.to(device)
        model_Gd_file.close()
        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
