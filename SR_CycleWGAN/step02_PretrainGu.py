# coding=utf-8

########################################################################################################################
# step02_PretrainG.py
########################################################################################################################
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
    ## set the hyper parameters
    manualSeed = 998
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    DEBUG = True
    N_GPU = 2  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = 0, 200  # train the model for n epochs
    learn_rate = 0.0005  # set the learning rate
    image_H, image_W = 128 * 8, 128 * 14
    minibatch_size = 1  # set the minibatch size
    MAX_MINIBATCH_NUM = int(1e10)
    select_rows, select_cols = 8, 12

    ## set the data set
    dataroot = "/home/zhangyong/Data/image2160x3840"
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
    Gu = Model.GeneratorU()  # create a generator
    Gu.apply(tools.weights_init)  # initialize weights for generator

    # Setup Adam optimizers for both G and D
    optimizerGu = optim.Adam(Gu.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    # Setup the loss function
    MSE = torch.nn.MSELoss()

    ## push models to GPUs
    Gu = Gu.to(device)
    if device.type == 'cuda' and N_GPU > 1:
        Gu = nn.DataParallel(Gu, list(range(N_GPU)))

    print("Start to pretrain Generator .... ")
    alpha1, alpha2 = 0.01, 0.001
    AVE_LOSS_S, AVE_LOSS_L = tools.EXPMA(alpha1), tools.EXPMA(alpha2)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update G network: minimize the MSE Loss
            Gu.zero_grad()  # set the generator gradient to zero
            ILR, IHR = dataLoader[minibatch_id]
            ILR, IHR = ILR.to(device), IHR.to(device)
            ISR = Gu(ILR)
            loss = MSE(ISR, IHR)
            loss.backward()  # back propogation
            # Update D parameters
            optimizerGu.step()

            V_AVE_LOSS_L, V_AVE_LOSS_S = AVE_LOSS_L.expma(loss.item()), AVE_LOSS_S.expma(loss.item())
            message = "Epoch: %5d/%05d, MinibatchID: %5d/%05d, V_AVE_LOSS: % 12.10f[S=% 12.10f]" % (
                epoch, N_EPOCHS, minibatch_id, len(dataLoader), V_AVE_LOSS_L, V_AVE_LOSS_S)
            print(message)

            if minibatch_id % 500 == 0:
                # save model every 1000 iteration
                model_Gu_file = open(r"./model/model_PretrainGu_CPU_%03d.pkl" % epoch, "wb")
                pickle.dump(Gu.to('cpu'), model_Gu_file)
                Gu.to(device)
                model_Gu_file.close()

        # save model every epoch
        model_Gu_file = open(r"./model/model_PretrainGu_CPU_%03d.pkl" % epoch, "wb")
        pickle.dump(Gu.to('cpu'), model_Gu_file)
        Gu.to(device)
        model_Gu_file.close()
        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
