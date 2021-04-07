# coding=utf-8

########################################################################################################################
# step04_Train.py
# train the UHD_GP_WGAN model, include parameters initializing
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
    manualSeed = 989
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    DEBUG = True
    N_GPU = int(_NGPU)  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = int(_B_EPOCHS), int(_N_EPOCHS)  # train the model for n epochs
    learn_rate = 0.0005  # set the learning rate
    image_H, image_W = 128 * 8, 128 * 14
    minibatch_size = 1  # set the minibatch size
    isLoadPretrainedGu, isLoadPretrainedGd, isLoadPretrainedD = True, True, False
    MAX_MINIBATCH_NUM = int(1e10)
    select_rows, select_cols = int(_select_rows), int(_select_cols)

    ## set the data set
    dataroot = _dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=select_rows, col=select_cols, shuffle=True)
    minibatch_count = min(MAX_MINIBATCH_NUM, len(dataLoader))

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and N_GPU > 0 else "cpu")
    # device = torch.device("cpu")

    # show some data samples
    if DEBUG:
        print("Show some images ...., press ENTER to continue. ")
        n = random.randint(0, len(dataLoader))
        ILR, IHR = dataLoader[n + 1]
        tools.showNineGrid_3x3(ILR[0], ILR[1], ILR[2], ILR[3], ILR[4], ILR[5], ILR[6], ILR[7], ILR[8])
        tools.showNineGrid_3x3(IHR[0], IHR[1], IHR[2], IHR[3], IHR[4], IHR[5], IHR[6], IHR[7], IHR[8])

    if isLoadPretrainedGu:
        ##########################################################################
        ## load the pretrained G model
        modelGu_file = open("./model/model_PretrainGu_CPU_002.pkl", "rb")  # open the model file
        Gu = pickle.load(modelGu_file)  # load the model file
        if isinstance(Gu, nn.DataParallel):
            Gu = Gu.module
        Gu.to(device)  # push model to GPU device
        modelGu_file.close()  # close the model file
    else:
        Gu = Model.GeneratorU()  # create a generator
        Gu.apply(tools.weights_init)  # initialize weights for generator

    if isLoadPretrainedGd:
        ##########################################################################
        ## load the pretrained G model
        modelGd_file = open("./model/model_PretrainGd_CPU_005.pkl", "rb")  # open the model file
        Gd = pickle.load(modelGd_file)  # load the model file
        if isinstance(Gd, nn.DataParallel):
            Gd = Gd.module
        Gd.to(device)  # push model to GPU device
        modelGd_file.close()  # close the model file
    else:
        Gd = Model.GeneratorD()  # create a generator
        Gd.apply(tools.weights_init)  # initialize weights for generator

    if isLoadPretrainedD:
        ##########################################################################
        ## load the pretrained Db model
        modelD_file = open("model/model_PretrainD_GP_CPU_000.pkl", "rb")  # open the model file
        D = pickle.load(modelD_file)  # load the model file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push model to GPU device
        modelD_file.close()  # close the model file
    else:
        D = Model.Discriminator()  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Initialize BCE and MSE function
    MSE = nn.MSELoss(reduction='mean')

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=1.0 * learn_rate, betas=(0.9, 0.999))
    optimizerGu = optim.Adam(Gu.parameters(), lr=1.0 * learn_rate, betas=(0.9, 0.999))
    optimizerGd = optim.Adam(Gd.parameters(), lr=1.0 * learn_rate, betas=(0.9, 0.999))

    ## push models to GPUs
    Gu = Gu.to(device)
    Gd = Gd.to(device)
    D = D.to(device)
    if device.type == 'cuda' and N_GPU > 1:
        Gu = nn.DataParallel(Gu, list(range(N_GPU)))
        Gd = nn.DataParallel(Gd, list(range(N_GPU)))
        D  = nn.DataParallel(D,  list(range(N_GPU)))

    print("Start to train .... ")
    alpha1, alpha2 = 0.001, 1.000
    AVE_DIFF_S = tools.EXPMA(alpha2)
    AVE_DIFF_L = tools.EXPMA(alpha1)
    AVE_REAL_S = tools.EXPMA(alpha2)
    AVE_REAL_L = tools.EXPMA(alpha1)
    AVE_FAKE_S = tools.EXPMA(alpha2)
    AVE_FAKE_L = tools.EXPMA(alpha1)
    AVE_GRDP_S = tools.EXPMA(alpha2)
    AVE_GRDP_L = tools.EXPMA(alpha1)
    AVE_MMSE_S = tools.EXPMA(alpha2)
    AVE_MMSE_L = tools.EXPMA(alpha1)

    # leakyRELU = nn.LeakyReLU(0.0)
    for epoch in range(B_EPOCHS, N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update D network:
            # train with all-real batch
            ILR, IHR = dataLoader[minibatch_id]
            ILR, IHR = ILR.to(device), IHR.to(device)

            ## Update D network: for WGAN maximize D(x) - D(G(z))
            D.zero_grad()  # set discriminator gradient to zero
            ISR = Gu(ILR)
            output_real_D = D(IHR)
            output_fake_D = D(ISR)
            diff = (output_real_D - output_fake_D).mean()
            gradient_penalty = tools.cal_gradient_penalty(D, device, IHR, ISR)
            loss = -diff + gradient_penalty
            loss.backward()
            optimizerD.step()

            Gu.zero_grad()  # set the generator gradient to zero
            Gd.zero_grad()
            ISR = Gu(ILR)
            RLR = Gd(ISR)
            output_fake_G_D = D(ISR)
            loss_recon_mmse = MSE(ILR, RLR)
            loss_G_D = -output_fake_G_D.mean() + loss_recon_mmse
            loss_G_D.backward()
            optimizerGu.step()  # Update Gu parameters
            optimizerGd.step()  # Update Gd parameters

            V_AVE_DIFF_S = AVE_DIFF_S.expma(diff.item())
            V_AVE_DIFF_L = AVE_DIFF_L.expma(diff.item())
            V_AVE_REAL_S = AVE_REAL_S.expma(output_real_D.mean().item())
            V_AVE_REAL_L = AVE_REAL_L.expma(output_real_D.mean().item())
            V_AVE_FAKE_S = AVE_FAKE_S.expma(output_fake_D.mean().item())
            V_AVE_FAKE_L = AVE_FAKE_L.expma(output_fake_D.mean().item())
            V_AVE_GRDP_S = AVE_GRDP_S.expma(gradient_penalty.mean().item())
            V_AVE_GRDP_L = AVE_GRDP_L.expma(gradient_penalty.mean().item())
            V_AVE_MMSE_S = AVE_MMSE_S.expma(loss_recon_mmse.mean().item())
            V_AVE_MMSE_L = AVE_MMSE_L.expma(loss_recon_mmse.mean().item())

            message = "Epoch: %5d, MinibatchID: %5d/%05d, DIFF:% 10.12f[s=% 10.12f], REAL: % 10.12f[s=% 10.12f], FAKE: % 10.12f[s=% 10.12f], GRDP: % 10.12f[s=% 10.12f], MMSE: % 10.12f[s=% 10.12f]" % (
                epoch, minibatch_id, minibatch_count,
                V_AVE_DIFF_L, V_AVE_DIFF_S, V_AVE_REAL_L, V_AVE_REAL_S, V_AVE_FAKE_L, V_AVE_FAKE_S, V_AVE_GRDP_L, V_AVE_GRDP_S, V_AVE_MMSE_L, V_AVE_MMSE_S
            )
            print(message)

            if minibatch_id % 500 == 0:
                # save model every 1000 iteration
                model_Gu_file = open(r"./model/model_Gu_CPU_%03d.pkl" % epoch, "wb")
                model_Gd_file = open(r"./model/model_Gd_CPU_%03d.pkl" % epoch, "wb")
                model_D_file = open(r"./model/model_D_CPU_%03d.pkl" % epoch, "wb")
                pickle.dump(Gu.to("cpu"), model_Gu_file)
                pickle.dump(Gd.to("cpu"), model_Gd_file)
                pickle.dump(D.to("cpu"), model_D_file)
                Gu.to(device)
                Gd.to(device)
                D.to(device)
                model_Gu_file.close()
                model_Gd_file.close()
                model_D_file.close()

        # save model every epoch
        model_Gu_file = open(r"./model/model_Gu_CPU_%03d.pkl" % epoch, "wb")
        model_Gd_file = open(r"./model/model_Gd_CPU_%03d.pkl" % epoch, "wb")
        model_D_file = open(r"./model/model_D_CPU_%03d.pkl" % epoch, "wb")
        pickle.dump(Gu.to("cpu"), model_Gu_file)
        pickle.dump(Gd.to("cpu"), model_Gd_file)
        pickle.dump(D.to("cpu"), model_D_file)
        Gu.to(device)
        Gd.to(device)
        D.to(device)
        model_Gu_file.close()
        model_Gd_file.close()
        model_D_file.close()
        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
