# coding=utf-8

########################################################################################################################
# step04_Train.py
# train the UHD_WGANGP model, include parameters initializing
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
    manualSeed = 997
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    DEBUG = True
    N_GPU = 2  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = 0, 200  # train the model for n epochs
    learn_rate = 0.0005  # set the learning rate
    image_H, image_W = 128 * 8, 128 * 14
    minibatch_size = 4  # set the minibatch size
    isLoadPretrainedD = False
    MAX_MINIBATCH_NUM = int(1e10)
    selected_rows, selected_cols = 8, 14

    ## set the data set
    dataroot = "/home/zhangyong/Data/image2160x3840"
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=selected_rows, col=selected_cols, shuffle=True)
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

    ##########################################################################
    ## load the Gu model
    modelGu_file = open("./model/model_PretrainGu_CPU_002.pkl", "rb")  # open the model file
    Gu = pickle.load(modelGu_file)  # load the model file
    if isinstance(Gu, nn.DataParallel):
        Gu = Gu.module
    Gu.to(device)  # push model to GPU device
    modelGu_file.close()  # close the model file

    if isLoadPretrainedD:
        ##########################################################################
        ## load the pretrained D model
        modelD_file = open("model/model_PretrainD_SP_CPU_000.pkl", "rb")  # open the model file
        D = pickle.load(modelD_file)  # load the model file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push model to GPU device
        modelD_file.close()  # close the model file
    else:
        D = Model.Discriminator_SP()  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    ## push models to GPUs
    D = D.to(device)

    # Data Parallel
    if device.type == 'cuda' and N_GPU > 1:
        Gu = nn.DataParallel(Gu, list(range(N_GPU)))
        D = nn.DataParallel(D, list(range(N_GPU)))

    print("Start to train .... ")
    alpha1, alpha2 = 0.001, 1.000
    AVE_DIFF_L, AVE_DIFF_S = tools.EXPMA(alpha1), tools.EXPMA(alpha2)
    AVE_REAL_L, AVE_REAL_S = tools.EXPMA(alpha1), tools.EXPMA(alpha2)
    AVE_FAKE_L, AVE_FAKE_S = tools.EXPMA(alpha1), tools.EXPMA(alpha2)

    for epoch in range(B_EPOCHS, N_EPOCHS):
        start_time = time.time()
        for minibatch_id in range(minibatch_count):
            ## Update D network: for WGAN maximize D(x) - D(G(z))
            # train with all-real batch
            D.zero_grad()  # set discriminator gradient to zero
            ILR, IHR = dataLoader[minibatch_id]
            ILR, IHR = ILR.to(device), IHR.to(device)
            with torch.no_grad():
                ISR = Gu(ILR)
            output_real_D = D(IHR)
            output_fake_D = D(ISR)
            diff = (output_real_D - output_fake_D).mean()
            loss = -diff
            loss.backward()
            optimizerD.step()

            V_AVE_DIFF_L = AVE_DIFF_L.expma(diff.item())
            V_AVE_DIFF_S = AVE_DIFF_S.expma(diff.item())
            V_AVE_REAL_L = AVE_REAL_L.expma(output_real_D.mean().item())
            V_AVE_REAL_S = AVE_REAL_S.expma(output_real_D.mean().item())
            V_AVE_FAKE_L = AVE_FAKE_L.expma(output_fake_D.mean().item())
            V_AVE_FAKE_S = AVE_FAKE_S.expma(output_fake_D.mean().item())

            message = "Epoch: %5d/%05d, MinibatchID: %5d/%05d, DIFF:% 12.10f[S=% 12.10f], REAL:% 12.10f[S=% 12.10f], FAKE:% 12.10f[S=% 12.10f]" % (
                epoch, N_EPOCHS, minibatch_id, minibatch_count,
                V_AVE_DIFF_L, V_AVE_DIFF_S,
                V_AVE_REAL_L, V_AVE_REAL_S,
                V_AVE_FAKE_L, V_AVE_FAKE_S)
            print(message)

            if minibatch_id % 500 == 0:
                # save model every 500 iteration
                model_D_file = open(r"./model/model_PretrainD_CPU_%03d.pkl" % epoch, "wb")
                pickle.dump(D.to("cpu"), model_D_file)
                D.to(device)
                model_D_file.close()

        # save model every epoch
        model_D_file = open(r"./model/model_PretrainD_CPU_%03d.pkl" % epoch, "wb")
        pickle.dump(D.to("cpu"), model_D_file)
        D.to(device)
        model_D_file.close()
        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')
