# coding=utf-8

########################################################################################################################
# step04_Train.py
# train the UHD_GP_WGAN model, include parameters initializing
########################################################################################################################
import argparse
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
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The manual random seed")
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--learn_rate", type=float, help="The learn rate")
    parser.add_argument("--minibatch_size", type=int, help="The learn rate")
    parser.add_argument("--select_rows", type=int, help="The frame is cut into 8x14 patches, select how many rows to use")
    parser.add_argument("--select_cols", type=int, help="The frame is cut into 8x14 patches, select how many cols to use")
    parser.add_argument("--NGPU", type=int, help="specify the number of GPUs to use")
    parser.add_argument("--B_EPOCHS", type=int, help="The start epoch id")
    parser.add_argument("--N_EPOCHS", type=int, help="The end epoch id")
    parser.add_argument("--isLoadPretrainedGu", type=str, help="None or the path for pretrained Gu model")
    parser.add_argument("--isLoadPretrainedD", type=str, help="None or the path for pretrained D model")
    parser.add_argument("--logdir", type=str, help="The log dir")
    args = parser.parse_args()

    writer = SummaryWriter(args.logdir + "/Train_Log_" + time.strftime("%Y%m%d[%H:%M:%S]", time.localtime()))

    ## set the hyper parameters
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    DEBUG = True
    N_GPU = args.NGPU  # we have 2 GPUs
    B_EPOCHS, N_EPOCHS = args.B_EPOCHS, args.N_EPOCHS  # train the model for n epochs
    learn_rate = args.learn_rate  # set the learning rate
    image_H, image_W = 128 * 8, 128 * 14
    minibatch_size = args.minibatch_size  # set the minibatch size
    isLoadPretrainedGu, isLoadPretrainedD = args.isLoadPretrainedGu, args.isLoadPretrainedD
    MAX_MINIBATCH_NUM = int(1e10)
    select_rows, select_cols = args.select_rows, args.select_cols

    ## set the data set
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=select_rows, col=select_cols, shuffle=True)
    minibatch_count = min(MAX_MINIBATCH_NUM, len(dataLoader))

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and N_GPU > 0 else "cpu")

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
        modelGu_file = open(isLoadPretrainedGu, "rb")  # open the model file
        Gu = pickle.load(modelGu_file)  # load the model file
        if isinstance(Gu, nn.DataParallel):
            Gu = Gu.module
        Gu.to(device)  # push model to GPU device
        modelGu_file.close()  # close the model file
    else:
        Gu = Model.GeneratorU()  # create a generator
        Gu.apply(tools.weights_init)  # initialize weights for generator

    if isLoadPretrainedD:
        ##########################################################################
        ## load the pretrained Db model
        modelD_file = open(isLoadPretrainedD, "rb")  # open the model file
        D = pickle.load(modelD_file)  # load the model file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push model to GPU device
        modelD_file.close()  # close the model file
    else:
        D = Model.Discriminator_SP()  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Initialize BCE and MSE function
    MSE = nn.MSELoss(reduction='mean')

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=learn_rate, betas=(0.9, 0.999))
    optimizerGu = optim.Adam(Gu.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    ## push models to GPUs
    Gu = Gu.to(device)
    D = D.to(device)
    if device.type == 'cuda' and N_GPU > 1:
        Gu = nn.DataParallel(Gu, list(range(N_GPU)))
        D = nn.DataParallel(D, list(range(N_GPU)))

    print("Start to train .... ")
    alpha = 0.01
    AVE_DIFF = tools.EXPMA(alpha)
    AVE_MMSE = tools.EXPMA(alpha)

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
            loss = -diff
            loss.backward()
            optimizerD.step()

            Gu.zero_grad()  # set the generator gradient to zero
            ISR = Gu(ILR)
            output_fake_G_D = D(ISR)
            loss_optim_mmse = MSE(ISR, IHR)
            loss_G_D = -1e-3 * output_fake_G_D.mean() + loss_optim_mmse
            loss_G_D.backward()
            optimizerGu.step()  # Update Gu parameters

            V_AVE_DIFF = AVE_DIFF.expma(abs(diff.item()))
            V_AVE_MMSE = AVE_MMSE.expma(loss_optim_mmse.mean().item())

            message = "Epoch:%3d, MinibatchID:%5d/%05d, DIFF:% 6.12f, HMSE: % 6.12f" % (
                epoch, minibatch_id, minibatch_count, V_AVE_DIFF, V_AVE_MMSE)
            print(message)

            writer.add_scalar("AVE_DIFF", V_AVE_DIFF, minibatch_count * (epoch - B_EPOCHS) + minibatch_id)
            writer.add_scalar("AVE_MMSE", V_AVE_MMSE, minibatch_count * (epoch - B_EPOCHS) + minibatch_id)

            if minibatch_id % 500 == 0:
                # save model every 1000 iteration
                model_Gu_file = open(r"./model/model_Gu_CPU_%03d.pkl" % epoch, "wb")
                model_D_file = open(r"./model/model_D_SP_CPU_%03d.pkl" % epoch, "wb")
                pickle.dump(Gu.to("cpu"), model_Gu_file)
                pickle.dump(D.to("cpu"), model_D_file)
                Gu.to(device)
                D.to(device)
                model_Gu_file.close()
                model_D_file.close()

        # save model every epoch
        model_Gu_file = open(r"./model/model_Gu_CPU_%03d.pkl" % epoch, "wb")
        model_D_file = open(r"./model/model_D_SP_CPU_%03d.pkl" % epoch, "wb")
        pickle.dump(Gu.to("cpu"), model_Gu_file)
        pickle.dump(D.to("cpu"), model_D_file)
        Gu.to(device)
        D.to(device)
        model_Gu_file.close()
        model_D_file.close()
        end_time = time.time()
        print(f'train_time_for_epoch = {(end_time - start_time) / 60} min')