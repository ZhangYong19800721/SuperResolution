# coding=utf-8

########################################################################################################################
# step04_Train_CycleSPWGAN.py
# train the output, include parameters initializing
########################################################################################################################
import os
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

"""
--seed=998
--dataroot=/home/zhangyong/Data/image2160x3840
--learn_rate=0.0005
--minibatch_size=1
--select_rows=3
--select_cols=4
--NGPU=1
--B_EPOCHS=1
--N_EPOCHS=10000
--outputDir=./output
--logDir=./logdir
--shuffle=1
--isLoadGu=./output/20210525[11:32:25]/model_Gu_CPU.pkl
--isLoadGd=./output/20210525[11:32:25]/model_Gd_CPU.pkl
--isLoadD=./output/20210525[11:32:25]/model_D_SP_CPU.pkl
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="The manual random seed")
    parser.add_argument("--shuffle", type=int, help="is shuffle the trainset")
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--learn_rate", type=float, help="The learn rate")
    parser.add_argument("--minibatch_size", type=int, help="The learn rate")
    parser.add_argument("--select_rows", type=int, help="The frame is cut into 8x14 patches, select how many rows to use")
    parser.add_argument("--select_cols", type=int, help="The frame is cut into 8x14 patches, select how many cols to use")
    parser.add_argument("--NGPU", type=int, help="specify the number of GPUs to use")
    parser.add_argument("--B_EPOCHS", type=int, help="The start epoch id")
    parser.add_argument("--N_EPOCHS", type=int, help="The end epoch id")
    parser.add_argument("--alfa", type=float, help="the weight for loss_G_D")
    parser.add_argument("--beda", type=float, help="The weight for loss_reco")
    parser.add_argument("--isLoadGu", type=str, help="None or the path for pretrained Gu model")
    parser.add_argument("--isLoadGd", type=str, help="None or the path for pretrained Gd model")
    parser.add_argument("--isLoadD", type=str, help="None or the path for pretrained D model")
    parser.add_argument("--outputDir", type=str, help="the output directory")
    parser.add_argument("--logDir", type=str, help="The log dir")
    args = parser.parse_args()

    open_time_str = time.strftime("%Y%m%d[%H:%M:%S]", time.localtime())
    os.mkdir(args.outputDir + "/" + open_time_str)
    writer = SummaryWriter(args.logDir + "/" + open_time_str)

    ## set the hyper parameters
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    image_H, image_W = 128 * 8, 128 * 14

    ## set the data set
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=args.minibatch_size, select_row=args.select_rows, select_col=args.select_cols,
                                 shuffle=True if args.shuffle else False)
    minibatch_count = len(dataLoader)

    ## specify the computing device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.NGPU > 0 else "cpu")
    # device = torch.device("cpu")

    # show some data samples
    print("Show some images ...., press ENTER to continue. ")
    n = random.randint(0, len(dataLoader) - 1)
    ILR, IHR = dataLoader[n]
    tools.showNineGrid_3x3(ILR[0], ILR[1], ILR[2], ILR[3], ILR[4], ILR[5], ILR[6], ILR[7], ILR[8])
    tools.showNineGrid_3x3(IHR[0], IHR[1], IHR[2], IHR[3], IHR[4], IHR[5], IHR[6], IHR[7], IHR[8])

    if args.isLoadGu:
        ##########################################################################
        ## load the pretrained G output
        modelGu_file = open(args.isLoadGu, "rb")  # open the output file
        Gu = pickle.load(modelGu_file)  # load the output file
        if isinstance(Gu, nn.DataParallel):
            Gu = Gu.module
        Gu.to(device)  # push output to GPU device
        modelGu_file.close()  # close the output file
    else:
        Gu = Model.GeneratorUx2()  # create a generator
        Gu.apply(tools.weights_init)  # initialize weights for generator

    if args.isLoadGd:
        ##########################################################################
        ## load the pretrained G output
        modelGd_file = open(args.isLoadGd, "rb")  # open the output file
        Gd = pickle.load(modelGd_file)  # load the output file
        if isinstance(Gd, nn.DataParallel):
            Gd = Gd.module
        Gd.to(device)  # push output to GPU device
        modelGd_file.close()  # close the output file
    else:
        Gd = Model.GeneratorDx2()  # create a generator
        Gd.apply(tools.weights_init)  # initialize weights for generator

    if args.isLoadD:
        ##########################################################################
        ## load the pretrained Db output
        modelD_file = open(args.isLoadD, "rb")  # open the output file
        D = pickle.load(modelD_file)  # load the output file
        if isinstance(D, nn.DataParallel):
            D = D.module
        D.to(device)  # push output to GPU device
        modelD_file.close()  # close the output file
    else:
        D = Model.DiscriminatorSPx2()  # create a discriminator
        D.apply(tools.weights_init)  # initialize weights for discriminator

    # Initialize BCE and MSE function
    MSE = nn.MSELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(D.parameters(), lr=args.learn_rate, betas=(0.5, 0.999))
    optimizerGu = optim.Adam(Gu.parameters(), lr=args.learn_rate, betas=(0.5, 0.999))
    optimizerGd = optim.Adam(Gd.parameters(), lr=args.learn_rate, betas=(0.5, 0.999))

    ## push models to GPUs
    Gu = Gu.to(device)
    Gd = Gd.to(device)
    D = D.to(device)
    if device.type == 'cuda' and args.NGPU > 1:
        Gu = nn.DataParallel(Gu, list(range(args.NGPU)))
        Gd = nn.DataParallel(Gd, list(range(args.NGPU)))
        D = nn.DataParallel(D, list(range(args.NGPU)))

    print("Start to train .... ")
    alpha = 0.01
    AVE_DIFF = tools.EXPMA(alpha)
    AVE_MMSE = tools.EXPMA(alpha)
    AVE_RMSE = tools.EXPMA(alpha)

    # leakyRELU = nn.LeakyReLU(0.0)
    for epoch in range(args.B_EPOCHS, args.N_EPOCHS + 1):
        start_time = time.time()
        for minibatch_id in range(1, minibatch_count + 1):
            ## Update D network:
            # train with all-real batch
            ILR, IHR = dataLoader[minibatch_id - 1]
            ILR, IHR = ILR.to(device), IHR.to(device)

            ## Update D network: for WGAN maximize D(x) - D(G(z))
            D.zero_grad()  # set discriminator gradient to zero
            ISR = Gu(ILR).detach()
            output_real_D = D(IHR)
            output_fake_D = D(ISR)
            diff = (output_real_D - output_fake_D).mean()
            loss_D = -diff
            loss_D.backward()
            optimizerD.step()

            Gu.zero_grad()  # set the generator gradient to zero
            Gd.zero_grad()
            ISR = Gu(ILR)
            RLR = Gd(ISR)
            output_fake_G_D = D(ISR)
            loss_mmse = MSE(ILR, RLR)
            loss_reco = MSE(ISR, IHR)
            loss_G_D = -output_fake_G_D.mean()

            loss_G = loss_mmse + args.alfa * loss_G_D + args.beda * loss_reco
            loss_G.backward()
            optimizerGu.step()  # Update Gu parameters
            optimizerGd.step()  # Update Gd parameters

            V_AVE_DIFF = AVE_DIFF.expma(abs(diff.item()))
            V_AVE_MMSE = AVE_MMSE.expma(loss_mmse.item())
            V_AVE_RMSE = AVE_RMSE.expma(loss_reco.item())

            message = "Epoch:%3d, MinibatchID:%5d/%05d, DIFF:% 6.12f, MMSE: % 6.12f, RMSE: % 6.12f" % (
                epoch, minibatch_id, minibatch_count, V_AVE_DIFF, V_AVE_MMSE, V_AVE_RMSE)
            print(message)

            istep = minibatch_count * (epoch - args.B_EPOCHS) + minibatch_id
            writer.add_scalar("AVE_DIFF", V_AVE_DIFF, istep)
            writer.add_scalar("AVE_MMSE", V_AVE_MMSE, istep)
            writer.add_scalar("AVE_RMSE", V_AVE_RMSE, istep)

            if istep % 300 == 0:
                # save output every 1000 iteration
                model_Gu_file = open(args.outputDir + "/" + open_time_str + "/model_Gu_CPU.pkl", "wb")
                model_Gd_file = open(args.outputDir + "/" + open_time_str + "/model_Gd_CPU.pkl", "wb")
                model_D_file = open(args.outputDir + "/" + open_time_str + "/model_D_SP_CPU.pkl", "wb")
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
