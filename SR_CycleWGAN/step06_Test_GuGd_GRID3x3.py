import argparse
import pickle
import torch
import torch.nn as nn
import tools
import torchvision.transforms as transforms
import torchvision.datasets as dset
import Data
import random

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device
    ##########################################################################
    ## load the AI output
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--ModelGuFile", type=str, help="None or the path for Gu output")
    parser.add_argument("--ModelGdFile", type=str, help="None or the path for Gd output")
    parser.add_argument("--MaxMinibatchID", type=int, help="the Max Minibatch ID, use this to cut the trainset")
    parser.add_argument("--shuffle", type=int, help="is shuffle the trainset")
    args = parser.parse_args()

    modelGu_file = open(args.ModelGuFile, "rb")  # open the output file
    modelGu = pickle.load(modelGu_file)  # load the output file
    if isinstance(modelGu, nn.DataParallel):
        modelGu = modelGu.module
    modelGu.to('cpu')  # push output to GPU device
    modelGu.eval()  # set the output to evaluation mode, (the dropout layer need this)
    modelGu_file.close()  # close the output file

    modelGd_file = open(args.ModelGdFile, "rb")  # open the output file
    modelGd = pickle.load(modelGd_file)  # load the output file
    if isinstance(modelGd, nn.DataParallel):
        modelGd = modelGd.module
    modelGd.to('cpu')  # push output to GPU device
    modelGd.eval()  # set the output to evaluation mode, (the dropout layer need this)
    modelGd_file.close()  # close the output file

    image_H, image_W = 128*8, 128*14
    minibatch_size = 1
    MAX_MINIBATCH_NUM = args.MaxMinibatchID if args.MaxMinibatchID != None else 1e100
    select_rows, select_cols = 8, 14
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=select_rows, col=select_cols, shuffle=True if args.shuffle else False)
    minibatch_count = min(MAX_MINIBATCH_NUM, len(dataLoader))

    print("Show some images .... ")
    n = random.randint(0, minibatch_count-1)
    ILR, IHR = dataLoader[n]
    with torch.no_grad():
        SHR = modelGu(ILR)
        RLR = modelGd(SHR)
    rand_r, rand_c = random.randint(0, 5), random.randint(0, 11)
    tools.showNineGrid_3x3(SHR[14*(rand_r+0) + (rand_c+0)],  SHR[14*(rand_r+0) + (rand_c+1)],  SHR[14*(rand_r+0) + (rand_c+2)],
                           SHR[14*(rand_r+1) + (rand_c+0)],  SHR[14*(rand_r+1) + (rand_c+1)],  SHR[14*(rand_r+1) + (rand_c+2)],
                           SHR[14*(rand_r+2) + (rand_c+0)],  SHR[14*(rand_r+2) + (rand_c+1)],  SHR[14*(rand_r+2) + (rand_c+2)])
    tools.showNineGrid_3x3(IHR[14*(rand_r+0) + (rand_c+0)],  IHR[14*(rand_r+0) + (rand_c+1)],  IHR[14*(rand_r+0) + (rand_c+2)],
                           IHR[14*(rand_r+1) + (rand_c+0)],  IHR[14*(rand_r+1) + (rand_c+1)],  IHR[14*(rand_r+1) + (rand_c+2)],
                           IHR[14*(rand_r+2) + (rand_c+0)],  IHR[14*(rand_r+2) + (rand_c+1)],  IHR[14*(rand_r+2) + (rand_c+2)])
    tools.showNineGrid_3x3(ILR[14*(rand_r+0) + (rand_c+0)],  ILR[14*(rand_r+0) + (rand_c+1)],  ILR[14*(rand_r+0) + (rand_c+2)],
                           ILR[14*(rand_r+1) + (rand_c+0)],  ILR[14*(rand_r+1) + (rand_c+1)],  ILR[14*(rand_r+1) + (rand_c+2)],
                           ILR[14*(rand_r+2) + (rand_c+0)],  ILR[14*(rand_r+2) + (rand_c+1)],  ILR[14*(rand_r+2) + (rand_c+2)])
    tools.showNineGrid_3x3(RLR[14*(rand_r+0) + (rand_c+0)],  RLR[14*(rand_r+0) + (rand_c+1)],  RLR[14*(rand_r+0) + (rand_c+2)],
                           RLR[14*(rand_r+1) + (rand_c+0)],  RLR[14*(rand_r+1) + (rand_c+1)],  RLR[14*(rand_r+1) + (rand_c+2)],
                           RLR[14*(rand_r+2) + (rand_c+0)],  RLR[14*(rand_r+2) + (rand_c+1)],  RLR[14*(rand_r+2) + (rand_c+2)])