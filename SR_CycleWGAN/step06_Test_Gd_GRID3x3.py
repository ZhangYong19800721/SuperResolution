from sys import argv
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import tools
import torchvision.transforms as transforms
import torchvision.datasets as dset
import Data
import random
import torchvision.transforms.functional as ttf

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device
    ##########################################################################
    ## load the AI output
    script, _dataroot, _modelGdFile = argv

    modelGd_file = open(_modelGdFile, "rb")  # open the output file
    modelGd = pickle.load(modelGd_file)  # load the output file
    if isinstance(modelGd, nn.DataParallel):
        modelGd = modelGd.module
    modelGd.to('cpu')  # push output to GPU device
    modelGd.eval()  # set the output to evaluation mode, (the dropout layer need this)
    modelGd_file.close()  # close the output file

    image_H, image_W = 256*4, 256*7
    minibatch_size = 1
    select_rows, select_cols = 8, 14
    dataroot = _dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = Data.DataLoader(dataset, minibatch_size=minibatch_size, row=select_rows, col=select_cols, shuffle=False)


    n = random.randint(0, len(dataLoader)-1)
    ILR, IHR = dataLoader[n]
    with torch.no_grad():
        RLR = modelGd(IHR)
    rand_r, rand_c = random.randint(0, 5), random.randint(0, 11)
    print("Show original images patches")
    tools.showNineGrid_3x3(IHR[14*(rand_r+0) + (rand_c+0)],  IHR[14*(rand_r+0) + (rand_c+1)],  IHR[14*(rand_r+0) + (rand_c+2)],
                           IHR[14*(rand_r+1) + (rand_c+0)],  IHR[14*(rand_r+1) + (rand_c+1)],  IHR[14*(rand_r+1) + (rand_c+2)],
                           IHR[14*(rand_r+2) + (rand_c+0)],  IHR[14*(rand_r+2) + (rand_c+1)],  IHR[14*(rand_r+2) + (rand_c+2)])
    print("Show downsample images patches")
    tools.showNineGrid_3x3(ILR[14*(rand_r+0) + (rand_c+0)],  ILR[14*(rand_r+0) + (rand_c+1)],  ILR[14*(rand_r+0) + (rand_c+2)],
                           ILR[14*(rand_r+1) + (rand_c+0)],  ILR[14*(rand_r+1) + (rand_c+1)],  ILR[14*(rand_r+1) + (rand_c+2)],
                           ILR[14*(rand_r+2) + (rand_c+0)],  ILR[14*(rand_r+2) + (rand_c+1)],  ILR[14*(rand_r+2) + (rand_c+2)])
    print("Show reconstruction images patches")
    tools.showNineGrid_3x3(RLR[14*(rand_r+0) + (rand_c+0)],  RLR[14*(rand_r+0) + (rand_c+1)],  RLR[14*(rand_r+0) + (rand_c+2)],
                           RLR[14*(rand_r+1) + (rand_c+0)],  RLR[14*(rand_r+1) + (rand_c+1)],  RLR[14*(rand_r+1) + (rand_c+2)],
                           RLR[14*(rand_r+2) + (rand_c+0)],  RLR[14*(rand_r+2) + (rand_c+1)],  RLR[14*(rand_r+2) + (rand_c+2)])