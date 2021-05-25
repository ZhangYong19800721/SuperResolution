from sys import argv
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import tools
import torchvision.transforms as transforms
import torchvision.datasets as dset
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

    image_H, image_W = 1080, 1920
    dataroot = _dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    n = random.randint(0, len(dataset)-1)
    #n = 100
    original_image = dataset[n][0]
    downsamp_image = tools.ImageSuperResolution(original_image, modelGd)
    downsamp_image.show()
    print("show the down sample image")
    original_image.show()
    print("show the original image")