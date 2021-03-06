import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--ModelGuFile", type=str, help="None or the path for Gu output")
    parser.add_argument("--ModelGdFile", type=str, help="None or the path for Gd output")
    parser.add_argument("--sampleID", type=int, help="specify the target frame")
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

    image_H, image_W = 1080, 1920
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    n = args.sampleID if args.sampleID else random.randint(0, len(dataset)-1)
    original_image = dataset[n][0]
    downsamp_image = tools.imResize(original_image, (1080 // 2, 1920 // 2))
    upsample_image = tools.imResize(downsamp_image, (1080 // 1, 1920 // 1))
    downsamp_image.show()
    print("show the down sample image")
    superres_image = tools.ImageSuperResolution(downsamp_image, modelGu)
    superres_image.show()
    print("show the super resolution image")
    reconstr_image = tools.ImageSuperResolution(superres_image, modelGd)
    reconstr_image.show()
    print("show the reconstructed image")
    original_image.show()
    print("show the original image")