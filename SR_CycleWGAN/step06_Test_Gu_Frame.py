import argparse
import pickle
import torch
import torch.nn as nn
import tools
import torchvision.transforms as transforms
import torchvision.datasets as dset
import random

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # get the GPU device
    ##########################################################################
    ## load the AI output
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--ModelGuFile", type=str, help="None or the path for Gu output")
    parser.add_argument("--MaxSampleID", type=int, help="the Max Minibatch ID, use this to cut the trainset")
    args = parser.parse_args()

    maxSampleID = args.MaxSampleID if args.MaxSampleID else 1e100

    modelGu_file = open(args.ModelGuFile, "rb")  # open the output file
    modelGu = pickle.load(modelGu_file)  # load the output file
    if isinstance(modelGu, nn.DataParallel):
        modelGu = modelGu.module
    modelGu.to('cpu')  # push output to GPU device
    modelGu.eval()  # set the output to evaluation mode, (the dropout layer need this)
    modelGu_file.close()  # close the output file

    image_H, image_W = 1080, 1920
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    maxSampleID = min(maxSampleID, len(dataset)) - 1
    n = random.randint(0, maxSampleID)
    original_image = dataset[n][0]
    downsamp_image = tools.imResize(original_image, (1080 // 2, 1920 // 2))
    downsamp_image.show()
    print("show the down sample image")
    superres_image = tools.ImageSuperResolution(downsamp_image, modelGu)
    superres_image.show()
    print("show the super resolution image")
    original_image.show()
    print("show the original image")