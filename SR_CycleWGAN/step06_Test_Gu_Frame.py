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
    ## load the AI model
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, help="The root dir for dataset")
    parser.add_argument("--ModelGuFile", type=str, help="None or the path for Gu model")
    args = parser.parse_args()

    modelGu_file = open(args.ModelGuFile, "rb")  # open the model file
    modelGu = pickle.load(modelGu_file)  # load the model file
    if isinstance(modelGu, nn.DataParallel):
        modelGu = modelGu.module
    modelGu.to('cpu')  # push model to GPU device
    modelGu.eval()  # set the model to evaluation mode, (the dropout layer need this)
    modelGu_file.close()  # close the model file

    image_H, image_W = 1080, 1920
    dataroot = args.dataroot
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    n = random.randint(0, len(dataset)-1)
    original_image = dataset[n][0]
    downsamp_image = tools.imResize(original_image, (1080 // 2, 1920 // 2))
    downsamp_image.show()
    print("show the down sample image")
    superres_image = tools.ImageSuperResolution(downsamp_image, modelGu)
    superres_image.show()
    print("show the super resolution image")
    original_image.show()
    print("show the original image")