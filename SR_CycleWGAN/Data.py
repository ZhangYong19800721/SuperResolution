import matplotlib.pyplot as plt
import random
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf
import tools


# Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True, num_workers=workers)

class DataLoader(object):
    def __init__(self, dataset, minibatch_size, row=2, col=3, select_row=2, select_col=3, shuffle=False):
        super(DataLoader, self).__init__()
        self.dataset = dataset
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle
        self.shuffle_layer = list(range(len(self.dataset)))
        self.row, self.col = row, col
        self.select_row, self.select_col = select_row, select_col
        if shuffle:
            random.shuffle(self.shuffle_layer)

    def __len__(self):
        return len(self.dataset) // self.minibatch_size

    def __getitem__(self, item):
        item = item % len(self)
        selected = range(item * self.minibatch_size, (item + 1) * self.minibatch_size)
        selected = [self.shuffle_layer[i] for i in selected]
        minibatch_images_HR = [self.dataset[i][0] for i in selected]
        image_W, image_H = minibatch_images_HR[0].size
        minibatch_images_LR = [ttf.resize(IHR, (image_H // 2, image_W // 2)) for IHR in minibatch_images_HR]

        # change the data to tensor
        minibatch_images_HR = [ttf.to_tensor(x) for x in minibatch_images_HR]
        minibatch_images_LR = [ttf.to_tensor(x) for x in minibatch_images_LR]

        # stack the data
        minibatch_images_HR = torch.stack(minibatch_images_HR)
        minibatch_images_LR = torch.stack(minibatch_images_LR)

        # split to patch
        patchs_HR = tools.SplitHV(minibatch_images_HR, self.row, self.col)
        patchs_LR = tools.SplitHV(minibatch_images_LR, self.row, self.col)

        if self.shuffle:
            selected_row = random.sample(range(0, self.row), self.select_row)
            selected_col = random.sample(range(0, self.col), self.select_col)
        else:
            selected_row = list(range(self.select_row))
            selected_col = list(range(self.select_col))

        patchs_HR = [patchs_HR[i][j] for i in selected_row for j in selected_col]
        patchs_LR = [patchs_LR[i][j] for i in selected_row for j in selected_col]

        patchs_HR = torch.cat(patchs_HR, dim=0)
        patchs_LR = torch.cat(patchs_LR, dim=0)

        return patchs_LR, patchs_HR

if __name__ == "__main__":
    ## set the data set
    dataroot = "/home/zhangyong/Data/image2160x3840"
    image_H, image_W = 128*8, 128*14
    minibatch_size = 1
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([transforms.Resize((image_H, image_W))]))
    dataLoader = DataLoader(dataset, minibatch_size=minibatch_size, shuffle=True)
    minibatch = dataLoader[0]
    print("OK!")