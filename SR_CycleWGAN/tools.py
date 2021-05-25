import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import signal
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.transforms.functional as ttf


def Split04(x):
    U, D = torch.split(x, x.shape[2] // 2, dim=2)
    UL, UR = torch.split(U, U.shape[3] // 2, dim=3)
    DL, DR = torch.split(D, D.shape[3] // 2, dim=3)
    y = torch.cat((UL, UR, DL, DR), dim=1)
    return y


def Split16(x):
    y = Split04(x)
    y = Split04(y)
    return y


def SplitH(x, n):
    return list(torch.split(x, x.shape[2] // n, dim=2))


def SplitV(x, n):
    return list(torch.split(x, x.shape[3] // n, dim=3))


def SplitHV(x, m, n):
    patchMatrix = SplitH(x, m)
    patchMatrix = [SplitV(x, n) for x in patchMatrix]
    return patchMatrix


def SplitToPatch(image, patchSize=(27, 48)):
    if not isinstance(image, Image.Image):
        raise ("image must be an instance of PIL.Image.Image")
    image = ttf.to_tensor(image)
    R = list(torch.split(image, patchSize[0], dim=1))
    M = [list(torch.split(r, patchSize[1], dim=2)) for r in R]
    return M


# resize the image
def imResize(image, new_size=(1080, 1920)):
    newImage = ttf.resize(image, new_size, interpolation=Image.BICUBIC)
    return newImage


# montage the image1 and image2 according to mask
#
def imMontage(image1, image2, mask=None, line=True):
    if isinstance(image1, Image.Image):
        image1 = ttf.to_tensor(image1)
    elif isinstance(image1, torch.Tensor):
        pass
    else:
        raise ("image1 must be a PIL.Image or a torch.Tensor")

    if isinstance(image2, Image.Image):
        image2 = ttf.to_tensor(image2)
    elif isinstance(image2, torch.Tensor):
        pass
    else:
        raise ("image2 must be a PIL.Image or a torch.Tensor")

    if mask != None:
        if isinstance(mask, Image.Image):
            mask = ttf.to_tensor(mask)
        elif isinstance(mask, torch.Tensor):
            pass
        else:
            raise ("mask must be a PIL.Image or a torch.Tensor")

        if image1.shape != image2.shape or image1.shape != mask.shape:
            raise ("image1 and image2 and mask must have the same shape")
    else:
        mask = torch.ones_like(image1)
        mask[:, :, 0:mask.size(2) // 2] = 0

    reverse_mask = torch.ones_like(mask)
    reverse_mask = reverse_mask - mask
    outImage = image1 * mask + image2 * reverse_mask

    if line:
        maskma = mask[0].numpy()  # convert tensor to numpy ndarray
        kernal = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        red = signal.convolve2d(maskma, kernal, 'same')
        red = torch.tensor((red > 0) + 0.0, dtype=torch.float32)  # the red inline mask
        red = torch.stack((red, red, red))
        reverse_red = torch.ones_like(red)
        reverse_red = reverse_red - red
        back_ground = torch.ones_like(outImage)
        outImage = outImage * reverse_red + back_ground * red

    outImage = ttf.to_pil_image(outImage)
    return outImage


def ImageSuperResolution(image, G):
    image = ttf.to_tensor(image)
    image = torch.stack((image,))
    with torch.no_grad():
        image_SR = G(image)

    outputImage = ttf.to_pil_image(image_SR[0])
    return outputImage


def showNineGrid_1x9(I1, I2, I3, I4, I5, I6, I7, I8, I9):
    image_H, image_W = I1.shape[1:]
    I1 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I1), (image_H, image_W), interpolation=Image.BICUBIC))
    I2 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I2), (image_H, image_W), interpolation=Image.BICUBIC))
    I3 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I3), (image_H, image_W), interpolation=Image.BICUBIC))
    I4 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I4), (image_H, image_W), interpolation=Image.BICUBIC))
    I5 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I5), (image_H, image_W), interpolation=Image.BICUBIC))
    I6 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I6), (image_H, image_W), interpolation=Image.BICUBIC))
    I7 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I7), (image_H, image_W), interpolation=Image.BICUBIC))
    I8 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I8), (image_H, image_W), interpolation=Image.BICUBIC))
    I9 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I9), (image_H, image_W), interpolation=Image.BICUBIC))

    I = torch.cat((I1, I2, I3, I4, I5, I6, I7, I8, I9), dim=2)
    I = ttf.to_pil_image(I)
    I.show()


def showNineGrid_2x2(I1, I2, I3, I4):
    image_H, image_W = I1.shape[1:]
    I1 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I1), (image_H, image_W), interpolation=Image.BICUBIC))
    I2 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I2), (image_H, image_W), interpolation=Image.BICUBIC))
    I3 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I3), (image_H, image_W), interpolation=Image.BICUBIC))
    I4 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I4), (image_H, image_W), interpolation=Image.BICUBIC))

    R1 = torch.cat((I1, I2), dim=2)
    R2 = torch.cat((I3, I4), dim=2)
    I = torch.cat((R1, R2), dim=1)
    I = ttf.to_pil_image(I)
    I.show()


def showNineGrid_1x2(I1, I2):
    image_H, image_W = I1.shape[1:]
    I1 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I1), (image_H, image_W), interpolation=Image.BICUBIC))
    I2 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I2), (image_H, image_W), interpolation=Image.BICUBIC))
    I = torch.cat((I1, I2), dim=2)
    I = ttf.to_pil_image(I)
    I.show()


def showNineGrid_3x3(I1, I2, I3, I4, I5, I6, I7, I8, I9):
    image_H, image_W = I1.shape[1:]
    I1 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I1), (image_H, image_W), interpolation=Image.BICUBIC))
    I2 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I2), (image_H, image_W), interpolation=Image.BICUBIC))
    I3 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I3), (image_H, image_W), interpolation=Image.BICUBIC))
    I4 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I4), (image_H, image_W), interpolation=Image.BICUBIC))
    I5 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I5), (image_H, image_W), interpolation=Image.BICUBIC))
    I6 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I6), (image_H, image_W), interpolation=Image.BICUBIC))
    I7 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I7), (image_H, image_W), interpolation=Image.BICUBIC))
    I8 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I8), (image_H, image_W), interpolation=Image.BICUBIC))
    I9 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I9), (image_H, image_W), interpolation=Image.BICUBIC))

    R1 = torch.cat((I1, I2, I3), dim=2)
    R2 = torch.cat((I4, I5, I6), dim=2)
    R3 = torch.cat((I7, I8, I9), dim=2)
    I = torch.cat((R1, R2, R3), dim=1)
    I = ttf.to_pil_image(I)
    I.show()


def showNineGrid_4x7(I11, I12, I13, I14, I15, I16, I17,
                     I21, I22, I23, I24, I25, I26, I27,
                     I31, I32, I33, I34, I35, I36, I37,
                     I41, I42, I43, I44, I45, I46, I47):
    image_H, image_W = I11.shape[1:]
    I11 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I11), (image_H, image_W), interpolation=Image.BICUBIC))
    I12 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I12), (image_H, image_W), interpolation=Image.BICUBIC))
    I13 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I13), (image_H, image_W), interpolation=Image.BICUBIC))
    I14 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I14), (image_H, image_W), interpolation=Image.BICUBIC))
    I15 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I15), (image_H, image_W), interpolation=Image.BICUBIC))
    I16 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I16), (image_H, image_W), interpolation=Image.BICUBIC))
    I17 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I17), (image_H, image_W), interpolation=Image.BICUBIC))

    I21 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I21), (image_H, image_W), interpolation=Image.BICUBIC))
    I22 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I22), (image_H, image_W), interpolation=Image.BICUBIC))
    I23 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I23), (image_H, image_W), interpolation=Image.BICUBIC))
    I24 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I24), (image_H, image_W), interpolation=Image.BICUBIC))
    I25 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I25), (image_H, image_W), interpolation=Image.BICUBIC))
    I26 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I26), (image_H, image_W), interpolation=Image.BICUBIC))
    I27 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I27), (image_H, image_W), interpolation=Image.BICUBIC))

    I31 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I31), (image_H, image_W), interpolation=Image.BICUBIC))
    I32 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I32), (image_H, image_W), interpolation=Image.BICUBIC))
    I33 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I33), (image_H, image_W), interpolation=Image.BICUBIC))
    I34 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I34), (image_H, image_W), interpolation=Image.BICUBIC))
    I35 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I35), (image_H, image_W), interpolation=Image.BICUBIC))
    I36 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I36), (image_H, image_W), interpolation=Image.BICUBIC))
    I37 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I37), (image_H, image_W), interpolation=Image.BICUBIC))

    I41 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I41), (image_H, image_W), interpolation=Image.BICUBIC))
    I42 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I42), (image_H, image_W), interpolation=Image.BICUBIC))
    I43 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I43), (image_H, image_W), interpolation=Image.BICUBIC))
    I44 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I44), (image_H, image_W), interpolation=Image.BICUBIC))
    I45 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I45), (image_H, image_W), interpolation=Image.BICUBIC))
    I46 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I46), (image_H, image_W), interpolation=Image.BICUBIC))
    I47 = ttf.to_tensor(ttf.resize(ttf.to_pil_image(I47), (image_H, image_W), interpolation=Image.BICUBIC))

    R1 = torch.cat((I11, I12, I13, I14, I15, I16, I17), dim=2)
    R2 = torch.cat((I21, I22, I23, I24, I25, I26, I27), dim=2)
    R3 = torch.cat((I31, I32, I33, I34, I35, I36, I37), dim=2)
    R4 = torch.cat((I41, I42, I43, I44, I45, I46, I47), dim=2)
    I = torch.cat((R1, R2, R3, R4), dim=1)
    I = ttf.to_pil_image(I)
    I.show()


class EXPMA(object):
    def __init__(self, alfa=0.01, init=None):
        super(EXPMA, self).__init__()
        self.alpha = alfa
        self.value = init

    def expma(self, new_value):
        self.value = new_value if self.value == None else ((1 - self.alpha) * self.value + self.alpha * new_value)
        return self.value


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# calculate the gradient penalty
def cal_gradient_penalty(D, device, real_samples, fake_samples):
    # weight alpha
    N = real_samples.size(0)
    C = real_samples.size(1)
    H = real_samples.size(2)
    W = real_samples.size(3)

    alpha = torch.rand(N, 1)
    alpha = alpha.expand((-1, C * H * W)).reshape((N, C, H, W))
    alpha = alpha.to(device)

    # calculate interpolates
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)

    # want to get the gradient, calculate the Discriminator's output firstly
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = D(interpolates)

    # 计算梯度
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # 利用梯度计算出gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == '__main__':
    image = torch.rand((3, 270, 480))
    image = ttf.to_pil_image(image)
    image.show()
    super_image = ImageSuperResolution(image)
    super_image.show()
