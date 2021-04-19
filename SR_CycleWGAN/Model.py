import torch
import torch.nn as nn
import tools


class ResidualBlock(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlock, self).__init__()
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L02_BatchNorm = nn.BatchNorm2d(self.interChannel)
        self.L03_PReLU = nn.PReLU(self.interChannel)
        self.L04_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L05_BatchNorm = nn.BatchNorm2d(self.interChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_BatchNorm(y)
        y = self.L03_PReLU(y)
        y = self.L04_Conv2d(y)
        y = self.L05_BatchNorm(y)
        y = y + x
        return y


class ResidualBlocks4(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks4, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlock = ResidualBlock(self.interChannel)
        self.L02_ResidualBlock = ResidualBlock(self.interChannel)
        self.L03_ResidualBlock = ResidualBlock(self.interChannel)
        self.L04_ResidualBlock = ResidualBlock(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlock(x)
        y = self.L02_ResidualBlock(y)
        y = self.L03_ResidualBlock(y)
        y = self.L04_ResidualBlock(y)
        return y


class ResidualBlocks8(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks8, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlocks4 = ResidualBlocks4(self.interChannel)
        self.L02_ResidualBlocks4 = ResidualBlocks4(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlocks4(x)
        y = self.L02_ResidualBlocks4(y)
        return y


class ResidualBlocks16(nn.Module):
    def __init__(self, interChannel):
        super(ResidualBlocks16, self).__init__()
        self.interChannel = interChannel
        self.L01_ResidualBlocks8 = ResidualBlocks8(self.interChannel)
        self.L02_ResidualBlocks8 = ResidualBlocks8(self.interChannel)

    def forward(self, x):
        y = self.L01_ResidualBlocks8(x)
        y = self.L02_ResidualBlocks8(y)
        return y


class PatchFeatureExtractor16(nn.Module):
    def __init__(self, interChannel = 64):
        super(PatchFeatureExtractor16, self).__init__()
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(3, self.interChannel, 9, padding=9 // 2, bias=True)
        self.L02_PReLU = nn.PReLU(self.interChannel)
        self.L03_ResidualBlocks = ResidualBlocks16(self.interChannel)
        self.L04_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L05_BatchNorm = nn.BatchNorm2d(self.interChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_PReLU(y)
        z = self.L03_ResidualBlocks(y)
        z = self.L04_Conv2d(z)
        z = self.L05_BatchNorm(z)
        z = z + y
        return z

class PatchFeatureExtractor8(nn.Module):
    def __init__(self, interChannel = 64):
        super(PatchFeatureExtractor8, self).__init__()
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(3, self.interChannel, 9, padding=9 // 2, bias=True)
        self.L02_PReLU = nn.PReLU(self.interChannel)
        self.L03_ResidualBlocks = ResidualBlocks8(self.interChannel)
        self.L04_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L05_BatchNorm = nn.BatchNorm2d(self.interChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_PReLU(y)
        z = self.L03_ResidualBlocks(y)
        z = self.L04_Conv2d(z)
        z = self.L05_BatchNorm(z)
        z = z + y
        return z

class PatchFeatureExtractor4(nn.Module):
    def __init__(self, interChannel = 64):
        super(PatchFeatureExtractor4, self).__init__()
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(3, self.interChannel, 9, padding=9 // 2, bias=True)
        self.L02_PReLU = nn.PReLU(self.interChannel)
        self.L03_ResidualBlocks = ResidualBlocks4(self.interChannel)
        self.L04_Conv2d = nn.Conv2d(self.interChannel, self.interChannel, 3, padding=1, bias=True)
        self.L05_BatchNorm = nn.BatchNorm2d(self.interChannel)

    def forward(self, x):
        y = self.L01_Conv2d(x)
        y = self.L02_PReLU(y)
        z = self.L03_ResidualBlocks(y)
        z = self.L04_Conv2d(z)
        z = self.L05_BatchNorm(z)
        z = z + y
        return z


class GeneratorD(nn.Module):
    def __init__(self):
        super(GeneratorD, self).__init__()
        self.patchFeatureExtractor = PatchFeatureExtractor16(interChannel=64)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(64, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 256, 3, stride=2, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 3, 3, padding=3//2),
                                            nn.BatchNorm2d(3),
                                            nn.PReLU(3),
                                            )

    # the x is low resolution images minibatch
    def forward(self, image):
        v = self.patchFeatureExtractor(image)
        ImageLR = self.L01_Sequential(v)
        return ImageLR

class GeneratorU(nn.Module):
    def __init__(self):
        super(GeneratorU, self).__init__()
        self.patchFeatureExtractor = PatchFeatureExtractor16(interChannel=64)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(64, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 3, 9, padding=9 // 2),
                                            nn.BatchNorm2d(3),
                                            nn.PReLU(3),
                                            )

    # the x is low resolution images minibatch
    def forward(self, image):
        v = self.patchFeatureExtractor(image)
        ImageSR = self.L01_Sequential(v)
        return ImageSR

class GeneratorNoise(nn.Module):
    def __init__(self, nz):
        super(GeneratorNoise, self).__init__()
        self.nz = nz
        self.patchFeatureExtractor = PatchFeatureExtractor16(interChannel=64)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(64 + self.nz, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.PReLU(128),
                                            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.PReLU(64),
                                            nn.Conv2d(64, 3, 9, padding=9 // 2),
                                            )

    # the x is low resolution images minibatch
    def forward(self, image, noise):
        v = self.patchFeatureExtractor(image)
        v = torch.cat((v, noise), dim=1)
        ImageSR = self.L01_Sequential(v)
        return ImageSR


class CV2D_SP_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(CV2D_SP_LeakyReLU, self).__init__()
        self.L01_Sequential = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=True)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        y = self.L01_Sequential(x)
        return y


class Discriminator_SP(nn.Module):
    def __init__(self):
        super(Discriminator_SP, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(32)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(32)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(64)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(64)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(128)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(128)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(2048)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            # nn.utils.spectral_norm(nn.BatchNorm2d(2048)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2048, 1024)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(1024, 512)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(512, 1)),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y

class Discriminator_GP(nn.Module):
    def __init__(self):
        super(Discriminator_GP, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y

if __name__ == '__main__':
    Gu = GeneratorU()
    image = torch.rand((2, 3, 64, 64))
    with torch.no_grad():
        SR = Gu(image)

    Gd = GeneratorD()
    with torch.no_grad():
        LR = Gd(SR)

    D_SP = Discriminator_SP()
    image = torch.rand((2, 3, 128, 128))
    with torch.no_grad():
        z = D_SP(image)
    print(z.shape)

    D = Discriminator()
    with torch.no_grad():
        z = D(image)
    print(z.shape)
