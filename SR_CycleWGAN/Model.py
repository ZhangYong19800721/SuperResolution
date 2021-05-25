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
    def __init__(self, inChannel=3, interChannel=64):
        super(PatchFeatureExtractor16, self).__init__()
        self.inChannel = inChannel
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(self.inChannel, self.interChannel, 9, padding=9 // 2, bias=True)
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
    def __init__(self, inChannel=3, interChannel=64):
        super(PatchFeatureExtractor8, self).__init__()
        self.inChannel = inChannel
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(self.inChannel, self.interChannel, 9, padding=9 // 2, bias=True)
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
    def __init__(self, inChannel=3, interChannel=64):
        super(PatchFeatureExtractor4, self).__init__()
        self.inChannel = inChannel
        self.interChannel = interChannel
        self.L01_Conv2d = nn.Conv2d(self.inChannel, self.interChannel, 9, padding=9 // 2, bias=True)
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


class Generator(nn.Module):
    def __init__(self, nz=100):
        super(Generator, self).__init__()
        self.L01_Sequential = nn.Sequential(nn.ConvTranspose2d(nz, 512, kernel_size=4, stride=1, padding=0),
                                            nn.BatchNorm2d(512),
                                            nn.PReLU(512),
                                            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.PReLU(128),
                                            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
                                            nn.LeakyReLU(0.2),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L01_Sequential(x)
        return y


class GeneratorUx1(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=3):
        super(GeneratorUx1, self).__init__()
        self.L00_Transform = nn.Conv2d(in_channels=inChannel, out_channels=inChannel, kernel_size=1, padding=0, bias=True)
        self.patchFeatureExtractor = PatchFeatureExtractor16(inChannel, interChannel)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(interChannel, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, outChannel, 9, padding=9 // 2),
                                            nn.BatchNorm2d(outChannel),
                                            # nn.LeakyReLU(0.02),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L00_Transform(x)
        z = self.patchFeatureExtractor(y)
        z = self.L01_Sequential(z)
        y = y + z
        y = torch.nn.functional.leaky_relu(y, 0.02)
        y = 1 - torch.nn.functional.leaky_relu(1 - y, 0.02)
        return y


class GeneratorUx2(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=3):
        super(GeneratorUx2, self).__init__()
        self.L00_Transform = nn.ConvTranspose2d(in_channels=inChannel, out_channels=outChannel, kernel_size=2, stride=2, padding=0)
        self.patchFeatureExtractor = PatchFeatureExtractor16(inChannel, interChannel)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(interChannel, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, outChannel, 9, padding=9 // 2),
                                            nn.BatchNorm2d(outChannel),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L00_Transform(x)
        z = self.patchFeatureExtractor(x)
        z = self.L01_Sequential(z)
        y = y + z
        y = torch.nn.functional.leaky_relu(y, 0.02)
        y = 1 - torch.nn.functional.leaky_relu(1 - y, 0.02)
        return y


class GeneratorDx1(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=3):
        super(GeneratorDx1, self).__init__()
        self.L00_Transform = nn.Conv2d(in_channels=inChannel, out_channels=inChannel, kernel_size=1, padding=0, bias=True)
        self.patchFeatureExtractor = PatchFeatureExtractor16(inChannel, interChannel)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(interChannel, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, outChannel, 3, padding=3 // 2),
                                            nn.BatchNorm2d(outChannel),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L00_Transform(x)
        z = self.patchFeatureExtractor(y)
        z = self.L01_Sequential(z)
        y = y + z
        y = torch.nn.functional.leaky_relu(y, 0.02)
        y = 1 - torch.nn.functional.leaky_relu(1 - y, 0.02)
        return y


class GeneratorDx2(nn.Module):
    def __init__(self, inChannel=3, interChannel=64, outChannel=3):
        super(GeneratorDx2, self).__init__()
        self.L00_Transform = nn.Conv2d(in_channels=inChannel, out_channels=outChannel, kernel_size=2, stride=2, padding=0)
        self.patchFeatureExtractor = PatchFeatureExtractor16(inChannel, interChannel)
        self.L01_Sequential = nn.Sequential(nn.Conv2d(interChannel, 256, 3, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, 256, 3, stride=2, padding=3 // 2),
                                            nn.BatchNorm2d(256),
                                            nn.PReLU(256),
                                            nn.Conv2d(256, outChannel, 3, padding=3 // 2),
                                            nn.BatchNorm2d(outChannel),
                                            )

    # the x is low resolution images minibatch
    def forward(self, x):
        y = self.L00_Transform(x)
        z = self.patchFeatureExtractor(x)
        z = self.L01_Sequential(z)
        y = y + z
        y = torch.nn.functional.leaky_relu(y, 0.02)
        y = 1 - torch.nn.functional.leaky_relu(1 - y, 0.02)
        return y


class DiscriminatorSPx1(nn.Module):
    def __init__(self, inChannel=1):
        super(DiscriminatorSPx1, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
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
    def __init__(self, inChannel=1):
        super(Discriminator_GP, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y


class Discriminator(nn.Module):
    def __init__(self, inChannel=1):
        super(Discriminator, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.Full_Sequential = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, imageHR):
        y = self.FeatureExtractor(imageHR)
        y = self.Full_Sequential(y)
        return y


class DiscriminatorSPx2(nn.Module):
    def __init__(self, inChannel=3):
        super(DiscriminatorSPx2, self).__init__()

        self.FeatureExtractor = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=inChannel, out_channels=32, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=3 // 2, bias=True)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=3 // 2, bias=True)),
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


if __name__ == '__main__':
    Gu = GeneratorUx2(3, 64, 3)
    image = torch.rand((2, 3, 64, 64))
    with torch.no_grad():
        z = Gu(image)
    print(z.shape)


    transCov2d = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0)
    y = transCov2d(image)
    print(y.shape)

    big_image = torch.rand((2, 3, 128, 128))
    Cov2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0)
    q = Cov2d(big_image)
    print(q.shape)

    D = DiscriminatorSPx2(inChannel=3)
    output_D = D(big_image)
    print(output_D.shape)
