import torch
import torch.nn as nn

conv2d_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
conv2d_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
x = torch.randn((1, 3, 200, 100))

y = conv2d_1(x)
y = conv2d_2(y)

print(y.shape)


leakyReLU = torch.nn.LeakyReLU(0.02)
y = leakyReLU(-torch.ones((2,1)))
print(y)
