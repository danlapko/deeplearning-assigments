import torch.nn as nn
import torch


class DCGenerator(nn.Module):

    def __init__(self, image_size):
        super(DCGenerator, self).__init__()

        # in 100x1x1

        # out 512x4x4
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True))

        # out 256x8x8
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        # out 128x16x16
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True))

        # out 3x32x32
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                                   nn.Tanh())

    # --> in [batch_size, n_channels=embedding, width, height] == [256, 100, 1, 1]
    # --> out [batch_size, n_channels, width, height] == [256, 3, 32, 32]
    def forward(self, data):
        # print("g in:", data.shape)

        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # print("g out:", x.shape)

        return x


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv2d(3, 1, 3),
        #                            nn.Sigmoid(),
        #                            nn.AdaptiveMaxPool2d(1))

        # in 3x32x132

        # out 64x16x16
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True))

        # out 128x8x8
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(0.2, inplace=True))

        # out 256x4x4
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(0.2, inplace=True))

        # out 1x1x1
        self.conv4 = nn.Sequential(nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                                   nn.Sigmoid())

    # --> in [batch_size, n_channels, width, height] == [256, 3, 32, 32]
    # --> out [batch_size] == [256]
    def forward(self, data):
        # print("d in:", data.shape)

        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = torch.squeeze(x)
        # print("d out:", x.shape)

        return x
