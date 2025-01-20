# import torch
# import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self, in_channels=3, features=64):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             # Input Layer
#             nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             self._block(features, features * 2, 4, 2, 1),
#             self._block(features * 2, features * 4, 4, 2, 1),
#             self._block(features * 4, features * 8, 4, 2, 1),
#             # Output Layer
#             nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )

#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def forward(self, x):
#         return self.model(x)

# def test():
#     model = Discriminator(in_channels=3)
#     x = torch.randn(2, 3, 256, 256)  # Batch of 2 images, 256x256 size
#     print(model(x).shape)  # Expecting output of size (2, 1, 16, 16)

# if __name__ == "__main__":
#     test()


"""
Discriminator model for CycleGAN

"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()