import torch
import torch.nn as nn

__all__ = [
    "VGG",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
]

#  -----------------------------------------------
#              input : (3, 224, 224)
#  -----------------------------------------------
#       VGG11 |     VGG13 |     VGG16 |     VGG19
#  -----------------------------------------------
#    conv3-64 |  conv3-64 |  conv3-64 |  conv3-64
#             |  conv3-64 |  conv3-64 |  conv3-64
#  -----------------------------------------------
#                      maxpool
#  -----------------------------------------------
#   conv3-128 | conv3-128 | conv3-128 | conv3-128
#             | conv3-128 | conv3-128 | conv3-128
#  -----------------------------------------------
#                      maxpool
#  -----------------------------------------------
#   conv3-256 | conv3-256 | conv3-256 | conv3-256
#   conv3-256 | conv3-256 | conv3-256 | conv3-256
#             |           | conv3-256 | conv3-256
#             |           |           | conv3-256
#  -----------------------------------------------
#                      maxpool
#  -----------------------------------------------
#   conv3-512 | conv3-512 | conv3-512 | conv3-512
#   conv3-512 | conv3-512 | conv3-512 | conv3-512
#             |           | conv3-512 | conv3-512
#             |           |           | conv3-512
#  -----------------------------------------------
#                      maxpool
#  -----------------------------------------------
#   conv3-512 | conv3-512 | conv3-512 | conv3-512
#   conv3-512 | conv3-512 | conv3-512 | conv3-512
#             |           | conv3-512 | conv3-512
#             |           |           | conv3-512
#  -----------------------------------------------
#                      avgpool
#  -----------------------------------------------
#                      fc-4096
#  -----------------------------------------------
#                      fc-4096
#  -----------------------------------------------
#                       fc-10
#  -----------------------------------------------
#                      softmax
#  -----------------------------------------------


class Conv2dModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=stride + 2,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((7, 7)), nn.Flatten())
        self.dense = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.fc_in = 4096
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.dense(x)
        x = self.fc(x)
        return x


class VGG11(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        features = nn.Sequential(
            Conv2dModule(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(128, 256),
            Conv2dModule(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(256, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        super().__init__(features, num_classes)


class VGG13(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        features = nn.Sequential(
            Conv2dModule(in_channels, 64),
            Conv2dModule(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(64, 128),
            Conv2dModule(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(128, 256),
            Conv2dModule(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(256, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        super().__init__(features, num_classes)


class VGG16(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        features = nn.Sequential(
            Conv2dModule(in_channels, 64),
            Conv2dModule(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(64, 128),
            Conv2dModule(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(128, 256),
            Conv2dModule(256, 256),
            Conv2dModule(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(256, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        super().__init__(features, num_classes)


class VGG19(VGG):
    def __init__(self, in_channels=3, num_classes=10):
        features = nn.Sequential(
            Conv2dModule(in_channels, 64),
            Conv2dModule(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(64, 128),
            Conv2dModule(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(128, 256),
            Conv2dModule(256, 256),
            Conv2dModule(256, 256),
            Conv2dModule(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(256, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            Conv2dModule(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        super().__init__(features, num_classes)
