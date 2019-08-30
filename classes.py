import torch
from torch import nn
from torch.nn import functional as F, init
from torch.utils.data import Dataset  # , DataLoader
from torchvision import models, transforms as T

from PIL import Image
from os import listdir
from pandas import DataFrame


class Extractor(nn.Module):

    def __init__(self, content, style, content_wt=1.0, style_wt=0.2, device=None):

        super().__init__()

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.base = models.vgg16(pretrained=True).features.to(self.device)
        self.base = models.vgg19(pretrained=True).features.to(self.device)
        self.base.eval()

        self.ST_WEIGHT = content_wt
        self.CT_WEIGHT = style_wt

        modules = list(self.base.children())

        # self.layer1 = nn.Sequential(*modules[:4])
        # self.layer2 = nn.Sequential(*modules[4:9])
        # self.layer3 = nn.Sequential(*modules[9:18])
        # self.layer3_5 = nn.Sequential(*modules[18:23])  # Content layer
        # self.layer4 = nn.Sequential(*modules[23:27])
        # self.layer5 = nn.Sequential(*modules[27:36])
        # self.end_layer = nn.Sequential(*modules[36:])

        # -----------------------------------------------------------------------------------------
        # New layers: 4, 9, 16, 23c(or 25c), 27, 34c
        # --> From the paper: Perceptual Losses for Real-Time Style Transfer and Super-Resolution.
        #                     https://arxiv.org/abs/1603.08155

        self.layer1 = nn.Sequential(*modules[:4])
        self.layer2 = nn.Sequential(*modules[4:9])
        self.layer3 = nn.Sequential(*modules[9:16])
        self.layer4_c = nn.Sequential(*modules[16:23])  # Content layer
        self.layer5 = nn.Sequential(*modules[23:27])
        self.layer6_c = nn.Sequential(*modules[27:34])    # Content layer
        self.end_layer = nn.Sequential(*modules[34:])

        y = self.layer1(style)
        self.st_loss_1 = StyleLoss(y.detach())

        y = self.layer2(y)
        self.st_loss_2 = StyleLoss(y.detach())

        y = self.layer3(y)
        self.st_loss_3 = StyleLoss(y.detach())

        y = self.layer5(self.layer4_c(y.detach()))
        self.st_loss_4 = StyleLoss(y.detach())

        # y = self.layer5(y)
        # self.st_loss_5 = StyleLoss(y.detach())

        y = self.layer4_c(self.layer3(self.layer2(self.layer1(content))))
        self.ct_loss_1 = ContentLoss(y.detach())

        y = self.layer6_c(self.layer5(y))
        self.ct_loss_2 = ContentLoss(y.detach())

        y = None

        for param in self.parameters():
            param.requires_grad = False

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):

        y = self.layer1(x)
        style_loss = self.st_loss_1(y)

        y = self.layer2(y)
        style_loss += self.st_loss_2(y)

        y = self.layer3(y)
        style_loss += self.st_loss_3(y)

        y = self.layer4_c(y)
        content_loss = self.ct_loss_1(y)

        y = self.layer5(y)
        style_loss += self.st_loss_4(y)

        y = self.layer6_c(y)
        content_loss += self.ct_loss_2(y)

        # y = self.end_layer(y)

        return style_loss * self.ST_WEIGHT + content_loss * self.CT_WEIGHT
        # return style_loss * self.ST_WEIGHT, content_loss * self.CT_WEIGHT, y


class ContentLoss(nn.Module):
    def __init__(self, content):
        super(ContentLoss, self).__init__()

        self.target = content

        self.crit = nn.MSELoss()

        pass

    def forward(self, x):
        return self.crit(x, self.target)


class StyleLoss(nn.Module):
    def __init__(self, style):
        super(StyleLoss, self).__init__()

        self.target = self.gram_matrix(style)
        # self.target = style

        self.crit = nn.MSELoss()

        pass

    def forward(self, x):
        return self.crit(self.gram_matrix(x), self.target) / (x.size(1) ** 2)
        # return self.crit(x, self.target)

    def gram_matrix(self, tensor):
        """
        Code found here:
        https://gist.github.com/mwitiderrick/cd0983f7d5f93354790580969928ee66#file-gram_matrix-ph

        :param tensor:  Image tensor
        :return:        Gram matrix for the input tensor.
        """

        a, b, c, d = tensor.size()
        features = tensor.view(a * b, c * d)  # resise F_XL into \hat F_XL
        gram = torch.mm(features, features.t())

        return gram.div(a * b * c * d)


class StyleTransfer(nn.Module):
    """
    Model inspired in part by this repo:
    https://github.com/KushajveerSingh/Photorealistic-Style-Transfer/blob/master/src/style_transfer.py

    Model weights initialized using Normal Distribution with help from this answer:
    https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    """

    def __init__(self):
        super().__init__()

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=8,
                                                  kernel_size=3,
                                                  padding=1,
                                                  stride=1),
                                        nn.InstanceNorm2d(8),
                                        nn.ReLU()
                                        )
        self.init_layer.apply(fn=init_weights)

        self.down_block_1 = nn.Sequential(nn.Conv2d(in_channels=8,
                                                    out_channels=8,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=2),
                                          nn.InstanceNorm2d(3),
                                          nn.ReLU(),
                                          BottleNeck(in_channels=8,
                                                     out_channels=16,
                                                     kernel_size=3,
                                                     stride=1)
                                          )
        self.down_block_1.apply(fn=init_weights)

        self.down_block_2 = nn.Sequential(nn.Conv2d(in_channels=16,
                                                    out_channels=16,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=2),
                                          nn.InstanceNorm2d(3),
                                          nn.ReLU(),
                                          BottleNeck(in_channels=16,
                                                     out_channels=64,
                                                     kernel_size=3,
                                                     stride=1)
                                          )
        self.down_block_2.apply(fn=init_weights)

        self.down_block_3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                    out_channels=64,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=2),
                                          nn.InstanceNorm2d(3),
                                          nn.ReLU(),
                                          BottleNeck(in_channels=64,
                                                     out_channels=64,
                                                     kernel_size=3,
                                                     stride=1)
                                          )
        self.down_block_3.apply(fn=init_weights)

        self.res_blocks = nn.Sequential(BottleNeck(in_channels=64,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   stride=1),
                                        BottleNeck(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   stride=1),
                                        BottleNeck(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   stride=1),
                                        BottleNeck(in_channels=128,
                                                   out_channels=64,
                                                   kernel_size=3,
                                                   stride=1)
                                        )
        self.res_blocks.apply(fn=init_weights)

        self.up_block = nn.Sequential(UpBlock(in_channels=64,
                                              out_channels=64,
                                              kernel_size=5,
                                              stride=1,
                                              scale=2,
                                              mode='bilinear'),
                                      UpBlock(in_channels=64,
                                              out_channels=64,
                                              kernel_size=5,
                                              stride=1,
                                              scale=2,
                                              mode='bilinear'),
                                      UpBlock(in_channels=64,
                                              out_channels=64,
                                              kernel_size=5,
                                              stride=1,
                                              scale=2,
                                              mode='bilinear')
                                      )
        self.up_block.apply(fn=init_weights)

        self.last_block = nn.Sequential(BottleNeck(in_channels=64,
                                                   out_channels=8,
                                                   kernel_size=3,
                                                   stride=1),
                                        BottleNeck(in_channels=8,
                                                   out_channels=8,
                                                   kernel_size=3,
                                                   stride=1),
                                        nn.Conv2d(in_channels=8,
                                                  out_channels=3,
                                                  kernel_size=3,
                                                  padding=1,
                                                  stride=1),
                                        # nn.InstanceNorm2d(3),
                                        nn.BatchNorm2d(3),
                                        nn.ReLU()
                                        )
        self.last_block.apply(fn=init_weights)

    def forward(self, x):
        y = self.init_layer(x)
        y = self.down_block_1(y)
        y = self.down_block_2(y)
        y = self.down_block_3(y)
        y = self.res_blocks(y)
        y = self.up_block(y)
        y = self.last_block(y)

        return y


class BottleNeck(nn.Module):
    """
    https://github.com/KushajveerSingh/Photorealistic-Style-Transfer/blob/master/src/hrnet.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # self.in_c = in_channels
        # self.out_c = out_channels

        self.identity_block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels // 4,
                                                      kernel_size=1,
                                                      padding=0,
                                                      stride=1),
                                            nn.InstanceNorm2d(out_channels // 4),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=out_channels // 4,
                                                      out_channels=out_channels // 4,
                                                      kernel_size=kernel_size,
                                                      padding=kernel_size // 2,
                                                      stride=stride),
                                            nn.InstanceNorm2d(out_channels // 4),
                                            nn.ReLU(),
                                            nn.Conv2d(in_channels=out_channels // 4,
                                                      out_channels=out_channels,
                                                      kernel_size=1,
                                                      padding=0,
                                                      stride=1),
                                            nn.InstanceNorm2d(out_channels),
                                            nn.ReLU(),
                                            )

        # init.normal_(x, mean=0, std=0.1)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=1,
                                                    stride=stride),
                                          nn.InstanceNorm2d(out_channels),
                                          )
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.identity_block(x)

        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        out += residual
        return out


class UpBlock(nn.Module):
    """
    https://github.com/KushajveerSingh/Photorealistic-Style-Transfer/blob/master/src/hrnet.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale=2, mode='nearest'):
        super().__init__()
        self.scale = scale
        self.mode = mode

        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            padding=kernel_size // 2,
                                            stride=stride),
                                  nn.ReLU(),
                                  nn.InstanceNorm2d(out_channels),
                                  nn.Conv2d(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            padding=kernel_size // 2,
                                            stride=stride),
                                  nn.ReLU(),
                                  nn.InstanceNorm2d(out_channels)
                                  )

    def forward(self, x):
        y = F.interpolate(input=x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        y = self.conv(y)

        return y


def init_weights(module):
    if type(module) in [nn.Conv2d, nn.Linear]:
        module.weight.data.normal_(mean=0, std=0.1)
        module.bias.data.fill_(0)


class STData(Dataset):

    def __init__(self, train=True, transform=T.ToTensor()):

        self.train = train
        self.transform = transform

        self.train_folder = "images/train"
        self.test_folder = "images/test"

        self.train_set = DataFrame(data=listdir(self.train_folder), columns=["filename"])
        self.train_set["filename"] = self.train_folder + self.train_set["filename"]

        self.test_set = DataFrame(data=listdir(self.test_folder), columns=["filename"])
        self.test_set["filename"] = self.test_folder + self.test_set["filename"]

    def __getitem__(self, item):

        if self.train:
            image = Image.open(self.train_set.iloc[item, 0])
        else:
            image = Image.open(self.test_set.iloc[item, 0])

        return self.transform(image)

    def __len__(self):

        if self.train:
            return self.train_set.size
        else:
            return self.test_set.size
