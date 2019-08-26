import torch
from torch import nn

from torch.utils.data import Dataset  # , DataLoader
from torchvision import models  # , transforms as T
# from torchvision.utils import save_image


class Extractor(nn.Module):

    def __init__(self, content, style, content_wt=1.0, style_wt=0.2, total_wt=1.0, device=None):

        super(Extractor, self).__init__()

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # self.base = models.vgg16(pretrained=True).features.to(self.device)
        self.base = models.vgg19(pretrained=True).features.to(self.device)
        self.base.eval()

        self.ST_WEIGHT = content_wt
        self.CT_WEIGHT = style_wt
        self.TOT_WEIGHT = total_wt

        # [2, 7, 14, 23, 32]
        # for i in ['3', '8', '24', '33']:
        # for i in ['2', '7', '23', '32']:
        # for i in ['3', '8', '22', '29']:
        #     self.base._modules[i].register_forward_hook(hook=self.style_hook)

        # [14]
        # self.base._modules['15'].register_forward_hook(hook=self.c_and_s_hook)
        # self.base._modules['14'].register_forward_hook(hook=self.c_and_s_hook)

        # self.base.register_forward_hook(hook=self.end_game)

        modules = list(self.base.children())

        self.layer1 = nn.Sequential(*modules[:4])
        self.layer2 = nn.Sequential(*modules[4:9])
        self.layer3 = nn.Sequential(*modules[9:18])
        self.layer3_5 = nn.Sequential(*modules[18:23])      # Content layer
        self.layer4 = nn.Sequential(*modules[23:27])
        self.layer5 = nn.Sequential(*modules[27:36])
        self.end_layer = nn.Sequential(*modules[36:])

        # def forward(self, x, content, style):

        y = self.layer1(style)
        self.st_loss_1 = StyleLoss(y)

        y = self.layer2(y)
        self.st_loss_2 = StyleLoss(y)

        y = self.layer3(y)
        self.st_loss_3 = StyleLoss(y)

        y = self.layer4(self.layer3_5(y))
        self.st_loss_4 = StyleLoss(y)

        y = self.layer5(y)
        self.st_loss_5 = StyleLoss(y)

        y = self.layer3_5(self.layer3(self.layer2(self.layer1(content))))
        self.ct_loss = ContentLoss(y)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        y = self.layer1(x)
        style_loss = self.st_loss_1(y)

        y = self.layer2(y)
        style_loss += self.st_loss_2(y)

        y = self.layer3(y)
        style_loss += self.st_loss_3(y)

        y = self.layer3_5(y)
        content_loss = self.ct_loss(y)

        y = self.layer4(y)
        style_loss += self.st_loss_4(y)

        y = self.layer5(y)
        style_loss += self.st_loss_5(y)

        y = self.end_layer(y)

        return style_loss * self.ST_WEIGHT, content_loss * self.CT_WEIGHT, y


class StyleTransfer(nn.Module):

    def __init__(self, intrim_layers=6, resnet=False, bias=False):

        super(StyleTransfer, self).__init__()

        self.resnet = resnet

        self.init_layer = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=8,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=bias),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(8),
                                        # nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=8,
                                                  out_channels=16,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=bias),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        # nn.ReflectionPad2d(1),
                                        nn.Conv2d(in_channels=16,
                                                  out_channels=32,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=bias),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32)
                                        )

        layers = []

        for i in range(intrim_layers):
            layers.append(Residual(channels=32,
                                   k_size=3,
                                   resnet=resnet,
                                   bias=bias)
                          )
        #     layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(nn.Conv2d(in_channels=32,
                                                   out_channels=8,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   bias=bias),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(8),
                                         # nn.ReflectionPad2d(1),
                                         nn.Conv2d(in_channels=8,
                                                   out_channels=3,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   bias=bias),
                                         nn.ReLU(),
                                         nn.BatchNorm2d(3)
                                         )

    def forward(self, x):
        jump = self.init_layer(x)
        if self.resnet:
            y = self.layers(jump) + jump
        else:
            y = self.layers(jump)
        out = self.final_layer(y)

        return out


class Residual(nn.Module):

    def __init__(self, channels, k_size=3, stride=1, resnet=False, bias=False):
        super(Residual, self).__init__()

        self.resnet = resnet

        self.conv = nn.Sequential(nn.Conv2d(in_channels=channels,
                                            out_channels=channels,
                                            kernel_size=k_size,
                                            stride=stride,
                                            padding=k_size // 2,
                                            bias=bias),
                                  nn.ReLU(),
                                  # nn.ReflectionPad2d(k_size // 2),
                                  # nn.Conv2d(in_channels=channels,
                                  #           out_channels=channels,
                                  #           kernel_size=k_size,
                                  #           stride=stride,
                                  #           bias=bias),
                                  # nn.ReLU(),
                                  nn.BatchNorm2d(channels)
                                  )

    def forward(self, x):
        if self.resnet:
            # return self.conv(x) + self.conv2(x)
            return self.conv(x) + x
        else:
            return self.conv(x)


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
        return self.crit(self.gram_matrix(x), self.target)
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
