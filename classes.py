import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T


# class vgg_hook(models.VGG):
class Extractor(nn.Module):

    def __init__(self, content, style, content_wt=1.0, style_wt=0.2):

        super(Extractor, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.base = models.vgg19(pretrained=True).features.to(self.device)
        self.base.eval()
        # self.base.zero_grad()

        self.ST_WEIGHT = content_wt
        self.CT_WEIGHT = style_wt

        self.gram_list_ip = []
        self.gram_list_st = []
        self.ct_feat = None
        self.ip_feat = None

        self.ct_loss = nn.MSELoss()
        self.st_loss = nn.MSELoss()

        self.ct_losses = 0
        self.st_losses = 0

        # [2, 7, 14, 23, 32]
        # for i in ['2', '7', '14', '23', '32']:
        for i in ['3', '8', '24', '33']:
            self.base._modules[i].register_forward_hook(hook=self.style_hook)

        # [14]
        self.base._modules['15'].register_forward_hook(hook=self.c_and_s_hook)

        # self.base.register_forward_hook(hook=self.end_game)

        self.pass_ = "C"
        self.base(content)

        self.pass_ = "S"
        self.base(style)

    # def forward(self, x, content, style):
    def forward(self, x):

        self.pass_ = "X"
        self.base(x)

        # self.pass_ = "C"
        # self.base(content)
        #
        # self.pass_ = "S"
        # self.base(style)

    def c_and_s_hook(self, module, x, output):
        """
        Forward hook for storing output for content, style and generated images.

        :param module:  Module object for hook
        :param x:       Input to the module
        :param output:  Output of the module
        :return:        None
        """

        if self.pass_ == "C":
            self.ct_feat = output
        elif self.pass_ == "S":
            self.gram_list_st.append(output)
        elif self.pass_ == "X":
            self.ip_feat = output
            self.gram_list_ip.append(output)

    def style_hook(self, module, x, output):
        """
        Forward hook for storing output for style and generated images.

        :param module:  Module object for hook
        :param x:       Input to the module
        :param output:  Output of the module
        :return:        None
        """
        # print(self.__class__.__name__)
        # print(self2.__class__.__name__)

        if self.pass_ == "S":
            self.gram_list_st.append(output)
        elif self.pass_ == "X":
            self.gram_list_ip.append(output)

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

    def end_game(self):

        self.ct_losses = self.ct_loss(self.ip_feat, self.ct_feat) * self.CT_WEIGHT

        for ip, st in zip(self.gram_list_ip, self.gram_list_st):
            self.st_losses += self.st_loss(self.gram_matrix(ip), self.gram_matrix(st)) * self.ST_WEIGHT

        self.ip_feat = None
        self.gram_list_ip.clear()

        return self.ct_losses + self.st_losses


class StyleTransfer(nn.Module):

    def __init__(self, bias=False):

        super(StyleTransfer, self).__init__()

        self.init_layer = Conv(in_ch=3,
                               out_ch=8,
                               k_size=3,
                               bias=bias)

        layers = []

        for i in range(3):
            layers.append(Conv(in_ch=8 * 2 ** i,
                               out_ch=16 * 2 ** i,
                               k_size=3,
                               bias=bias)
                          )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(16 * 2 ** i))

        for i in range(2, -1, -1):
            layers.append(Conv(in_ch=16 * 2 ** i,
                               out_ch=8 * 2 ** i,
                               k_size=3,
                               bias=bias)
                          )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(8 * 2 ** i))

        self.layers = nn.Sequential(*layers)

        self.final_layer = Conv(in_ch=8,
                                out_ch=3,
                                k_size=5,
                                bias=bias)

        # TODO: define a model for actual Style Transfer

    def forward(self, x):
        y = self.init_layer(x)
        y = self.layers(y)
        y = self.final_layer(y)
        return y


class Conv(nn.Module):

    def __init__(self, in_ch, out_ch, k_size=3, stride=1, bias=False):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(nn.ReflectionPad2d(k_size // 2),
                                  nn.Conv2d(in_channels=in_ch,
                                            out_channels=out_ch,
                                            kernel_size=k_size,
                                            stride=stride,
                                            bias=bias),
                                  nn.ReLU()
                                  )

    def forward(self, x):
        return self.conv(x)


class STDataSet(Dataset):

    def __init__(self):
        super(STDataSet, self).__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        return 0
