# Style transfer
#
# Basic algorithm to follow:
# TODO: algo.

# TODO: total variational loss

#
#  repeat style image to match batch size.
import torch

from torch import cuda, optim, nn
from torchvision import models, transforms as T
from torchvision.utils import save_image
from torchsummary import summary
from PIL import Image
from support.invert import Invert
# from torchviz import make_dot, make_dot_from_trace

import argparse as ap

from classes import Extractor, StyleTransfer


def main(args):

    device = torch.device("cuda:0" if cuda.is_available() and not args.cpu else "cpu")
    # device = torch.device("cpu")

    model = StyleTransfer(intrim_layers=args.layers,
                          resnet=True,
                          bias=True,
                          ).to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=args.lamb,
                           amsgrad=False
                           )

    # summary(model.cuda(), (3, 480, 480), 1)
    # exit(0)

    transform = T.Compose([T.Resize(480),
                           T.CenterCrop(480),
                           T.ToTensor(),
                           # T.Normalize([0.4, 0.4, 0.4], [0.2, 0.2, 0.2]),
                           ])

    end_tr = T.Compose([T.ToPILImage(mode="RGB"),
                        Invert(),
                        T.ToTensor()
                        ])

    content_image = Image.open("test/girl.jpg")
    content_image = transform(content_image).to(device).unsqueeze(dim=0)

    # style_image = Image.open("style/abstract2.jpg")
    style_image = Image.open("style/mosaic.jpg")
    style_image = transform(style_image).to(device).unsqueeze(dim=0)

    ex = Extractor(content=content_image,
                   style=style_image,
                   content_wt=args.content_weight,
                   style_wt=args.style_weight,
                   total_wt=args.total_weight,
                   device=device
                   ).to(device)

    total_crit = nn.MSELoss()

    # summary(ex.cuda(), (3, 480, 480), 1)
    # exit(0)

    ex.train(False)

    # model.ex = ex

    for i in range(args.iterations):

        # content_image = Variable(content_image, requires_grad=True)

        optimizer.zero_grad()
        st_im = model(content_image)

        # min_v = torch.min(st_im)
        # range_v = torch.max(st_im) - min_v
        # if range_v > 0:
        #     normalised = (st_im - min_v) / range_v
        # else:
        #     normalised = torch.zeros(st_im.size())
        # ex(normalised)

        # loss_1, loss_2 = ex(st_im)
        loss_1, loss_2, feat = ex(st_im)

        # feat2 = ex.base(content_image)

        # loss_3 = total_crit(feat, feat2) * args.total_weight

        losses = loss_1 + loss_2
        # losses = loss_1 + loss_2 + loss_3

        # ex(st_im)
        # losses = ex.end_game()

        # losses.backward()
        losses.backward(retain_graph=True)
        # https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
        # This link may hold the explanation for failed graphs.

        optimizer.step()
        # optimizer.step(closure)

        cuda.empty_cache()
        # st_im.clamp_(0, 1)

        # min_v = torch.min(st_im)
        # range_v = torch.max(st_im) - min_v
        # if range_v > 0:
        #     normalised = (st_im - min_v) / range_v
        # else:
        #     normalised = torch.zeros(st_im.size())

        # inv = end_tr(normalised.squeeze(0)).unsqueeze(dim=0)
        inv = end_tr(st_im.squeeze(0).cpu()).unsqueeze(dim=0).to(device)

        save_image(tensor=torch.cat([content_image, st_im, style_image]),
                   filename="output/ST_{0}.jpg".format(i),
                   nrow=3,
                   normalize=True,
                   scale_each=True
                   )

        # print("Iteration: {0}\tLosses ->\tCT: {1}\tST: {2}".format(i+1, ct_l.item(), st_l.item()))
        print("Iteration: {0}\tLosses: {1}".format(i+1, losses.item()))
        # ex.clear()

    # test_im = model(content_image)
    # ex = None
    # model.eval()
    # inv = end_tr(test_im.squeeze(0)).unsqueeze(dim=0)
    #
    # save_image(tensor=torch.cat([content_image, inv, style_image]),
    #            filename="output/ST_Test.jpg",
    #            nrow=3,
    #            normalize=False
    #            )


if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")

    parser.add_argument("-i",
                        "--iterations",
                        type=int,
                        default=10,
                        help="Number of Iterations (Default 10)")

    parser.add_argument("--style-weight",
                        type=float,
                        default=1e8,
                        help="Style weight (1e8)")

    parser.add_argument("--content-weight",
                        type=float,
                        default=1e4,
                        help="Content weight (1e4)")

    parser.add_argument("--total-weight",
                        type=float,
                        default=1.0,
                        help="Total loss weight (1)")

    parser.add_argument("--lamb",
                        type=float,
                        default=0.0,
                        help="Weight decay for optimizer")

    parser.add_argument("--cpu",
                        action="store_true",
                        help="Use CPU for training instead of GPU")

    parser.add_argument("-l",
                        "--layers",
                        type=int,
                        default=5,
                        help="Number of Layers in intermediate section")

    arg = parser.parse_args()

    main(arg)

#     https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
