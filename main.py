# Style transfer
#
# Basic algorithm to follow:
# TODO: algo.

# TODO: total variational loss

#
#  repeat style image to match batch size.
import torch

from torch import cuda, optim
from torchvision import models, transforms as T
from torchvision.utils import save_image
from torchsummary import summary
from PIL import Image

import argparse as ap

from classes import Extractor, StyleTransfer


def main(args):

    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    model = StyleTransfer().to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=0,
                           amsgrad=False
                           )

    # summary(model, (3, 500, 500), 1)

    test_tr = T.Compose([T.CenterCrop(1280),
                         T.Resize(500),
                         T.ToTensor()
                         ])

    style_tr = T.Compose([T.CenterCrop(895),
                          T.Resize(500),
                          T.ToTensor()
                          ])

    content_image = Image.open("test/test.jpg")
    content_image = test_tr(content_image).to(device).unsqueeze(dim=0)
    # content_image = torch.zeros(2).unsqueeze(dim=0)
    style_image = Image.open("style/abstract2.jpg")
    style_image = style_tr(style_image).to(device).unsqueeze(dim=0)
    # TODO: accomodate batch size

    ex = Extractor(content=content_image,
                   style=style_image,
                   content_wt=0.1,
                   style_wt=2.0
                   ).to(device)
    ex.train()

    for i in range(args.instances):

        optimizer.zero_grad()

        st_im = model(content_image)

        save_image(tensor=st_im,
                   filename="output/ST_{0}.jpg".format(i),
                   nrow=1,
                   normalize=False
                   )

        ex(st_im)

        losses = ex.end_game()

        print("Iteration: {0}\tLosses: {1}".format(i+1, losses.item()))

        losses.backward(retain_graph=True)
        # losses.backward()

        optimizer.step()

        cuda.empty_cache()


if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("-i",
                        "--instances",
                        type=int,
                        default=10,
                        help="Number of Instances")

    arg = parser.parse_args()

    main(arg)
