# Style transfer
#
# Basic algorithm to follow:
# TODO: algo.

# TODO: total variational loss

#
#  repeat style image to match batch size.
import torch

from torch import cuda, optim, nn
from torchvision import transforms as T
from torchvision.utils import save_image
from torchsummary import summary
from PIL import Image
from support.invert import Invert
# from torchviz import make_dot, make_dot_from_trace

import argparse as ap
import time

from classes import Extractor, StyleTransfer, STData


def main(args):
    device = torch.device("cuda:0" if cuda.is_available() and not args.cpu else "cpu")

    model = StyleTransfer().to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=args.lamb,
                           amsgrad=True
                           )

    transform = T.Compose([T.Resize(args.size),
                           T.CenterCrop(args.size),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ])

    end_tr = T.Compose([T.ToPILImage(mode="RGB"),
                        Invert(),
                        T.ToTensor()
                        ])

    # content_image = Image.open("test/emp.jpg")
    content_image = Image.open(args.content)
    content_image = transform(content_image).to(device).unsqueeze(dim=0)

    # style_image = Image.open("style/mosaic.jpg")
    style_image = Image.open(args.style)
    style_image = transform(style_image).to(device).unsqueeze(dim=0)

    ex = Extractor(content=content_image,
                   style=style_image,
                   content_wt=args.content_weight,
                   style_wt=args.style_weight,
                   device=device
                   ).to(device)

    if args.verbose > 1:
        summary(model.cuda(), (3, args.size, args.size), 1)
        summary(ex.cuda(), (3, args.size, args.size), 1)
    # exit(0)

    ex.train(False)
    print("Starting training")
    start_time = time.time()
    timing = []

    for i in range(args.iterations):

        lap = time.time()
        optimizer.zero_grad()
        st_im = model(content_image)

        # loss_1, loss_2 = ex(st_im)
        losses = ex(st_im)

        tv_loss = torch.sum(torch.abs(st_im[:, :, :, :-1] - st_im[:, :, :, 1:])) + \
                  torch.sum(torch.abs(st_im[:, :, :-1, :] - st_im[:, :, 1:, :]))

        losses += tv_loss * args.tv_weight
        # TODO: add total variational loss

        losses.backward()
        # https://jdhao.github.io/2017/11/12/pytorch-computation-graph/
        # This link may hold the explanation for failed graphs.
        # Needed to add ".detach()" to __init__() of the Extractor class.

        optimizer.step()

        # cuda.empty_cache()

        # inv = end_tr(normalised.squeeze(0)).unsqueeze(dim=0)
        # inv = end_tr(st_im.squeeze(0).cpu()).unsqueeze(dim=0).to(device)

        lap_end = time.time()
        timing.append(lap_end - lap)

        if (i + 1) % 10 == 0 or i == 0:
            save_image(tensor=torch.cat([content_image, st_im, style_image]),
                       filename="output/ST_{0}.jpg".format(i),
                       nrow=3,
                       normalize=True,
                       scale_each=True
                       )

            if args.verbose > 0:
                print("Iteration: {0}\tLosses: {1:.06f}".format(i + 1, losses.item()))

    fn = time.strftime("%Y%m%d_%H%M%S")
    # torch.save(obj=model, f="model_{0}.pt".format(time.strftime("%Y%m%d_%H%M%S")))
    torch.save(obj=model.state_dict(), f="model_{0}.pt".format(fn))
    print("\n\nAverage time per image: {0:.5f}\nFilename: {1}\n".format(sum(timing)/len(timing), fn))

    # save_image(tensor=torch.cat([content_image, model(content_image), style_image]),
    #            filename="output/ST_final.jpg",
    #            nrow=3,
    #            normalize=True,
    #            scale_each=True
    #            )


def bsd(args):
    device = torch.device("cuda:0" if cuda.is_available() and not args.cpu else "cpu")

    model = StyleTransfer().to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr,
                           weight_decay=args.lamb,
                           amsgrad=True
                           )

    transform = T.Compose([T.Resize(args.size),
                           T.CenterCrop(args.size),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ])

    content_image = Image.open(args.content)
    content_image = transform(content_image).to(device).unsqueeze(dim=0)

    style_image = Image.open(args.style)
    style_image = transform(style_image).to(device).unsqueeze(dim=0)

    ex = Extractor(content=content_image,
                   style=style_image,
                   content_wt=args.content_weight,
                   style_wt=args.style_weight,
                   device=device
                   ).to(device)

    if args.verbose > 1:
        summary(model.cuda(), (3, args.size, args.size), 1)
        summary(ex.cuda(), (3, args.size, args.size), 1)

    ex.train(False)
    print("Starting training")
    start_time = time.time()
    timing = []

    for i in range(args.iterations):
        pass


if __name__ == '__main__':
    parser = ap.ArgumentParser()

    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate (Default 1e-3)")

    parser.add_argument("-i",
                        "--iterations",
                        type=int,
                        default=500,
                        help="Number of Iterations (Default 500)")

    parser.add_argument("--size",
                        type=int,
                        default=480,
                        help="Size of output image (Default 480)")

    parser.add_argument("--style-weight",
                        type=float,
                        default=1.0,
                        help="Style weight (Default 1.0)")

    parser.add_argument("--content-weight",
                        type=float,
                        default=1.0,
                        help="Content weight (Default 1.0)")

    parser.add_argument("--tv-weight",
                        type=float,
                        default=1e-10,
                        help="Total loss weight (Default 1e-10)")

    parser.add_argument("-l",
                        "--lamb",
                        type=float,
                        default=0.1,
                        help="Weight decay for Optimizer (Default 0.1)")

    parser.add_argument("--cpu",
                        action="store_true",
                        help="Use CPU for training instead of GPU")

    parser.add_argument("-v",
                        "--verbose",
                        action="count",
                        help="Verbose Level")

    parser.add_argument("-s",
                        "--style",
                        type=str,
                        default="style/gogh.jpg",
                        help="Style image (Default \"style/gogh.jpg\")")

    parser.add_argument("-c",
                        "--content",
                        type=str,
                        default="test/emp.jpg",
                        help="Content image (Default \"test/emp.jpg\")")

    parser.add_argument("-V",
                        "--version",
                        action="version",
                        version="Style Transfer v0.1 by Ved Harish Valsangkar",
                        help="Version")

    parser.add_argument("--run-bsd",
                        action="store_true",
                        help="Use BSD Dataset for training instead of single image")

    arg = parser.parse_args()

    cuda.empty_cache()

    if arg.run_bsd:
        bsd(arg)
    else:
        main(arg)

#     https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
