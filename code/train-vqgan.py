"""
Code inspired by https://github.com/dome272/VQGAN-pytorch
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys
sys.path.insert(1, './taming-transformers')

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from PIL import Image
from torchvision import utils as vutils
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Vimeo90k import Vimeo90k

from taming.models import vqgan as VQGAN
from taming.modules.discriminator.model import NLayerDiscriminator
from taming.modules.losses.lpips import LPIPS


def adopt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
        disc_factor = value
    return disc_factor


def calculate_lambda(last_layer_weight, perceptual_loss, gan_loss):
    perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
    gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

    λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
    λ = torch.clamp(λ, 0, 1e4).detach()
    return 0.8 * λ


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_tensor_image(image, path):
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy()),
        transforms.Lambda(lambda t: np.clip(t, 0, 255).astype(np.uint8)),
        transforms.ToPILImage(mode='RGB'),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 

    image = reverse_transforms(image)
    image.save(path)

def create_dirs(args):
    if not os.path.exists(os.path.join(args.out_dir, "results")):
        os.makedirs(os.path.join(args.out_dir, "results"))

    if not os.path.exists(os.path.join(args.out_dir, "intermediates")):
        os.makedirs(os.path.join(args.out_dir, "intermediates"))

    if not os.path.exists(os.path.join(args.out_dir, "checkpoints")):
        os.makedirs(os.path.join(args.out_dir, "checkpoints"))   


def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config = OmegaConf.load(args.config_path)
    vqgan = VQGAN.VQModel(**config.model.params)
    vqgan.init_from_ckpt(args.ckpt_path)
    vqgan.to(device)

    discriminator = NLayerDiscriminator()
    discriminator.apply(weights_init)
    discriminator.to(device)

    dataset = Vimeo90k(path_to_dataset=args.dataset_path, datalist_filename=args.datalist_path)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    loss_fn = LPIPS().eval().to(device)
    opt, _ = vqgan.configure_optimizers(args.learning_rate)
    opt_vq, opt_disc = opt[0], opt[1]

    num_batches = len(dataloader)

    for epoch in range(args.epochs):
        if epoch < args.startfrom_epoch:
            continue

        vqgan.train()
        discriminator.train()

        with tqdm(range(len(dataloader))) as pbar:
            for i, (im1, im2, im3) in zip(pbar, dataloader):
                imgs = im1.to(device)

                decoded_imgs, q_loss = vqgan(imgs)

                disc_real = discriminator(imgs)
                disc_fake = discriminator(decoded_imgs)

                disc_factor = adopt_weight(args.disc_factor, epoch * num_batches + i, threshold=args.disc_start)

                perceptual_loss = loss_fn(imgs, decoded_imgs)
                rec_loss = torch.abs(imgs - decoded_imgs)

                perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                perceptual_rec_loss = perceptual_rec_loss.mean()

                g_loss = -torch.mean(disc_fake)

                last_layer_weights = vqgan.get_last_layer()
                lamb = calculate_lambda(last_layer_weights, perceptual_rec_loss, g_loss)

                vq_loss = perceptual_rec_loss + q_loss + disc_factor * lamb * g_loss

                d_loss_real = torch.mean(F.relu(1. - disc_real))
                d_loss_fake = torch.mean(F.relu(1. + disc_fake))

                gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                opt_vq.zero_grad()
                vq_loss.backward(retain_graph=True)

                opt_disc.zero_grad()
                gan_loss.backward()

                opt_vq.step()
                opt_disc.step()

                if i % 200 == 0:
                    with torch.no_grad():
                        real_fake_images = torch.cat((imgs[:4], decoded_imgs[:4]))
                        vutils.save_image(real_fake_images, os.path.join(args.out_dir, "intermediates", f"{epoch}_{i}.jpg"), nrow=args.batch_size)

                pbar.set_postfix(
                    VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                    GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                )
                pbar.update(0)

        torch.save( {
                        'state_dict': vqgan.state_dict()
                    }, os.path.join(args.out_dir, "checkpoints", f"vqgan_epoch_{epoch}.pt"))

        vqgan.eval()

        sample_real_imgs = torch.cat((dataset[5000][0].unsqueeze(0), dataset[25990][0].unsqueeze(0))).to(device)
        z, _, _ = vqgan.encode(sample_real_imgs)
        sample_fake_imgs = vqgan.decode(z)

        sample_real_fake_imgs = torch.cat((sample_real_imgs[:4], sample_fake_imgs[:4]))
        vutils.save_image(sample_real_fake_imgs, os.path.join(args.out_dir, "results", f"epoch_{epoch}.jpg"), nrow=2)

        del sample_real_imgs
        del z
        del sample_fake_imgs
        del sample_real_fake_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-7, help='Learning rate')
    
    parser.add_argument('--disc-start', type=int, default=0, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--config-path', type=str, default="model_16384.yaml", help="Path to config")
    parser.add_argument('--ckpt-path', type=str, default="last_16384.ckpt", help="Path to checkpoint")
    parser.add_argument('--dataset-path', type=str, default="/data/chyuam/COMP5214/Project/data/vimeo_triplet/sequences", help="Path to dataset")
    parser.add_argument('--datalist-path', type=str, default="/data/chyuam/COMP5214/Project/data/vimeo_triplet/tri_trainlist.txt", help="Path to train list")
    parser.add_argument('--out_dir', type=str, default="out", help="Output directory")

    parser.add_argument('--startfrom-epoch', type=int, default=0, help="For training resumption")

    args = parser.parse_args()

    create_dirs(args)
    train(args)