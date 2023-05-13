import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(1, './taming-transformers')

import argparse
import numpy as np
import torch

from scipy import linalg
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Vimeo90k import Vimeo90k
from models.inception import InceptionV3

from taming.models import vqgan as VQGAN

def calculate_fid(act1, act2, eps=1e-6):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"fid calculation produces singular product: adding {eps} to diagonal of cov estimates")
        
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))

        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_activations(args, dataset, use_reconstructed_imgs=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    inceptionV3 = InceptionV3().to(device)
    inceptionV3.eval()

    activations = np.empty((len(dataset), 2048))
    start_idx = 0

    if use_reconstructed_imgs:
        config = OmegaConf.load(args.vqgan_config_path)
        vqgan = VQGAN.VQModel(**config.model.params)
        vqgan.init_from_ckpt(args.vqgan_ckpt_path)
        vqgan.to(device)
        vqgan.eval()

        for im1, im2, im3 in tqdm(dataloader):
            imgs = im1.to(device)

            with torch.no_grad():
                reconstructed_imgs, _ = vqgan(imgs)
                acts = inceptionV3(reconstructed_imgs)[0].squeeze(3).squeeze(2).detach().cpu().numpy()

            activations[start_idx: start_idx + acts.shape[0]] = acts

            start_idx += acts.shape[0]
    
    else:
        for im1, im2, im3 in tqdm(dataloader):
            imgs = im1.to(device)

            with torch.no_grad():
                acts = inceptionV3(imgs)[0].squeeze(3).squeeze(2).detach().cpu().numpy()

            activations[start_idx: start_idx + acts.shape[0]] = acts

            start_idx += acts.shape[0]

    return np.array(activations)

def main(args):
    dataset = Vimeo90k(path_to_dataset=args.dataset_path, datalist_filename=args.datalist_path)

    act0 = compute_activations(args, dataset, use_reconstructed_imgs=True)
    act1 = compute_activations(args, dataset, use_reconstructed_imgs=False)

    fid = calculate_fid(act0, act1)

    print(fid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID")

    parser.add_argument('--batch-size', type=int, default=16, help='Input batch size for training (default: 16)')

    parser.add_argument('--dataset-path', type=str, default="/data/chyuam/COMP5214/Project/data/vimeo_triplet/sequences", help="Path to dataset")
    parser.add_argument('--datalist-path', type=str, default="/data/chyuam/COMP5214/Project/data/vimeo_triplet/tri_testlist.txt", help="Path to data list")

    parser.add_argument('--vqgan-config-path', type=str, default="model_f8_16384.yaml", help="Path to config")
    parser.add_argument('--vqgan-ckpt-path', type=str, default="last_f8_16384.ckpt", help="Path to checkpoint")

    args = parser.parse_args()

    main(args)
