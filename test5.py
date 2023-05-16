import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import sys
from accelerate import Accelerator, notebook_launcher
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import math
sys.path.insert(1, './taming-transformers')
from utils import *
from tqdm.auto import tqdm
from pathlib import Path
from dataset.Vimeo90k import Vimeo90k
from omegaconf import OmegaConf
from taming.models import vqgan

import argparse
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


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    config = get_config()
    print(config)

    model = get_unet(config)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    timesteps = torch.LongTensor([50])

    #device = torch.device('cuda')
    auto_config = OmegaConf.load("config/model.yaml")
    autoEncoder = vqgan.VQModel(**auto_config.model.params)
    autoEncoder.init_from_ckpt("ckpts/model.ckpt")
    #autoEncoder.to(device)

    train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_testlist.txt')
    train_dataloader = DataLoader(dataset = train_data, batch_size=config.train_batch_size, shuffle=False, drop_last=True)

    #fixed_data = train_data[1000]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_dir=os.path.join(config.output_dir, "logs"),
        project_dir="output"
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
        
    model, optimizer, train_dataloader, lr_scheduler, autoEncoder = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, autoEncoder
    )
    
    global_step = 0
    if accelerator.is_main_process and config.retrain:
        if (os.path.exists(config.output_dir)):
            pipeline = self_DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            #gen_result(fixed_data, pipeline, autoEncoder, epoch)
            #pipeline.from_pretrained(config.output_dir)
            #accelerator.save_state("output")
            accelerator.load_state("output")
            config.retrain = False
            #gen_result(fixed_data, pipeline, autoEncoder, epoch)
            print("loaded", flush=True)
            
            inceptionV3 = InceptionV3().to(model.device)
            inceptionV3.eval()
            activations = np.empty((len(train_data), 2048))
            start_idx = 0
            
            for i, batch in tqdm(enumerate(train_dataloader), total=50):
                if i>=50:
                    break
                with torch.no_grad():
                    reconstructed_imgs = gen_image(batch, pipeline, autoEncoder)
                    acts = inceptionV3(reconstructed_imgs)[0].squeeze(3).squeeze(2).detach().cpu().numpy()
                activations[start_idx: start_idx + acts.shape[0]] = acts
                start_idx += acts.shape[0]
                
            reconstr_acts = np.array(activations)
            
            for i, batch in tqdm(enumerate(train_dataloader), total=50):
                if i>=50:
                    break
                with torch.no_grad():
                    acts = inceptionV3(batch[:,1].to(model.device))[0].squeeze(3).squeeze(2).detach().cpu().numpy()
                activations[start_idx: start_idx + acts.shape[0]] = acts
                start_idx += acts.shape[0]
            acts = np.array(activations)
            
        print(calculate_fid(reconstr_acts, acts))
                
            