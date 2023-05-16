from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
import torch
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import math
import os
import sys
sys.path.insert(1, './taming-transformers')
from utils import *
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
from dataset.Vimeo90k import Vimeo90k
from omegaconf import OmegaConf
from taming.models import vqgan
import matplotlib.pyplot as plt


config = get_config()
print(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoconfig = OmegaConf.load("config/16384_model.yaml")
model = vqgan.VQModel(**autoconfig.model.params)
model.init_from_ckpt("ckpts/16384_last.ckpt")
model.to(device)

Umodel = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=256*3,  # the number of input channels, 3 for RGB images
    out_channels=256,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    #block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    block_out_channels=(256, 256, 512),
    down_block_types=(
        # "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D",
        #"DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        #"UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
    ),
).to(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_trainlist.txt')
train_dataloader = DataLoader(dataset = train_data, batch_size=config.train_batch_size, shuffle=True)
optimizer = torch.optim.AdamW(Umodel.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    log_with="tensorboard",
    logging_dir=os.path.join(config.output_dir, "logs"),
)
if accelerator.is_main_process:
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
    accelerator.init_trackers("train_example")
    
Umodel, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    Umodel, optimizer, train_dataloader, lr_scheduler
)


pipeline = self_DDPMPipeline(unet=accelerator.unwrap_model(Umodel), scheduler=noise_scheduler)
pipeline.from_pretrained("output")


im1 = train_data[5000][0].unsqueeze(0).to(device)
im3 = train_data[5000][2].unsqueeze(0).to(device)
im1_code, _, _ = model.encode(im1)
im3_code, _, _ = model.encode(im3)
im1_code.detach()
im3_code.detach()
print(im1_code.shape)

images = pipeline(im1_code, im3_code, batch_size=1, generator=torch.Generator(device='cuda').manual_seed(0))
print(im1_code)
print(images.cpu())

out1 = model.decode(im1_code).detach()
out2 = model.decode(images).detach()
out3 = model.decode(im3_code).detach()

fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 3, 1)
plt.imshow(image_ten2pil(out1))
fig.add_subplot(1, 3, 2)
plt.imshow(image_ten2pil(out2))
fig.add_subplot(1, 3, 3)
plt.imshow(image_ten2pil(out3))
plt.savefig("test.png")