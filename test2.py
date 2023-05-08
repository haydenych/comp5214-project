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

config = get_config()

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=256*3,  # the number of input channels, 3 for RGB images
    out_channels=256,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    #block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    block_out_channels=(256, 256, 512, 512),
    down_block_types=(
        # "DownBlock2D",  # a regular ResNet downsampling block
        # "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
    ),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
timesteps = torch.LongTensor([50])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auto_config = OmegaConf.load("config/16384_model.yaml")
autoEncoder = vqgan.VQModel(**auto_config.model.params)
autoEncoder.init_from_ckpt("ckpts/16384_last.ckpt")
autoEncoder.to(device)

train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_trainlist.txt')
train_dataloader = DataLoader(dataset = train_data, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, autoEncoder, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
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
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            img1 = batch[:,0].squeeze().to(device)
            img2 = batch[:,1].squeeze().to(device)
            img3 = batch[:,2].squeeze().to(device)
            
            img1_code, _, _ = autoEncoder.encode(img1)
            img2_code, _, _ = autoEncoder.encode(img2)
            img3_code, _, _ = autoEncoder.encode(img3)
            
            noise = torch.randn(img2_code.shape).to(device)
            bs = img2_code.shape[0]
            
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()
            
            noisy_images = noise_scheduler.add_noise(img2_code, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(torch.cat((img1_code,noisy_images,img3_code), 1), timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                print("saved", flush=True)
    
from accelerate import notebook_launcher

args = (config, model, autoEncoder, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)