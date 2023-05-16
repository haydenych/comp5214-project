import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4,5,6,7"
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

if __name__ == '__main__': 
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

    train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_trainlist.txt')
    train_dataloader = DataLoader(dataset = train_data, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    fixed_data = train_data[5000]

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
    
    for epoch in range(config.num_epochs):
        #print(autoEncoder.device)
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        if epoch<= config.starting_epoch:
            continue
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
        
        for step, batch in enumerate(train_dataloader):
            # img1 = batch[:,0].squeeze().to(device)
            # img2 = batch[:,1].squeeze().to(device)
            # img3 = batch[:,2].squeeze().to(device)
            img1 = batch[:,0].squeeze()
            img2 = batch[:,1].squeeze()
            img3 = batch[:,2].squeeze()
            
            img1_code, _, _ = autoEncoder.module.encode(img1)
            img2_code, _, _ = autoEncoder.module.encode(img2)
            img3_code, _, _ = autoEncoder.module.encode(img3)
            
            #noise = torch.randn(img2_code.shape).to(device)
            noise = torch.randn(img2_code.shape).to(img2_code.device)
            bs = img2_code.shape[0]
            
            # timesteps = torch.randint(
            #     0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            # ).long()
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device = img2_code.device
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
            pipeline = self_DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            
            # if (os.path.exists(config.output_dir)):
            #     pipeline.from_pretrained(config.output_dir)
            #     print("loaded", flush=True)

            # if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            #     evaluate(config, epoch, pipeline)

            if ((epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1):
                #pipeline.save_pretrained(config.output_dir)
                gen_result(fixed_data, pipeline, autoEncoder, epoch)
                accelerator.save_state("output")
                print("saved", flush=True)
