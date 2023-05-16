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
            train_iter = iter(train_dataloader)
            next(train_iter)
            gen_view_result(next(train_iter), pipeline, autoEncoder)