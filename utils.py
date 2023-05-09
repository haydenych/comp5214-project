from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from dataclasses import dataclass
import os
from diffusers import DDPMPipeline, UNet2DModel
from typing import List, Optional, Tuple, Union


@dataclass
class TrainingConfig:
    image_size = 16  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    #mixed_precision = "no"
    output_dir = "output"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    
def get_config():
    return TrainingConfig

def get_unet(config):
    return UNet2DModel(
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
        "AttnDownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "AttnUpBlock2D",
        #"UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
    ),
)


def image_ten2pil(image):
    reverse_transforms = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy()),
        transforms.Lambda(lambda t: np.clip(t, 0, 255).astype(np.uint8)),
        transforms.ToPILImage(mode='RGB'),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    return reverse_transforms(image)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
def gen_result(img_set, pipeline, autoEncoder, num, device):
    im1_code,_,_ = autoEncoder.encode(img_set[0].unsqueeze(0).to(device))
    im3_code,_,_ = autoEncoder.encode(img_set[2].unsqueeze(0).to(device))
    images = pipeline(im1_code.detach(), im3_code.detach(), batch_size=1, generator=torch.Generator(device='cuda').manual_seed(0))
    out1 = model.decode(im1_code).detach().cpu()
    out2 = model.decode(images).detach().cpu()
    out3 = model.decode(im3_code).detach().cpu()
    
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image_ten2pil(out1))
    fig.add_subplot(1, 3, 2)
    plt.imshow(image_ten2pil(out2))
    fig.add_subplot(1, 3, 3)
    plt.imshow(image_ten2pil(out3))
    plt.savefig("output_img/res_"+str(num)+".png")
    
class self_DDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)
        self.register_modules(unet=unet, scheduler=scheduler)
    
    @torch.no_grad()
    def __call__(
        self,
        im1_code,
        im3_code,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Tuple:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        # Sample gaussian noise to begin loop
        # if isinstance(self.unet.config.sample_size, int):
        #     image_shape = (
        #         batch_size,
        #         self.unet.config.in_channels,
        #         self.unet.config.sample_size,
        #         self.unet.config.sample_size,
        #     )
        # else:
        #     image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        image_shape = (1,256,16,16)
        #print(self.device)
        image = torch.randn(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            #print(im1_code.shape, image.shape, im3_code.shape)
            #print(torch.cat((im1_code,image,im3_code), 1).shape)
            model_output = self.unet(torch.cat((im1_code,image,im3_code), 1), t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        # image = self.numpy_to_pil(image)

        return image



