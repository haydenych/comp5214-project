from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np

def image_ten2pil(image):
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
    return reverse_transforms(image)