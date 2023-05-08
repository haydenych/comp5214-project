import os
import sys
sys.path.insert(1, './taming-transformers')

from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.models import vqgan
import taming.models

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from dataset.Vimeo90k import Vimeo90k

import matplotlib.pyplot as plt
import numpy as np
from utils import *

config = OmegaConf.load("config/model.yaml")
model = vqgan.VQModel(**config.model.params)
model.init_from_ckpt("ckpts/last.ckpt")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_trainlist.txt')
train_loader = DataLoader(dataset = train_data, batch_size=4, shuffle=True)

x = train_data[5000].unsqueeze(0)

z, emb_loss, info = model.encode(x)
out = model.decode(z).detach()
fig = plt.figure(figsize=(10, 7))
fig.add_subplot(1, 2, 1)
plt.imshow(image_ten2pil(x))
fig.add_subplot(1, 2, 2)
plt.imshow(image_ten2pil(out))
plt.show()