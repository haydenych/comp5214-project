import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = OmegaConf.load("config/model.yaml")
model = vqgan.VQModel(**config.model.params)
model.init_from_ckpt("ckpts/model.ckpt")
model.to(device)


train_data = Vimeo90k(path_to_dataset='../data/vimeo_triplet/sequences', datalist_filename='../data/vimeo_triplet/tri_trainlist.txt')
train_loader = DataLoader(dataset = train_data, batch_size=4, shuffle=True)

batch =  next(iter(train_loader))
im0 = batch[:,0].to(device)
im1 = batch[:,1].to(device)
im2 = batch[:,2].to(device)
# plt.imshow(image_ten2pil(img1))
# plt.show()

#x = train_data[5000][0].unsqueeze(0).to(device)
im0_code, _,_ = model.encode(im0)
im1_code, _,_ = model.encode(im1)
im1_code = torch.randn_like(im1_code).to(device)
im2_code, _,_ = model.encode(im2)

out1 = model.decode(im0_code).detach().cpu()
out2 = model.decode(im1_code).detach().cpu()
out3 = model.decode(im2_code).detach().cpu()

fig = plt.figure(figsize=(10, 7))
img_num = 2
for i in range(img_num):
    fig.add_subplot(img_num, 3, 1+i*3)
    plt.imshow(image_ten2pil(out1[i]))
    fig.add_subplot(img_num, 3, 2+i*3)
    plt.imshow(image_ten2pil(out2[i]))
    fig.add_subplot(img_num, 3, 3+i*3)
    plt.imshow(image_ten2pil(out3[i]))
plt.savefig("test2.png") 