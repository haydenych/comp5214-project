# COMP5214 Project

## Data
For this project, we will be using the VIMEO-90k triplet dataset.

Download data [here](http://toflow.csail.mit.edu/) or using the following command.
```
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
```

Unzip and place all data in the folder `data`. The folder structure should be as follows:
```
data/vimeo_triplet/sequences/...
data/vimeo_triplet/readme.txt
data/vimeo_triplet/tri_testlist.txt
data/vimeo_triplet/tri_trainlist.txt
```

## Finetuning VQGAN
Clone this [repo](https://github.com/CompVis/taming-transformers) into the folder `code`
```
cd code && git clone https://github.com/CompVis/taming-transformers
```

Please also download the pre-trained weights and config files from the above repo. They are listed in the section - Overview of pretrained models.

In the file `code/train-vqgan.py`, specify the gpu you would like to use. Below shows all command used in fine-tuning:

```
python3 train-vqgan.py --batch-size=2 --config-path=../configs/model_f16_16384.yaml --ckpt-path=../checkpoints/pre-trained/last_f16_16384.ckpt

python3 train-vqgan.py --batch-size=2 --config-path=../configs/model_f16_16384.yaml --ckpt-path=../checkpoints/pre-trained/last_f16_16384.ckpt --learning-rate=5e-6

python3 train-vqgan.py --batch-size=2 --config-path=../configs/model_f16_16384.yaml --ckpt-path=../checkpoints/pre-trained/last_f16_16384.ckpt --learning-rate=1e-8

python3 train-vqgan.py --batch-size=2 --config-path=../configs/model_f8_256.yaml --ckpt-path=../checkpoints/pre-trained/last_f8_256.ckpt

python3 train-vqgan.py --batch-size=2 --config-path=../configs/model_f8_16384.yaml --ckpt-path=../checkpoints/pre-trained/last_f8_16384.ckpt
```

## Running FID
In the file `code/fid.py`, specify the gpu you would like to use. Below shows all command used:

```
python3 fid.py --vqgan-config-path=../configs/model_f16_16384.yaml --vqgan-ckpt-path=../checkpoints/f16-16384\ lr=1e-7/vqgan_epoch_9.pt

python3 fid.py --vqgan-config-path=../configs/model_f8_256.yaml --vqgan-ckpt-path=../checkpoints/f8-256\ lr=1e-7/vqgan_epoch_9.pt

python3 fid.py --vqgan-config-path=../configs/model_f8_16384.yaml --vqgan-ckpt-path=../checkpoints/f8-16384\ lr=1e-7/vqgan_epoch_9.pt

python3 fid.py --vqgan-config-path=../configs/model_f16_16384.yaml --vqgan-ckpt-path=../checkpoints/f16-16384\ lr=1e-7/vqgan_epoch_9.pt --datalist-path=../data/vimeo_triplet/tri_trainlist.txt

python3 fid.py --vqgan-config-path=../configs/model_f8_256.yaml --vqgan-ckpt-path=../checkpoints/f8-256\ lr=1e-7/vqgan_epoch_9.pt --datalist-path=../data/vimeo_triplet/tri_trainlist.txt

python3 fid.py --vqgan-config-path=../configs/model_f8_16384.yaml --vqgan-ckpt-path=../checkpoints/f8-16384\ lr=1e-7/vqgan_epoch_9.pt --datalist-path=../data/vimeo_triplet/tri_trainlist.txt
```