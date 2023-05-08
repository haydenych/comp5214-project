from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PIL import Image
import os

transform = transforms.Compose(
    [
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
)

class Vimeo90k(Dataset):
    def __init__(self, path_to_dataset, datalist_filename):
        self.path_to_dataset = path_to_dataset
        self.datalist_filename = datalist_filename

        self.datalist = open(datalist_filename, "r").read().split("\n")
        if self.datalist[-1] == "":
            self.datalist.pop()


        self.transform = transform

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        im_folder_path = os.path.join(self.path_to_dataset, self.datalist[idx])

        im1 = Image.open(os.path.join(im_folder_path, "im1.png"))
        im2 = Image.open(os.path.join(im_folder_path, "im2.png"))
        im3 = Image.open(os.path.join(im_folder_path, "im3.png"))

        im1 = self.transform(im1)
        im2 = self.transform(im2)
        im3 = self.transform(im3)

        return im1