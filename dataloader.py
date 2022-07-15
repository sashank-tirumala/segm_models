import torch
import torchvision
import torchvision.transforms as ttf
from torch.utils.data import Dataset
import os
from PIL import Image
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SemanticDataset(Dataset):
    def __init__(self, inp_pth, outp_pth):
        self.img_names = [x for x in os.listdir(outp_path)]
        self.use_data_augmentations=False
        self.x_path = inp_pth
        self.y_path = outp_pth
        self.transforms = ttf.Compose([ttf.ToTensor()])
    def __getitem__(self, idx):
        # x = self.x_imgs[idx]
        # y = self.y_imgs[idx]
        img_name = self.img_names[idx]
        x = self.transforms(Image.open(self.x_path +"/"+ img_name.split(".")[0]+".jpg"))
        y = self.transforms(Image.open(self.y_path +"/"+ img_name))
        return x, y

    def __len__(self):
        return len(self.x_img_path)

if(__name__ == "__main__"):
    inp_path = "/root/datasets/bdd100k/images/10k/train"
    outp_path = "/datasets/bdd100k_sem/labels/sem_seg/colormaps/train"
    semd = SemanticDataset(inp_path, outp_path)
    st = time.time()
    for i in range(1000):
        x,y = semd[i]
    print(time.time() - st)
