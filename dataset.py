from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class BellPepperDataset(Dataset):
    def __init__(self, root_healthy, root_bacterial, transform=None):
        self.root_healthy = root_healthy
        self.root_bacterial = root_bacterial
        self.transform = transform

        self.healthy_images = os.listdir(root_healthy)
        self.bacterial_images = os.listdir(root_bacterial)
        self.length_dataset = max(len(self.healthy_images), len(self.bacterial_images))
        self.healthy_len = len(self.healthy_images)
        self.bacterial_len = len(self.bacterial_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        healthy_img = self.healthy_images[index % self.healthy_len]
        bacterial_img = self.bacterial_images[index % self.bacterial_len]

        healthy_path = os.path.join(self.root_healthy, healthy_img)
        bacterial_path = os.path.join(self.root_bacterial, bacterial_img)

        healthy_img = np.array(Image.open(healthy_path).convert("RGB"))
        bacterial_img = np.array(Image.open(bacterial_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=healthy_img, image0=bacterial_img)
            healthy_img = augmentations["image"]
            bacterial_img = augmentations["image0"]

        return healthy_img, bacterial_img
