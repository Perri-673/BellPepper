from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class BellPepperDiseaseDataset(Dataset):
    def __init__(self, root_infected, root_healthy, transform=None):
        self.root_infected = root_infected
        self.root_healthy = root_healthy
        self.transform = transform

        self.infected_images = os.listdir(root_infected)
        self.healthy_images = os.listdir(root_healthy)
        self.length_dataset = max(len(self.infected_images), len(self.healthy_images))  # e.g., 1000, 1500
        self.infected_len = len(self.infected_images)
        self.healthy_len = len(self.healthy_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        infected_img = self.infected_images[index % self.infected_len]
        healthy_img = self.healthy_images[index % self.healthy_len]

        infected_path = os.path.join(self.root_infected, infected_img)
        healthy_path = os.path.join(self.root_healthy, healthy_img)

        infected_img = np.array(Image.open(infected_path).convert("RGB"))
        healthy_img = np.array(Image.open(healthy_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=infected_img, image0=healthy_img)
            infected_img = augmentations["image"]
            healthy_img = augmentations["image0"]

        return infected_img, healthy_img
