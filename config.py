import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEALTHY_PLANT_DIR = "data/healthy_plant"
INFECTED_PLANT_DIR = "data/infected_plant"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_HEALTHY = "gen_healthy.pth.tar"
CHECKPOINT_GEN_INFECTED = "gen_infected.pth.tar"
CHECKPOINT_CRITIC_HEALTHY = "critic_healthy.pth.tar"
CHECKPOINT_CRITIC_INFECTED = "critic_infected.pth.tar"
# Directory for training healthy plant images
TRAIN_HEALTHY_DIR = "data/train/healthy"

# Directory for training bacterial spot plant images
TRAIN_BACTERIAL_DIR = "data/train/bacterial_spot"

# Directory for validation healthy plant images
VAL_HEALTHY_DIR = "data/val/healthy"

# Directory for validation bacterial spot plant images
VAL_BACTERIAL_DIR = "data/val/bacterial_spot"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
