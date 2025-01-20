import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_HEALTHY_DIR = "data/train/healthy"
TRAIN_BACTERIAL_DIR = "data/train/bacterial_spot"
VAL_HEALTHY_DIR = "data/val/healthy"
VAL_BACTERIAL_DIR = "data/val/bacterial_spot"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_HEALTHY = "gen_healthy.pth.tar"
CHECKPOINT_GEN_BACTERIAL = "gen_bacterial.pth.tar"
CHECKPOINT_CRITIC_HEALTHY = "critic_healthy.pth.tar"
CHECKPOINT_CRITIC_BACTERIAL = "critic_bacterial.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
