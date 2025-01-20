import torch
from dataset import BellPepperDiseaseDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
import torch.nn.functional as F
from utils import calculate_fid, calculate_inception_score, get_features

def train_fn(
    disc_healthy, disc_bacterial, gen_bacterial, gen_healthy, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    healthy_reals = 0
    healthy_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (healthy, bacterial) in enumerate(loop):
        healthy = healthy.to(config.DEVICE)
        bacterial = bacterial.to(config.DEVICE)

        # Train Discriminators Healthy and Bacterial
        with torch.cuda.amp.autocast():
            fake_bacterial = gen_bacterial(healthy)
            D_healthy_real = disc_healthy(bacterial)
            D_healthy_fake = disc_healthy(fake_bacterial.detach())
            healthy_reals += D_healthy_real.mean().item()
            healthy_fakes += D_healthy_fake.mean().item()
            D_healthy_real_loss = mse(D_healthy_real, torch.ones_like(D_healthy_real))
            D_healthy_fake_loss = mse(D_healthy_fake, torch.zeros_like(D_healthy_fake))
            D_healthy_loss = D_healthy_real_loss + D_healthy_fake_loss

            fake_healthy = gen_healthy(bacterial)
            D_bacterial_real = disc_bacterial(healthy)
            D_bacterial_fake = disc_bacterial(fake_healthy.detach())
            D_bacterial_real_loss = mse(D_bacterial_real, torch.ones_like(D_bacterial_real))
            D_bacterial_fake_loss = mse(D_bacterial_fake, torch.zeros_like(D_bacterial_fake))
            D_bacterial_loss = D_bacterial_real_loss + D_bacterial_fake_loss

            # Total Discriminator Loss
            D_loss = (D_healthy_loss + D_bacterial_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators Healthy and Bacterial
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_healthy_fake = disc_healthy(fake_bacterial)
            D_bacterial_fake = disc_bacterial(fake_healthy)
            loss_G_healthy = mse(D_healthy_fake, torch.ones_like(D_healthy_fake))
            loss_G_bacterial = mse(D_bacterial_fake, torch.ones_like(D_bacterial_fake))

            # Cycle Loss
            cycle_healthy = gen_healthy(fake_bacterial)
            cycle_bacterial = gen_bacterial(fake_healthy)
            cycle_healthy_loss = l1(healthy, cycle_healthy)
            cycle_bacterial_loss = l1(bacterial, cycle_bacterial)

            # Identity Loss (remove these for efficiency if you set lambda_identity=0)
            identity_healthy = gen_healthy(healthy)
            identity_bacterial = gen_bacterial(bacterial)
            identity_healthy_loss = l1(healthy, identity_healthy)
            identity_bacterial_loss = l1(bacterial, identity_bacterial)

            # Total Generator Loss
            G_loss = (
                loss_G_healthy
                + loss_G_bacterial
                + cycle_healthy_loss * config.LAMBDA_CYCLE
                + cycle_bacterial_loss * config.LAMBDA_CYCLE
                + identity_healthy_loss * config.LAMBDA_IDENTITY
                + identity_bacterial_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Save images periodically
        if idx % 200 == 0:
            save_image(bacterial * 0.5 + 0.5, f"saved_images/real_bacterial_{idx}.png")
            save_image(fake_bacterial * 0.5 + 0.5, f"saved_images/healthy_to_bacterial_{idx}.png")
            save_image(healthy * 0.5 + 0.5, f"saved_images/real_healthy_{idx}.png")
            save_image(fake_healthy * 0.5 + 0.5, f"saved_images/bacterial_to_healthy_{idx}.png")

        loop.set_postfix(healthy_real=healthy_reals / (idx + 1), healthy_fake=healthy_fakes / (idx + 1))


def main():
    # Initialize Discriminators and Generators for both healthy and bacterial
    disc_healthy = Discriminator(in_channels=3).to(config.DEVICE)
    disc_bacterial = Discriminator(in_channels=3).to(config.DEVICE)
    gen_bacterial = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_healthy = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    # Optimizers for Discriminators and Generators
    opt_disc = optim.Adam(
        list(disc_healthy.parameters()) + list(disc_bacterial.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_bacterial.parameters()) + list(gen_healthy.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Loss functions
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    # Prepare DataLoader
    train_dataset = BellPepperDiseaseDataset(config.TRAIN_HEALTHY_DIR, config.TRAIN_BACTERIAL_DIR, transform=config.transforms)
    val_dataset = BellPepperDiseaseDataset(config.VAL_HEALTHY_DIR, config.VAL_BACTERIAL_DIR, transform=config.transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Scalers for mixed precision training
    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_healthy,
            disc_bacterial,
            gen_bacterial,
            gen_healthy,
            train_loader,
            opt_disc,
            opt_gen,
            l1,
            mse,
            d_scaler,
            g_scaler,
        )

        # Save checkpoints periodically
        if config.SAVE_MODEL:
            save_checkpoint(gen_healthy, opt_gen, filename=config.CHECKPOINT_GEN_HEALTHY)
            save_checkpoint(gen_bacterial, opt_gen, filename=config.CHECKPOINT_GEN_BACTERIAL)
            save_checkpoint(disc_healthy, opt_disc, filename=config.CHECKPOINT_CRITIC_HEALTHY)
            save_checkpoint(disc_bacterial, opt_disc, filename=config.CHECKPOINT_CRITIC_BACTERIAL)


if __name__ == "__main__":
    main()
