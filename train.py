"""
Training for CycleGAN with Evaluation Metrics
"""

import torch
from dataset import BellPepperDiseaseDataset
from utils import (
    save_checkpoint,
    load_checkpoint,
    calculate_fid,
    calculate_inception_score,
    calculate_ssim,
    calculate_psnr,
    get_features,
)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.models import inception_v3
import numpy as np


def train_fn(
    disc_H,
    disc_B,
    gen_B,
    gen_H,
    loader,
    opt_disc,
    opt_gen,
    l1,
    mse,
    d_scaler,
    g_scaler,
    real_loader,
    fake_loader,
    inception_model,
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (healthy, bacterial) in enumerate(loop):
        healthy = healthy.to(config.DEVICE)
        bacterial = bacterial.to(config.DEVICE)

        # Train Discriminators H and B
        with torch.cuda.amp.autocast():
            fake_bacterial = gen_B(healthy)
            D_H_real = disc_H(bacterial)
            D_H_fake = disc_H(fake_bacterial.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_healthy = gen_H(bacterial)
            D_B_real = disc_B(healthy)
            D_B_fake = disc_B(fake_healthy.detach())
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # Combine discriminator losses
            D_loss = (D_H_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and B
        with torch.cuda.amp.autocast():
            # Adversarial loss for both generators
            D_H_fake = disc_H(fake_bacterial)
            D_B_fake = disc_B(fake_healthy)
            loss_G_B = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_H = mse(D_B_fake, torch.ones_like(D_B_fake))

            # Cycle consistency loss
            cycle_healthy = gen_H(fake_bacterial)
            cycle_bacterial = gen_B(fake_healthy)
            cycle_healthy_loss = l1(healthy, cycle_healthy)
            cycle_bacterial_loss = l1(bacterial, cycle_bacterial)

            # Total generator loss
            G_loss = (
                loss_G_B
                + loss_G_H
                + config.LAMBDA_CYCLE * (cycle_healthy_loss + cycle_bacterial_loss)
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(
            D_loss=D_loss.item(),
            G_loss=G_loss.item(),
            H_reals=H_reals / (idx + 1),
            H_fakes=H_fakes / (idx + 1),
        )

        # Save generated images for evaluation
        if idx % 200 == 0:
            save_image(fake_bacterial * 0.5 + 0.5, f"saved_images/fake_bacterial_{idx}.png")
            save_image(fake_healthy * 0.5 + 0.5, f"saved_images/fake_healthy_{idx}.png")

    # Evaluate Metrics at the end of each epoch
    print("Calculating Evaluation Metrics...")
    real_features = get_features(real_loader, inception_model, config.DEVICE)
    fake_features = get_features(fake_loader, inception_model, config.DEVICE)
    fid_score = calculate_fid(real_features, fake_features)
    inception_score = calculate_inception_score(fake_loader, inception_model, config.DEVICE)

    print(f"FID Score: {fid_score:.4f}, Inception Score: {inception_score:.4f}")

    # Calculate SSIM and PSNR for a few pairs of generated and target images
    ssim_total = 0
    psnr_total = 0
    num_samples = 0

    for real_img, fake_img in zip(real_loader, fake_loader):
        real_img = real_img[0].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        fake_img = fake_img[0].cpu().numpy().transpose(1, 2, 0)

        ssim_total += calculate_ssim(real_img, fake_img)
        psnr_total += calculate_psnr(real_img, fake_img)
        num_samples += 1

    avg_ssim = ssim_total / num_samples
    avg_psnr = psnr_total / num_samples

    print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.4f}")


def main():
    # Initialize models
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    gen_B = Generator(img_channels=3).to(config.DEVICE)
    gen_H = Generator(img_channels=3).to(config.DEVICE)

    # Optimizers
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_B.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load models if necessary
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_B, gen_B, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_B, disc_B, opt_disc, config.LEARNING_RATE)

    # Dataloader
    dataset = BellPepperDiseaseDataset(
        root_infected=config.TRAIN_BACTERIAL_DIR,
        root_healthy=config.TRAIN_HEALTHY_DIR,
        transform=config.transforms,
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    # For evaluation
    real_loader = DataLoader(
        BellPepperDiseaseDataset(config.VAL_HEALTHY_DIR, config.VAL_BACTERIAL_DIR, config.transforms),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    fake_loader = DataLoader(
        BellPepperDiseaseDataset(config.TRAIN_BACTERIAL_DIR, config.TRAIN_HEALTHY_DIR, config.transforms),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    inception_model = inception_v3(pretrained=True, transform_input=False).to(config.DEVICE)
    inception_model.fc = nn.Identity()

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_B,
            gen_B,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            torch.cuda.amp.GradScaler(),
            torch.cuda.amp.GradScaler(),
            real_loader,
            fake_loader,
            inception_model,
        )
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_GEN_B)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_CRITIC_B)


if __name__ == "__main__":
    main()
