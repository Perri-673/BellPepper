import random
import torch
import os
import numpy as np
import config
import copy
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models, transforms

#from inception_v3 import InceptionV3


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Saves the model and optimizer state dicts to a checkpoint file.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Loads a checkpoint into the model and optimizer.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, lr_scheduler_type='StepLR', step_size=10, gamma=0.1):
    """
    Returns the learning rate scheduler based on the provided type.
    """
    if lr_scheduler_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    else:
        raise ValueError(f"Unknown scheduler type: {lr_scheduler_type}")


def get_device():
    """
    Returns the appropriate device ('cuda' if GPU is available, else 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Perfect image match
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))


# Load the InceptionV3 model
def get_inception_model():
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.eval()
    return model

def calculate_fid(real_images, generated_images, device='cuda'):
    # Prepare the InceptionV3 model
    model = get_inception_model().to(device)
    
    # Get features for real and generated images
    real_features = extract_features(real_images, model, device)
    fake_features = extract_features(generated_images, model, device)

    # Calculate the mean and covariance of real and generated features
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_fake, sigma_fake = calculate_statistics(fake_features)

    # Compute the FID score
    fid_score = calculate_fid_score(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score

def extract_features(images, model, device):
    # Preprocess images before passing them through InceptionV3
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = [preprocess(img).unsqueeze(0).to(device) for img in images]  # Convert list of images to tensor
    features = []
    
    with torch.no_grad():
        for img in images:
            # Get the features (the output of the InceptionV3's penultimate layer)
            feature = model(img)
            features.append(feature.squeeze().cpu().numpy())
    
    return np.array(features)

def calculate_statistics(features):
    # Compute the mean and covariance of the features
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mean, cov

def calculate_fid_score(mu_real, sigma_real, mu_fake, sigma_fake):
    # Compute the FID score from the means and covariances
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid_score

def calculate_inception_score(images, device='cuda', splits=10):
    # Prepare the InceptionV3 model
    model = get_inception_model().to(device)

    # Preprocess the images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = [preprocess(img).unsqueeze(0).to(device) for img in images]

    # Get predictions for each image
    predictions = []
    with torch.no_grad():
        for img in images:
            output = model(img)
            predictions.append(output.squeeze().cpu().numpy())

    predictions = np.array(predictions)

    # Calculate inception score
    scores = []
    for i in range(splits):
        part = predictions[i * len(predictions) // splits: (i + 1) * len(predictions) // splits]
        kl_divergence = compute_kl_divergence(part)
        scores.append(np.exp(kl_divergence))

    return np.mean(scores), np.std(scores)

def compute_kl_divergence(predictions):
    # Compute the Kullback-Leibler (KL) divergence between the conditional probabilities of the model
    p_y = np.mean(predictions, axis=0)
    kl_divergence = np.mean([entropy(pred, p_y) for pred in predictions])
    return kl_divergence

# Function to extract features from the InceptionV3 model
def get_features(images, device='cuda'):
    # Prepare the InceptionV3 model
    model = get_inception_model().to(device)

    # Preprocess the images for InceptionV3
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = [preprocess(img).unsqueeze(0).to(device) for img in images]
    
    # Get features from the Inception model (up to the last fully connected layer)
    features = []
    with torch.no_grad():
        for img in images:
            # We forward pass through the model up to the last convolutional layer
            x = model.Conv2d_1a_3x3(img)  # Pass through first layer
            x = model.Conv2d_2a_3x3(x)  # Pass through second layer
            x = model.Conv2d_2b_3x3(x)  # and so on...
            x = model.maxpool1(x)  # Max-pooling layer

            x = model.Mixed_5b(x)  # First mixed layer
            x = model.Mixed_5c(x)  # and so on...

            # Extract features from the last layer
            x = model.Mixed_7c(x)  # The last convolutional block
            features.append(x.squeeze().cpu().numpy())

    return np.array(features)