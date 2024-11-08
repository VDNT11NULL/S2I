import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import save_some_examples
from UvU_Net_Generator import OuterUNet as Generator
from UvU_Discriminator import Discriminator
import config
from matplotlib import pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

        assert len(self.input_images) == len(self.target_images), "Mismatch between input and target images!"

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])

        input_image = Image.open(input_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

def calculate_gradient_penalty(discriminator, real_images, fake_images):
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(DEVICE)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    interpolated_images.requires_grad_(True)
    D_interpolated = discriminator(interpolated_images, interpolated_images)
    gradients = torch.autograd.grad(
        outputs=D_interpolated,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(D_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, lambda_gp
):
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            gradient_penalty = calculate_gradient_penalty(disc, y, y_fake)
            D_loss = (D_real_loss + D_fake_loss + lambda_gp * gradient_penalty) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss=D_loss.item()
            )

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    l1_lambda = trial.suggest_int('l1_lambda', 10, 200)
    lambda_gp = trial.suggest_int('lambda_gp', 1, 10)
    num_epochs = 500

    input_dir = r"C:\Users\user\S2I\Sample_dataset\sobel_images1"
    target_dir = r"C:\Users\user\S2I\Sample_dataset\input_images"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = PairedImageDataset(input_dir=input_dir, target_dir=target_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, lambda_gp
        )

    total_loss = 0
    gen.eval()
    with torch.no_grad():
        for input_image, target_image in val_loader:
            input_image, target_image = input_image.to(DEVICE), target_image.to(DEVICE)
            y_fake = gen(input_image)
            loss = L1_LOSS(y_fake, target_image) * l1_lambda
            total_loss += loss.item()

    return total_loss / len(val_loader)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    losses = []  # Store losses for each trial

    def callback(study, trial):
        losses.append(trial.value)  # Store the loss

    study.optimize(objective, n_trials=20, callbacks=[callback])

    print("Best Hyperparameters: ", study.best_params)

    # Plot the loss graph
    plt.plot(losses, marker='o')
    plt.title("Loss vs. Trials")
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
