import argparse
import time
import datetime
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from saved_models.WGAN.models import weights_init_normal, GeneratorUNet, Discriminator
from viewdata.data import get_data_loader
from utils import compute_gradient_penalty, sample_images, evaluate_generated_signal_quality

###############################################################################
# CPU-FRIENDLY SETTINGS
###############################################################################
# Force CPU
device = torch.device("cpu")
torch.set_num_threads(4)          # use only 4 CPU threads
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="cpu_run")
parser.add_argument("--dataset_prefix", type=str, required=True)
parser.add_argument("--n_epochs", type=int, default=200)       # smaller default
parser.add_argument("--batch_size", type=int, default=16)      # smaller default
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--lambda_gp", type=float, default=10)
parser.add_argument("--ncritic", type=int, default=1)          # fewer D steps
parser.add_argument("--checkpoint_interval", type=int, default=50)
parser.add_argument("--shuffle_training", type=bool, default=True)
parser.add_argument("--shuffle_testing", type=bool, default=False)

args, _ = parser.parse_known_args()
print("\nUsing CPU-optimized settings:", args, "\n")

# Load data — disable multiprocessing for macOS
dataloader, val_dataloader = get_data_loader(
    args.dataset_prefix,
    args.batch_size,
    from_ppg=True,
    shuffle_training=args.shuffle_training,
    shuffle_testing=args.shuffle_testing,
)

# Models
generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

criterion_L2 = torch.nn.MSELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Logging
os.makedirs(f"logs/{args.experiment_name}", exist_ok=True)
writer = SummaryWriter(f"logs/{args.experiment_name}")

os.makedirs(f"saved_models/{args.experiment_name}", exist_ok=True)
os.makedirs(f"sample_signals/{args.experiment_name}", exist_ok=True)

###############################################################################
# TRAINING
###############################################################################

prev_time = time.time()

for epoch in range(1, args.n_epochs + 1):

    generator.train()
    discriminator.train()

    for i, batch in enumerate(dataloader):

        real_A = batch[0].to(device)
        real_B = batch[1].to(device)

        ###############################################
        # Train Generator
        ###############################################
        if i % args.ncritic == 0:

            for p in generator.parameters():
                p.grad = None

            fake_B = generator(real_A)
            loss_L2 = criterion_L2(fake_B, real_B)

            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = -torch.mean(pred_fake)

            loss_G = loss_GAN + 50 * loss_L2
            loss_G.backward()
            optimizer_G.step()

        ###############################################
        # Train Discriminator
        ###############################################
        for p in discriminator.parameters():
            p.grad = None

        fake_B = generator(real_A).detach()

        real_valid = discriminator(real_B, real_A)
        fake_valid = discriminator(fake_B, real_A)

        gp = compute_gradient_penalty(
            discriminator, real_B, fake_B, real_A, (1, 9), device
        )

        loss_D = -torch.mean(real_valid) + torch.mean(fake_valid) + args.lambda_gp * gp
        loss_D.backward()
        optimizer_D.step()

        ###############################################
        # Print progress (lightweight)
        ###############################################
        batches_done = (epoch - 1) * len(dataloader) + i
        batches_left = args.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            f"\r[Epoch {epoch}/{args.n_epochs}] "
            f"[Batch {i}/{len(dataloader)}] "
            f"[D loss: {loss_D.item():.4f}] "
            f"[G loss: {loss_G.item():.4f}] ETA: {time_left}"
        )

        # Log less frequently to speed up training
        if i % 10 == 0:
            writer.add_scalar("g_loss", loss_G.item(), batches_done)
            writer.add_scalar("d_loss", loss_D.item(), batches_done)

    ###############################################
    # Save checkpoint occasionally
    ###############################################
    if epoch % args.checkpoint_interval == 0:
        torch.save(
            {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            },
            f"saved_models/{args.experiment_name}/model_{epoch}.pth"
        )
        sample_images(args.experiment_name, val_dataloader, generator, epoch, device)

print("\nTraining complete.")
