#!/usr/bin/env python3
"""Flow‑Matching training with dataset‑dependent forward problems.

Usage examples
--------------
MNIST (Forward problem: Gaussian blur):
$ python train.py --dataset MNIST --data-emb atb

OrganCMNIST (Forward problem: Tomographic Radon transform):
$ python train.py --dataset OrganCMNIST --data-emb cgls 
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Local imports (repository modules)
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / ".."))

from forwardProb import BlurFFT, Tomography, conjugate_gradient_least_squares 
from models import UNetFMG_DE_NE, UNetFMGAttention_DE_NE                     
from load_datasets import (
    get_mnist_dataset,
    get_organcmnist_dataset,
)  
from utils import (
    get_gaussian_noise_std,
    sample_conditional_pt,
    pad_zeros_at_front,
)  
from config import MAIN_DIR                                                 

# Mapping dataset name -> loader callable
_DATASET_LOADERS = {
    "MNIST": get_mnist_dataset,
    "OrganCMNIST": get_organcmnist_dataset,
}

# -----------------------------------------------------------------------------
# Cmd‑line args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Flow‑Matching UNet‑FMG-Attention")
    p.add_argument("--dataset", choices=_DATASET_LOADERS.keys(), default="MNIST")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-emb", choices=["atb", "cgls"], default="cgls")
    p.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--max-noise", type=float, default=0.20)
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---------------- Device --------------------------------------------------
    if args.device.lower() == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    print(f"Using device: {device}")

    # ---------------- Dataset -------------------------------------------------
    dataset_loader = _DATASET_LOADERS[args.dataset]
    dataset = dataset_loader(split="train")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    image_size = dataset[0][0].shape[-1]

    # ---------------- Forward problem selection ------------------------------
    if args.dataset == "OrganCMNIST": 
        FP = Tomography(dim=image_size, num_angles=180, device=device)
        arch = [1, 16, 32, 64]
    else:  # For MNIST, default to Gaussian blur
        FP = BlurFFT(dim=image_size, sigma=[3, 3], device=device)
        arch = [1, 16, 32]

    # ---------------- Model ---------------------------------------------------
    model = UNetFMGAttention_DE_NE(arch=arch, dims=torch.tensor([image_size, image_size]))
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable model parameters: {n_params/1e6:.2f} M")

    # ---------------- Optimiser & scheduler ----------------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=0)
    scaler = torch.amp.GradScaler()

    # ---------------- I/O -----------------------------------------------------
    experiment_code = f"{args.dataset}_unetfmgattention-{'x'.join(map(str, arch))}_{FP.__class__.__name__}_{args.data_emb}"
    results_dir = Path(MAIN_DIR) / experiment_code
    logs_dir, models_dir = results_dir / "logs", results_dir / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    train_logs_fpath = logs_dir / "trainlogs.csv"

    # ---------------- Train ---------------------------------------------------
    loss_log, lossU_log, lossD_log = [], [], []
    start_time = time.time()
    
    tqdm_epoch = tqdm(range(args.max_epochs), desc="Training")
    for epoch in tqdm_epoch:
        model.train()
        run_L = run_Lu = run_Ld = 0.0

        for inputs, _ in train_loader:
            x1 = inputs.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)

            # Antithetic sampling
            x1 = torch.cat((x1, x1), dim=0)
            x0 = torch.cat((x0, -x0), dim=0)

            # Generate noisy data
            p_noise = args.max_noise * torch.rand(x0.shape[0], device=device)
            data = FP(x1)
            std = get_gaussian_noise_std(data, p_noise)
            data = data + std[:, None, None, None] * torch.randn_like(data)

            # Data embedding
            if args.data_emb == "atb":
                data_emb = FP.adjoint(data)
            elif args.data_emb == "cgls":
                data_emb = conjugate_gradient_least_squares(FP, data)
            else:
                print('Invalid data embedding! The allowed choices are `atb` and `cgls`.')
                return 

            # Flow‑matching samples
            t = torch.rand(x0.shape[0], device=device)
            xt, ut = sample_conditional_pt(x0, x1, t, sigma=0.01)

            # Forward / loss
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                vt = model(xt, t, data_emb, std)
                loss_u = torch.mean((vt - ut) ** 2) / torch.mean(ut ** 2)
                x1_hat = xt + (1 - t.view(-1, 1, 1, 1)) * vt
                loss_d = torch.mean((FP(x1_hat) - data) ** 2) / torch.mean(data ** 2)
                loss = loss_u + loss_d
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            run_L += loss.item() * inputs.size(0)
            run_Lu += loss_u.item() * inputs.size(0)
            run_Ld += loss_d.item() * inputs.size(0)
            tqdm_epoch.set_description(f'epochs = {epoch + 1}, Loss = {loss:3.6e}, LossU =  {loss_u:3.6e}, LossD =  {loss_d:3.6e}')

        # ---- epoch summary ----
        N = len(train_loader.dataset)
        L, Lu, Ld = run_L / N, run_Lu / N, run_Ld / N
        loss_log.append(L); lossU_log.append(Lu); lossD_log.append(Ld)
        pd.DataFrame({"Loss": loss_log, "LossU": lossU_log, "LossD": lossD_log}).to_csv(train_logs_fpath, index=False)

        scheduler.step()
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), models_dir / f"model_ep={pad_zeros_at_front(epoch+1,5)}.pth")

    # ------------------------------------------------------------------------
    print("Training complete!  Elapsed: {:.2f} h".format((time.time()-start_time)/3600))


if __name__ == "__main__":
    main()
