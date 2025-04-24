#!/usr/bin/env python3
"""Inference script for Flow‑Matching UNet‑FMG.

Automatically derives the experiment folder name from the dataset, UNet
architecture, forward operator and embedding (no need to pass --experiment).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# Local repository imports
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / ".."))

from forwardProb import BlurFFT, Tomography, conjugate_gradient_least_squares  
from models import UNetFMGAttention_DE_NE, odeSol                                  
from load_datasets import (
    get_mnist_dataset,
    get_organcmnist_dataset,
    DatasetWithImageID,
) 
from utils import get_gaussian_noise_std, pad_zeros_at_front                  
from config import MAIN_DIR                                                   

# Dataset loaders ----------------------------------------------------------------
_DATASET_LOADERS = {
    "MNIST": get_mnist_dataset,
    "OrganCMNIST": get_organcmnist_dataset,
}

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Flow‑Matching generation")
    p.add_argument("--dataset", choices=_DATASET_LOADERS.keys(), default="MNIST")
    p.add_argument("--ckpt", type=int, default=None,
                   help="Epoch to load; latest if omitted")
    p.add_argument("--data-emb", choices=["atb", "cgls"], default="atb")
    p.add_argument("--device", default="0", help="CUDA idx or 'cpu'")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--noise-level", type=float, default=0.05)
    p.add_argument("--nsteps", type=int, default=100)
    p.add_argument("--num-runs", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Saving helpers ---------------------------------------------------------------

def _save(image, recon_stack, path):
    np.save(path, np.concatenate([image[None, ...], recon_stack], 0))


def save_batch(inputs, runs, save_dir, ids):
    inputs = inputs.cpu().numpy()
    runs = runs.cpu().numpy()  # (R,B,1,H,W)
    Parallel(n_jobs=-1)(
        delayed(_save)(inputs[i, 0], runs[:, i, 0], os.path.join(save_dir, f"{ids[i]}.npy"))
        for i in range(inputs.shape[0])
    )


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------

def main():
    args = parse_args()

    # device ------------------------------------------------------------------
    device = torch.device("cpu" if args.device.lower() == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")

    # Dataset, forward op, arch ----------------------------------------------
    ds = _DATASET_LOADERS[args.dataset](split="test")
    test_loader = DataLoader(DatasetWithImageID(ds, args.dataset), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=True)
    img_size = ds[0][0].shape[-1]
    if args.dataset == "OrganCMNIST":
        FP, arch = Tomography(img_size, 180, device), [1, 16, 32, 64]
    else:
        FP, arch = BlurFFT(img_size, [3, 3], device), [1, 16, 32]

    # Derive experiment code (matches training script) -----------------------
    exp_code = f"{args.dataset}_unetfmgattention-{'x'.join(map(str, arch))}_{FP.__class__.__name__}_{args.data_emb}"
    exp_dir = Path(MAIN_DIR) / exp_code
    models_dir = exp_dir / "models"
    gen_dir = exp_dir / "generated_images"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint --------------------------------------------------------------
    if args.ckpt is not None:
        ckpt_path = models_dir / f"model_ep={pad_zeros_at_front(args.ckpt,5)}.pth"
    else:
        ckpts = sorted(models_dir.glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints in " + str(models_dir))
        ckpt_path = ckpts[-1]
    print("Loading", ckpt_path)

    # Model -------------------------------------------------------------------
    model = UNetFMGAttention_DE_NE(arch=arch, dims=torch.tensor([img_size, img_size])).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Generation --------------------------------------------------------------
    for inputs, _, ids in tqdm(test_loader, desc="Generating"):
        x1 = inputs.to(device, non_blocking=True)
        B = x1.size(0)
        # noisy measurement ------------------------------------------------
        data = FP(x1)
        std = get_gaussian_noise_std(data, args.noise_level * torch.ones(B, device=device))
        noisy_data = data + std[:, None, None, None] * torch.randn_like(data)
        # embedding
        data_emb = FP.adjoint(noisy_data) if args.data_emb == "atb" else conjugate_gradient_least_squares(FP, noisy_data)
        # sample latent and integrate ------------------------------------
        x0 = torch.randn(B * args.num_runs, 1, img_size, img_size, device=device)
        traj = odeSol(x0, data_emb.repeat(args.num_runs, 1, 1, 1), std.repeat(args.num_runs),
                      model, nsteps=args.nsteps)
        xf_runs = traj[-1].view(args.num_runs, B, 1, img_size, img_size)
        save_batch(inputs, xf_runs, gen_dir, ids)

    print("Done. Outputs stored in", gen_dir)


if __name__ == "__main__":
    main()
