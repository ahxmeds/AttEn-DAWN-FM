#!/usr/bin/env python3
"""Compute MSE, PSNR, SSIM and Misfit for Flow‑Matching reconstructions.

The script autodetects the experiment directory purely from the dataset name
and embedding type—no need to pass an architecture flag.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Local imports ---------------------------------------------------------------
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / ".."))

from forwardProb import BlurFFT, Tomography                           
from utils import pad_zeros_at_front                                  
from config import MAIN_DIR                                        

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Flow‑Matching metrics calculator")
    p.add_argument("--dataset", choices=["MNIST", "OrganCMNIST"], default="MNIST")
    p.add_argument("--data-emb", choices=["atb", "cgls"], default="atb",
                   help="Embedding tag that appears in the experiment folder name")
    p.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    p.add_argument("--num-runs", type=int, default=32,
                   help="Number of stochastic reconstructions stored per .npy file")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Metric helpers --------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_misfit(FP, gt, recon):
    """Relative data-space error"""
    data = FP(gt.unsqueeze(0))
    recon_data = FP(recon.unsqueeze(0))
    return (data - recon_data).norm().item() / data.norm().item()


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    import torch  # defer heavy import until we know we need it
    device = torch.device("cpu" if args.device.lower() == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    

    # Default architectures per dataset (kept in sync with training/inference)   
    if args.dataset == "OrganCMNIST": 
        image_size = 64
        FP = Tomography(dim=image_size, num_angles=360, device=device)
        arch = [1, 16, 32, 64]
    else:  # For MNIST, default to Gaussian blur
        image_size = 28
        FP = BlurFFT(dim=image_size, sigma=[3, 3], device=device)
        arch = [1, 16, 32]

    exp_code = f"{args.dataset}_unetfmgattention-{'x'.join(map(str, arch))}_{FP.__class__.__name__}_{args.data_emb}"
    exp_dir = Path(MAIN_DIR) / exp_code
    gen_dir = exp_dir / "generated_images"
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    if not gen_dir.exists():
        raise FileNotFoundError(f"Generated images directory not found: {gen_dir}")

    image_files = sorted(gen_dir.glob("*.npy"))
    if not image_files:
        raise FileNotFoundError("No .npy image stacks found in " + str(gen_dir))
    print(f"Found {len(image_files)} image stacks in {gen_dir}")

    # Build forward operator --------------------------------------------------
    sample_stack = np.load(image_files[0])
    H = sample_stack.shape[1]
    if args.dataset == "OrganCMNIST":
        FP = Tomography(dim=H, num_angles=180, device=device)
    else:
        FP = BlurFFT(dim=H, sigma=[3,3], device=device)

    # Metric computation function --------------------------------------------
    def _process(path):
        stack = np.load(path)
        gt_np = stack[0]
        rng = np.ptp(gt_np)     
        
        gt = torch.tensor(gt_np, device=device)
        runs = torch.tensor(stack[1:], device=device) 
        # Per‑run metrics
        mse_r = ((runs - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        psnr_r = [psnr(gt_np, r.cpu().numpy(), data_range=rng) for r in runs]
        ssim_r = [ssim(gt_np, r.cpu().numpy(), data_range=rng) for r in runs]
        mis_r = [compute_misfit(FP, gt, runs[j]) for j in range(runs.size(0))]

        # Mean image metrics
        recon_mean = runs.mean(0)
        mse_m = float(((recon_mean - gt) ** 2).mean().cpu())
        psnr_m = psnr(gt_np, recon_mean.cpu().numpy(), data_range=rng)
        ssim_m = ssim(gt_np, recon_mean.cpu().numpy(), data_range=rng)
        mis_m = compute_misfit(FP, gt, recon_mean)

        return (path.stem, mse_r, psnr_r, ssim_r, mis_r, mse_m, psnr_m, ssim_m, mis_m)

    results = Parallel(n_jobs=-1, verbose=10)(delayed(_process)(p) for p in tqdm(image_files, desc="Computing"))

    # Assemble DataFrame ------------------------------------------------------
    R = args.num_runs
    mse_cols = [f"MSE_{pad_zeros_at_front(i,2)}" for i in range(R)]
    psnr_cols = [f"PSNR_{pad_zeros_at_front(i,2)}" for i in range(R)]
    ssim_cols = [f"SSIM_{pad_zeros_at_front(i,2)}" for i in range(R)]
    mis_cols = [f"MISFIT_{pad_zeros_at_front(i,2)}" for i in range(R)]

    df = pd.DataFrame(columns=["ImageID"] + mse_cols + psnr_cols + ssim_cols + mis_cols + ["MSE_mean", "PSNR_mean", "SSIM_mean", "MISFIT_mean"])

    for row, (img_id, mse_r, psnr_r, ssim_r, mis_r, mse_m, psnr_m, ssim_m, mis_m) in enumerate(results):
        df.loc[row, "ImageID"] = img_id
        df.loc[row, mse_cols] = mse_r
        df.loc[row, psnr_cols] = psnr_r
        df.loc[row, ssim_cols] = ssim_r
        df.loc[row, mis_cols] = mis_r
        df.loc[row, ["MSE_mean", "PSNR_mean", "SSIM_mean", "MISFIT_mean"]] = [mse_m, psnr_m, ssim_m, mis_m]

    out_csv = metrics_dir / "metrics.csv"
    df.to_csv(out_csv, index=False)
    print("Metrics saved to", out_csv)

    # Dataset‑level summary ----------------------------------------------------
    for col in ["MSE_mean", "MISFIT_mean", "SSIM_mean", "PSNR_mean"]:
        vals = df[col].astype(float).to_numpy()
        print(f"{col}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    main()
