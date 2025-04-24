#%%
import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve
import torchvision
from torch.utils.data.dataloader import DataLoader

#%%
# class for blurring forward problem.
# For blurring, the forward problem A = A^T
# Hence, the ``space'' of (clean) image and blurred data are the same
class BlurFFT(nn.Module):
    def __init__(self, dim=256, sigma=[3,3], device='cuda'):
        super(BlurFFT, self).__init__()
        self.dim = dim
        self.device = device
        self.sigma = sigma 

        P, center = self.psfGauss(self.dim)
        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0,1])).unsqueeze(0).unsqueeze(0)
        self.S = S.to(self.device)

    def forward(self, I):
        B = torch.real(torch.fft.ifft2(self.S * torch.fft.fft2(I))) 
        return B

    def adjoint(self, Ic):
        I = self.forward(Ic)
        return I
        
    def psfGauss(self, dim):
        s = self.sigma
        m = dim
        n = dim
        
        x = torch.arange(-n//2+1,n//2+1)
        y = torch.arange(-n//2+1,n//2+1)
        X,Y = torch.meshgrid(x,y,indexing='ij')

        PSF = torch.exp( -(X**2)/(2*s[0]**2) - (Y**2)/(2*s[1]**2))
        PSF = PSF / torch.sum(PSF)

        # Get center ready for output.
        center = [1-m//2, 1-n//2]

        return PSF, center


#%%
# class for tomography forward problem.
# For tomography, the forward problem A != A^T
# Hence, the ``space'' of image and data (sinogram) are NOT the same
# sinogram space is much smaller in size as compared to the image space
class Tomography(nn.Module):
    def __init__(self, dim=28, num_angles=180, device='cuda'):
        super(Tomography, self).__init__()
        self.dim = dim
        self.num_angles = num_angles
        self.device = device
        self.pad_size = self.dim // 2
        self.num_detectors = self.dim + 2*self.pad_size + 1
        # Create grid for image pixel coordinates (centered at 0)
        X, Y = torch.meshgrid(torch.arange(self.dim+2*self.pad_size, device=self.device) - self.dim, 
                              torch.arange(self.dim+2*self.pad_size, device=self.device) - self.dim)
        self.X = X.float()
        self.Y = Y.float()

        self.A = self.compute_tomography_matrix()
        self.A = self.A.to(self.device)

    def compute_tomography_matrix(self):
        A_rows = []
        theta = torch.linspace(0, 2*np.pi, self.num_angles, device=self.device)  # Angles in radians

        for i, angle in enumerate(theta):
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)

            for detector in range(self.num_detectors):
                ray_row = self.compute_ray_row(detector, cos_angle, sin_angle)
                A_rows.append(ray_row.flatten())  # Flatten and append the row

        A = torch.stack(A_rows, dim=0)
        return A

    def compute_ray_row(self, detector_idx, cos_angle, sin_angle):
        """
        Compute the interactions of a ray with all pixels in the image grid.
        This returns a vector representing the weights of the ray's interactions
        with each pixel in the image (delta function approximation).
        """
        t_vals = torch.linspace(-self.dim, self.dim, steps=self.num_detectors, device=self.device)
        X_rot = self.X * cos_angle + self.Y * sin_angle  # Rotate pixel grid
        ray_row = (X_rot - t_vals[detector_idx]).abs() < 0.5  # Delta approximation: 1 if the ray intersects, 0 otherwise
        return ray_row.float()
    
    def hamming_filter(self, sinogram):
        sinogram_fft = torch.fft.fft(sinogram, dim=3)

        # Create Hamming filter
        freqs = torch.fft.fftfreq(self.num_detectors, device=self.device)
        hamming_window = 0.54 + 0.46 * torch.cos(2 * np.pi * freqs)
        hamming_window = hamming_window.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape for broadcasting

        filtered_sinogram_fft = sinogram_fft * hamming_window

        filtered_sinogram = torch.real(torch.fft.ifft(filtered_sinogram_fft, dim=2))
        return filtered_sinogram
    
    
    def forward(self, I):
        batch_size = I.shape[0]
        I_pad = nn.functional.pad(I, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='constant', value=0)
        I_flatten = I_pad.view(batch_size, -1)  # Flatten the image for matrix multiplication
        sinogram = torch.matmul(I_flatten, self.A.T)  # Apply forward projection (Radon transform)
        sinogram = sinogram.view(batch_size, 1, self.num_angles, self.num_detectors)  # Reshape to sinogram form
        return sinogram

    def adjoint(self, sinogram, apply_hamming_filter=False):
        # Optionally apply the Hamming filter
        if apply_hamming_filter:
            sinogram = self.hamming_filter(sinogram)

        batch_size = sinogram.shape[0]
        sinogram_flatten = sinogram.reshape(batch_size, sinogram.shape[1]*sinogram.shape[2]*sinogram.shape[3])
        backprojected_image = torch.matmul(sinogram_flatten, self.A)  # Apply adjoint (backprojection)
        backprojected_image = backprojected_image.view(batch_size, 1, self.dim+2*self.pad_size, self.dim+2*self.pad_size)  # Reshape to image form
        return backprojected_image[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

#%%
def conjugate_gradient_least_squares(A, b, x=None, tol=1e-2, max_iter=100):
    if x is not None:
        r = b - A(x)
        s = A.adjoint(r)
    else:
        s = A.adjoint(b)
        x = torch.zeros_like(s)
        r = b.clone() 
    
    p = s.clone()
    norm_b = torch.norm(b)
    norm_r = torch.norm(r)
    iteration = 0

    while norm_r / norm_b > tol and iteration < max_iter:
        iteration += 1
        Ap = A(p)
        alpha = (s*s).mean() / (Ap*Ap).mean()
        x = x + alpha * p
        r = r - alpha * Ap
        s_new = A.adjoint(r)
        beta = (s_new*s_new).mean() / (s*s).mean()
        p = s_new + beta * p
        s = s_new
        norm_r = torch.norm(r)/torch.norm(b)
        # print(f"Iteration {iteration}: Residual norm = {norm_r}")

    return x

#%%
def wiener_deconvolution(I_blurred, H, K=0.01):
    # Fourier transform of the blurred image
    G = torch.fft.fft2(I_blurred)

    # Wiener filter
    H_conj = torch.conj(H)
    H_abs2 = torch.abs(H)**2
    W_filter = H_conj / (H_abs2 + K)

    # Apply Wiener filter in frequency domain
    F_hat = W_filter * G

    # Inverse Fourier transform to get the deblurred image
    I_deblurred = torch.real(torch.fft.ifft2(F_hat))

    return I_deblurred

#%%
if __name__ == '__main__':
    idx = 0
    path = images_fpaths[0]
    d = np.load(path)
    fig, ax = plt.subplots(1,5)
    image = torch.tensor(d[0]).permute(2,0,1).unsqueeze(dim=0).to("cuda")
    data = torch.tensor(d[1]).permute(2,0,1).unsqueeze(dim=0).to("cuda")
    FP = BlurFFT(96)
    image_deblur_wiener = wiener_deconvolution(data.squeeze(), FP.S.squeeze(), K=0.01) 
    image_deblur_cgls = conjugate_gradient_least_squares(FP, data, tol=1e-6)
    ax[0].imshow(d[0])
    ax[1].imshow(d[1])
    ax[2].imshow(image_deblur_wiener.permute(1,2,0).detach().cpu().numpy())
    ax[3].imshow(image_deblur_cgls[0].permute(1,2,0).detach().cpu().numpy())
    ax[4].imshow(d[2])