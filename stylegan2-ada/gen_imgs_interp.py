import os
import pickle
import numpy as np
import torch
import PIL.Image


model_path = './training-runs/00001-LSK-auto4-kimg2000/network-snapshot-001000.pkl'
outdir = './outs/fake_imgs_interp'
os.makedirs(outdir, exist_ok=True)

with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

n_imgs = 10
imgs = np.zeros((n_imgs+2, 1024, 1024), dtype=np.uint8)
np.random.seed(200)
z1 = np.random.randn(1, G.z_dim)
z2 = z1 + 2.0*np.random.randn(1, G.z_dim)
z1 = torch.Tensor(z1).to('cuda')           # latent codes
z2 = torch.Tensor(z2).to('cuda')           # latent codes
c = None                      # class labels (not used in this example)
w1 = G.mapping(z1, c)
w2 = G.mapping(z2, c)
img = G.synthesis(w1, noise_mode='const')      # NCHW, float32, dynamic range [-1, +1]
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img = PIL.Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/0.png')

img = G.synthesis(w2, noise_mode='const')      # NCHW, float32, dynamic range [-1, +1]
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img = PIL.Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/11.png')

# w = w2
for i in range(n_imgs):
    print(i)
    frac = i / float(n_imgs)
    w = frac*w2 + (1-frac)*w1
    img = G.synthesis(w, noise_mode='const')      # NCHW, float32, dynamic range [-1, +1]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/{i+1:06d}.png')

