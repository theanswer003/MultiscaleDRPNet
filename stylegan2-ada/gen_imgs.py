import os
import pickle
import numpy as np
import torch
import PIL.Image


model_path = './training-runs/00001-LSK-auto4-kimg2000/network-snapshot-001000.pkl'
outdir = './outs/fake_imgs'
os.makedirs(outdir, exist_ok=True)

with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

n_imgs = 100

for i in range(n_imgs):
    z = np.random.randn(1, G.z_dim)
    z = torch.Tensor(z).to('cuda')           # latent codes
    c = None                      # class labels (not used in this example)
    w = G.mapping(z, c)
    img = G.synthesis(w)      # NCHW, float32, dynamic range [-1, +1]
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0, :, :, 0].cpu().numpy(), 'L').save(f'{outdir}/{i:06d}.png')
