import torch
import torch.nn.functional as F
import torchvision

from pixelsnail import PixelSnail
from vqvae import VQVAE

import tqdm
import sys

NB_EMBED = 512
TRY_CUDA = True
NB_SAMPLES = 4
LATENT_TOP = (32, 32)
LATENT_BOTTOM = (64, 64)

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

vqvae_path = sys.argv[1]
pixelsnail_top_path = sys.argv[2]
pixelsnail_bottom_path = sys.argv[3]

vqvae = VQVAE(
    i_dim=3, h_dim=128, r_dim=64, nb_r_layers=2,
    nb_emd=NB_EMBED, emd_dim=64
).to(device).eval()

pixelsnail_top = PixelSnail(
    [32, 32],
    nb_class=NB_EMBED,
    channel=128,
    kernel_size=5,
    nb_pixel_block=2,
    nb_res_block=4,
    res_channel=128,
    dropout=0.0,
    nb_out_res_block=1,
).to(device).eval()

pixelsnail_bottom = PixelSnail(
    [64, 64],
    nb_class=NB_EMBED,
    channel=128,
    kernel_size=5,
    nb_pixel_block=2,
    nb_res_block=4,
    res_channel=128,
    dropout=0.0,
    nb_cond_res_block=3,
    cond_res_channel=128,
    nb_out_res_block=1,
    cond_interpolate=2,
    attention=False
).to(device).eval()

with torch.no_grad():
    pb = tqdm.tqdm(total=LATENT_TOP[0]*LATENT_TOP[1])
    top_sample = torch.zeros(NB_SAMPLES, *LATENT_TOP, dtype=torch.int64).to(device)
    cache = {}

    for i in range(LATENT_TOP[0]):
        for j in range(LATENT_TOP[1]):
            pred, cache = pixelsnail_top(top_sample, cache={})
            pred = F.softmax(pred[:, :, i, j], dim=1)
            top_sample[:, i, j] = torch.multinomial(pred, 1).squeeze()
            pb.update(1)

    pb = tqdm.tqdm(total=LATENT_BOTTOM[0]*LATENT_BOTTOM[1])
    bottom_sample = torch.zeros(NB_SAMPLES, *LATENT_BOTTOM, dtype=torch.int64).to(device)
    cache = {}

    for i in range(LATENT_BOTTOM[0]):
        for j in range(LATENT_BOTTOM[1]):
            pred, cache = pixelsnail_bottom(bottom_sample, c=top_sample, cache={})
            pred = F.softmax(pred[:, :, i, j], dim=1)
            bottom_sample[:, i, j] = torch.multinomial(pred, 1).squeeze()
            pb.update(1)

    sample = vqvae.decode_code(top_sample, bottom_sample)
    torchvision.utils.save_image(sample, f"imgs/sample.png")
