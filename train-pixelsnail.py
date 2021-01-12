import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pixelsnail import PixelSnail

import numpy as np

import tqdm
import sys

BATCH_SIZE = 16
NB_EPOCHS = 101
TRY_CUDA = True
NB_EMBED = 512
MODEL_SAVING = 5

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

model_level = sys.argv[2] 
if model_level == 'top':
    pixelsnail = PixelSnail(
        [32, 32],
        nb_class=NB_EMBED,
        channel=256,
        kernel_size=5,
        nb_pixel_block=2,
        nb_res_block=4,
        res_channel=128,
        dropout=0.0,
        nb_out_res_block=1,
    ).to(device)
elif model_level == 'bottom':
    pixelsnail = PixelSnail(
        [64, 64],
        nb_class=NB_EMBED,
        channel=256,
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
    ).to(device)
else:
    print("! Unrecognised level! Exiting..")
    exit()

if len(sys.argv) == 4:
    print(f"> Loading from file.")
    pixelsnail.load_state_dict(torch.load(sys.argv[3]))

crit = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(pixelsnail.parameters(), lr=3e-3)

dataset = torch.load(sys.argv[1])
dataset = torch.utils.data.TensorDataset(*dataset)

sample_indices = list(range(len(dataset)))
np.random.shuffle(sample_indices)
split_point = int(np.floor(0.05 * dataset.__len__()))
test_indices, train_indices = sample_indices[:split_point], sample_indices[split_point:]

test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

for ei in range(NB_EPOCHS):
    print(f"> Epoch {ei+1}/{NB_EPOCHS}")
    pixelsnail.train()
    train_loss = 0.0
    eval_loss = 0.0

    for t, b in tqdm.tqdm(train_loader):
        optim.zero_grad()

        t = t.to(device).long()
        if model_level == 'top':
            out, _ = pixelsnail(t)
            loss = crit(out, t)
        else:
            b = b.to(device).long()
            out, _ = pixelsnail(b, c=t)
            loss = crit(out, b)
        train_loss += loss.item()

        loss.backward()
        optim.step()

    with torch.no_grad():
        pixelsnail.eval()
        for i, (t, b) in enumerate(test_loader):
            optim.zero_grad()
            t = t.to(device).long()
            if model_level == 'top':
                out, _ = pixelsnail(t)
                loss = crit(out, t)
            else:
                b = b.to(device).long()
                out, _ = pixelsnail(b, c=t)
                loss = crit(out, b)
            eval_loss += loss.item()

            if i == 0:
                if model_level == 'top':
                    img = torch.cat([t / (NB_EMBED - 1), out.argmax(dim=1).long() / (NB_EMBED - 1)], dim=0)
                    torchvision.utils.save_image(img.unsqueeze(1), f'imgs/pixelsnail-top-{ei}.png', normalize=True, range=(-1,1))
                else:
                    img = torch.cat([b / (NB_EMBED - 1), out.argmax(dim=1).long() / (NB_EMBED - 1)], dim=0)
                    torchvision.utils.save_image(img.unsqueeze(1), f'imgs/pixelsnail-bottom-{ei}.png', normalize=True, range=(-1,1))

    if MODEL_SAVING > 0 and ei % MODEL_SAVING == 0:
        torch.save(pixelsnail.state_dict(), f"checkpoints/pixelsnail-{model_level}-{ei}.pt")

    print(f"Training Loss: {train_loss / len(train_loader)}")
    print(f"Evaluation Loss: {eval_loss / len(test_loader)}")
