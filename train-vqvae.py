import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from vqvae import VQVAE

import tqdm
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64
NB_EPOCHS = 50
TRY_CUDA = True
NB_EMBED = 512
MODEL_SAVING = 5

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

model = VQVAE(
    i_dim=3, h_dim=128, r_dim=64, nb_r_layers=2,
    nb_emd=NB_EMBED, emd_dim=64
).to(device)

# model.apply(init_weights)
crit = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.003)

dataset = torchvision.datasets.ImageFolder('data/pokegan/',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )

sample_indices = list(range(len(dataset)))
np.random.shuffle(sample_indices)
split_point = int(np.floor(0.1 * dataset.__len__()))
test_indices, train_indices = sample_indices[:split_point], sample_indices[split_point:]

test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)
print("> Using PokeGAN dataset\n")

for ei in range(NB_EPOCHS):
    print(f"> Epoch {ei+1}/{NB_EPOCHS}")
    model.train()
    training_loss = 0.0
    tr_loss = 0.0
    tl_loss = 0.0
    for x, _ in tqdm.tqdm(train_loader):
        optim.zero_grad()

        x = x.to(device)
        out, l_loss = model(x)
        l_loss = l_loss.mean()
        r_loss = crit(out, x)
        loss = r_loss + l_loss * 0.25
        training_loss += loss.item()
        tr_loss += r_loss.item()
        tl_loss += l_loss.item()

        loss.backward()
        optim.step()

    with torch.no_grad():
        # evaluate
        model.eval()
        test_loss = 0.0
        er_loss = 0.0
        el_loss = 0.0
        for i, (x, _) in enumerate(test_loader):
            optim.zero_grad()

            x = x.to(device)
            qt, qb, l_loss, id_t, id_b = model.encode(x)
            out = model.decode(qt, qb)
            l_loss = l_loss.mean()
            r_loss = crit(out, x)

            loss = r_loss + l_loss * 0.25
            test_loss += loss.item()
            er_loss += r_loss.item()
            el_loss += l_loss.item()

            if i == 0:
                img = torch.cat([x, out], dim=0)
                torchvision.utils.save_image(img, f'imgs/vqvae-{ei}.png', normalize=True, range=(-1,1))

                img = id_t.unsqueeze(1) / (NB_EMBED - 1.0)
                torchvision.utils.save_image(img, f'imgs/vqvae-top-{ei}.png', normalize=True, range=(-1,1))

                img = id_b.unsqueeze(1) / (NB_EMBED - 1.0)
                torchvision.utils.save_image(img, f'imgs/vqvae-bottom-{ei}.png', normalize=True, range=(-1,1))

    if MODEL_SAVING > 0 and ei % MODEL_SAVING == 0:
        torch.save(model.state_dict(), f"checkpoints/vqvae-{ei}.pt")

    print(f"Training Loss: {training_loss / len(train_loader)} [r_loss: {tr_loss / len(train_loader)}, l_loss: {tl_loss / len(train_loader)}]")
    print(f"Evaluation Loss: {test_loss / len(test_loader)} [r_loss: {er_loss / len(test_loader)}, l_loss: {el_loss / len(test_loader)}]\n")
