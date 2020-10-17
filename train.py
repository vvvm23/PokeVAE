import torch
import torch.nn
import torch.nn.functional as F

import torchvision

from model import VQVAE

import tqdm
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 128
NB_EPOCHS = 200
TRY_CUDA = True

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

model = VQVAE(3, 16, 4, 2, 64, 3).to(device)
crit = torch.nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = torchvision.datasets.ImageFolder('data/pokegan/',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize(128),
        torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomRotation(5),
        torchvision.transforms.ToTensor()
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
    for x, _ in tqdm.tqdm(train_loader):
        optim.zero_grad()
        model.zero_grad()

        x = x.to(device)
        out, l_loss = model(x)
        l_loss = l_loss.mean()
        r_loss = crit(out, x)
        loss = r_loss + l_loss * 0.25
        training_loss += loss.item()

        loss.backward()
        optim.step()

    # evaluate
    model.eval()
    test_loss = 0.0
    for x, _ in test_loader:
        optim.zero_grad()
        model.zero_grad()

        x = x.to(device)
        out, l_loss = model(x)
        l_loss = l_loss.mean()
        r_loss = crit(out, x)

        loss = r_loss + l_loss * 0.25
        test_loss += loss.item()

    print(f"Training Loss: {training_loss / len(train_loader)}")
    print(f"Evaluation Loss: {test_loss / len(test_loader)}\n")

model.eval()
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=test_sampler)

fig, axs = plt.subplots(8, 8)
for i, (x, _) in enumerate(test_loader):
    if i == 16:
        break
    x = x.to(device)

    qt, qb, _, _, _ = model.encode(x)

    y = model.decode(qt, qb)

    x = x.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
    qt = qt[0, :, :, :].reshape(16, 16, 3).detach().cpu().numpy()
    qb = qb[0, :, :, :].reshape(32, 32, 3).detach().cpu().numpy()
    y = y.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()

    axs[i // 2, 0 + (i % 2)*4].imshow(x, interpolation='none')
    axs[i // 2, 1 + (i % 2)*4].imshow(qt, interpolation='none')
    axs[i // 2, 2 + (i % 2)*4].imshow(qb, interpolation='none')
    axs[i // 2, 3 + (i % 2)*4].imshow(y, interpolation='none')

[[x.axis(False) for x in y] for y in axs]
# plt.savefig('fig.png', dpi=2000)
plt.show()
