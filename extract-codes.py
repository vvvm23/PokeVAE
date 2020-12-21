import torch
import torch.nn.functional as F
import torchvision

from vqvae import VQVAE

import tqdm
import sys

TRY_CUDA = True
BATCH_SIZE = 64
NB_EMBED = 512

device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'}) \n")

model = VQVAE(
    i_dim=3, h_dim=128, r_dim=64, nb_r_layers=2,
    nb_emd=NB_EMBED, emd_dim=64
)
model.load_state_dict(torch.load(sys.argv[1]))
model = model.to(device)

dataset = torchvision.datasets.ImageFolder('data/pokegan/',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )

loader = torch.utils.data.DataLoader(dataset, BATCH_SIZE)

latent_dataset = (torch.zeros(len(dataset), 32, 32, dtype=torch.int16), torch.zeros(len(dataset), 64, 64, dtype=torch.int16))

with torch.no_grad():
    model.eval()
    for i, (x, _) in enumerate(tqdm.tqdm(loader)):
        x = x.to(device)
        _, _, _, id_t, id_b = model.encode(x)
        latent_dataset[0][i*BATCH_SIZE:(i+1)*BATCH_SIZE] = id_t.detach().cpu()
        latent_dataset[1][i*BATCH_SIZE:(i+1)*BATCH_SIZE] = id_b.detach().cpu()

torch.save(latent_dataset, "latent-dataset.pt")
