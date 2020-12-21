import torch
import torch.nn as nn
import torch.nn.functional as F

# 2x Conv + ReLU + Skip connection
class ResidualLayer(nn.Module):
    def __init__(self, i_dim, r_dim, h_dim, dropout=0.1):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(i_dim, r_dim, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(r_dim)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(r_dim, h_dim, 1, stride=1)
        self.norm2 = nn.BatchNorm2d(h_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop1(self.norm1(self.conv1(F.relu(x))))
        z = self.drop2(self.norm2(self.conv2(F.relu(z))))
        return x + z

class ReZeroLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels//4, out_channels//4), "constant", 0))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        out = self.conv2(self.conv1(x)) * self.alpha + self.shortcut(x)
        return out

# nb_layers of ResidualLayer in one unit
class ResidualStack(nn.Module):
    def __init__(self, i_dim, r_dim, h_dim, nb_layers, rezero=True):
        super(ResidualStack, self).__init__()

        if rezero:
            stack = [ReZeroLayer(i_dim, h_dim) for _ in range(nb_layers)]
        else:
            stack = [ResidualLayer(i_dim, r_dim, h_dim) for _ in range(nb_layers)]
        self.stack = nn.Sequential(*stack)

    def forward(self, x):
        return self.stack(x)

class UnevenPad(nn.Module):
    def __init__(self):
        super(UnevenPad, self).__init__()

    def forward(self, x):
        return F.pad(x, (1,2,1,2))

# Encoder Module
# Just a few Conv layers followed by a residual stack
class Encoder(nn.Module):
    def __init__(self, i_dim, h_dim, nb_r_layers, r_dim, enc_type, dropout=0.1):
        super(Encoder, self).__init__()

        if enc_type == 'half':
            blocks = [
                nn.Conv2d(i_dim, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.Dropout(dropout)
            ]
        elif enc_type == 'quarter':
            blocks = [
                UnevenPad(), # TODO: replace with a simple function
                nn.Conv2d(i_dim, h_dim, 4, stride=2),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),

                UnevenPad(),
                nn.Conv2d(h_dim, h_dim, 4, stride=2),
                nn.BatchNorm2d(h_dim),
                nn.Dropout(dropout)
            ]
        else:
            raise NotImplementedError

        blocks.append(ResidualStack(h_dim, r_dim, h_dim, nb_r_layers))
        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

# VQ part of VQ-VAE-2
class Quantizer(nn.Module):
    def __init__(self, emd_dim, nb_emd):
        super(Quantizer, self).__init__()

        self.emd_dim = emd_dim # Input dimension
        self.nb_emd = nb_emd # Number of latent embedding vectors
        self.decay = 0.99
 
        self.eps = 1e-5
        
        embed = torch.randn(emd_dim, nb_emd) # Embedding tensor containing nb_emd latent embedding vectors
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(nb_emd))
        self.register_buffer('emd_avg', embed.clone())

    def forward(self, x):
        flatten = x.reshape(-1, self.emd_dim) # flatten input tensor 
    
        # calculate distance between input and all embeddings
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # select the closest embedding
        _, emd_id = (-dist).max(1)
        emd_oneshot = F.one_hot(emd_id, self.nb_emd).type(flatten.dtype)
        emd_id = emd_id.view(*x.shape[:-1])

        # given the minimum embedding idea, retrieve the embedding vector
        q = self.embed_code(emd_id)

        # if we are training, update training metrics
        if self.training:
            emd_oneshot_sum = emd_oneshot.sum(0)
            emd_sum = flatten.transpose(0, 1) @ emd_oneshot

            self.cluster_size.data.mul_(self.decay).add_(emd_oneshot_sum, alpha=1 - self.decay)
            self.emd_avg.data.mul_(self.decay).add_(emd_sum, alpha=1-self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.nb_emd * self.eps) * n
            emd_norm = self.emd_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(emd_norm)

        # calculate error, as max function has no gradient
        diff = (q.detach() - x).pow(2).mean()
        q = x + (q - x).detach() # what is point in this line? x + (q-x) is just q?

        return q, diff, emd_id

    # given an index, returns the corresponding embedding vector
    def embed_code(self, ei):
        return F.embedding(ei, self.embed.transpose(0, 1))

# Decoder module
# simply conv layer, residual stack and some upsampling layers
class Decoder(nn.Module):
    def __init__(self, i_dim, h_dim, o_dim, nb_r_layers, r_dim, dec_type, dropout=0.1):
        super(Decoder, self).__init__()
        blocks = []
        blocks.append(ResidualStack(i_dim, r_dim, i_dim, nb_r_layers))
        blocks.append(nn.ReLU(inplace=True))

        if dec_type == 'half':
            blocks.append(nn.ConvTranspose2d(i_dim, o_dim, 4, stride=2, padding=1))
            blocks.append(nn.BatchNorm2d(o_dim))
        elif dec_type == 'quarter':
            blocks.append(nn.ConvTranspose2d(i_dim, h_dim, 4, stride=2, padding=1))
            blocks.append(nn.BatchNorm2d(h_dim))
            # blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(dropout))

            blocks.append(nn.ConvTranspose2d(h_dim, o_dim, 4, stride=2, padding=1))
            blocks.append(nn.BatchNorm2d(o_dim))
        else:
            raise NotImplementedError

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class VQVAE(nn.Module):
    def __init__(self, i_dim, h_dim, r_dim, nb_r_layers, nb_emd, emd_dim, dropout=0.1):
        super(VQVAE, self).__init__()

        self.enc_b = Encoder(i_dim, h_dim, nb_r_layers, r_dim, 'quarter', dropout=dropout) # bottom level encoder
        self.enc_t = Encoder(h_dim, h_dim, nb_r_layers, r_dim, 'half', dropout=dropout) # top level encoder

        self.quan_ct = nn.Conv2d(h_dim, emd_dim, 1) # resize top encoder output to embedding dim
        self.quan_t = Quantizer(emd_dim, nb_emd) # top level vector quantizer
        self.dec_t = Decoder(emd_dim, h_dim, emd_dim, nb_r_layers, r_dim, 'half', dropout=dropout) # top level decoder

        self.quan_cb = nn.Conv2d(emd_dim + h_dim, emd_dim, 1) # resize bottom encoder output to embedding dim
        self.quan_b = Quantizer(emd_dim, nb_emd) # bottom level vector quantizer

        self.upsample = nn.ConvTranspose2d(emd_dim, emd_dim, 4, stride=2, padding=1) # upsample to embedding dimension

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(emd_dim, emd_dim, 4, stride=2, padding=1), 
            nn.BatchNorm2d(emd_dim),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(emd_dim, emd_dim, 4, stride=2, padding=1)
        )

        self.decoder = Decoder(emd_dim*2, h_dim, i_dim, nb_r_layers, r_dim, 'quarter', dropout=dropout) # final, bottom level decoder that produces final reconstruction

    def forward(self, x):
        qt, qb, diff, _, _ = self.encode(x)
        dec = self.decode(qt, qb)
        return dec, diff

    def encode(self, x):
        # pass input and bottom level encoding to the respective encoding layers
        enc_b = self.enc_b(x)
        enc_t = self.enc_t(enc_b)
    
        # quantize the top level
        qt = self.quan_ct(enc_t).permute(0, 2, 3, 1)
        qt, dt, id_t = self.quan_t(qt)
        qt = qt.permute(0, 3, 1, 2)
        dt = dt.unsqueeze(0)

        # decode the top level and concatenate with bottom level encoding
        dec_t = self.dec_t(qt)
        enc_b = torch.cat([dec_t, enc_b], 1)

        # quantize the bottom level
        qb = self.quan_cb(enc_b).permute(0, 2, 3, 1)
        qb, db, id_b = self.quan_b(qb)
        qb = qb.permute(0, 3, 1, 2)
        db = db.unsqueeze(0)

        return qt, qb, dt+db, id_t, id_b

    def decode(self, qt, qb):
        # produce final reconstruction from concatenation of top and bottom level embedding
        up_t = self.upsample(qt)
        q = torch.cat([up_t, qb], 1)
        dec = self.decoder(q)
        return dec

    def decode_code(self, ct, cb):
        qt = self.quan_t.embed_code(ct)
        qt = qt.permute(0, 3, 1, 2)

        qb = self.quan_b.embed_code(cb)
        qb = qb.permute(0, 3, 1, 2)

        dec = self.decode(qt, qb)
        # dec = F.max_pool2d(dec, 3, stride=3)
        return dec

    def dequantize(self, ct, cb):
        qt = self.quan_t.embed_code(ct)
        qt = qt.permute(0, 3, 1, 2)

        qb = self.quan_b.embed_code(cb)
        qb = qb.permute(0, 3, 1, 2)

        return qt, qb

