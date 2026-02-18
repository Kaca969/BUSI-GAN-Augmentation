import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


# =========================
# PODESAVANJA
# =========================
ROOT = "Dataset_BUSI_with_GT"
IZLAZ_DIR = "izlaz_gan"
os.makedirs(IZLAZ_DIR, exist_ok=True)

KLASE = ["benign", "malignant", "normal"]
MAPA = {k: i for i, k in enumerate(KLASE)}

BATCH = 64
IMG = 128
LATENT = 128

EPOHE = 15
LR = 2e-4
BETAS = (0.5, 0.999)

# Stabilizacija
REAL_LABEL_D = 0.9  # label smoothing samo za D real
NOISE_STD = 0.03    # mali šum na real slike

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# =========================
# DATASET
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
])


class BUSIDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform

        for k in KLASE:
            folder = os.path.join(root, k)
            if not os.path.isdir(folder):
                raise RuntimeError(f"Ne nalazim folder: {folder}")
            for f in os.listdir(folder):
                fl = f.lower()
                if fl.endswith((".png", ".jpg", ".jpeg")) and "mask" not in fl:
                    self.samples.append((os.path.join(folder, f), MAPA[k]))

        if len(self.samples) == 0:
            raise RuntimeError("Nema slika za trening. Proveri dataset putanju i strukturu.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        return img, label


# =========================
# MODELI (cGAN)
# =========================
class Generator(nn.Module):
    def __init__(self, latent_dim=128, broj_klasa=3):
        super().__init__()
        # embedding dim = broj_klasa (ostavljamo kao kod tebe)
        self.embed = nn.Embedding(broj_klasa, broj_klasa)
        ulaz_dim = latent_dim + broj_klasa

        self.fc = nn.Sequential(
            nn.Linear(ulaz_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 128x128
            nn.Tanh()
        )

    def forward(self, z, y):
        y_emb = self.embed(y)            # (B, broj_klasa)
        x = torch.cat([z, y_emb], dim=1) # (B, latent+broj_klasa)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        return self.deconv(x)


class Discriminator(nn.Module):
    """
    cGAN diskriminator:
    ulaz: slika (B,1,H,W) + label mapa (B,cond_ch,H,W)
    izlaz: logit (B,1)

    Ispravka:
    - nema embedding 128*128
    - nema hardkodovanih dimenzija (radi i za IMG != 128)
    """
    def __init__(self, broj_klasa=3, cond_ch=32):
        super().__init__()
        self.cond_ch = cond_ch
        self.embed = nn.Embedding(broj_klasa, cond_ch)

        self.net = nn.Sequential(
            nn.Conv2d(1 + cond_ch, 32, 4, 2, 1),   # H/2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),            # H/4
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),           # H/8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),          # H/16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            # Linear dim zavisi od H,W -> rešimo u forward sa lazy linear (ili računamo za IMG)
        )

        # LazyLinear radi bez ručnog računanja dimenzije
        self.fc = nn.LazyLinear(1)

    def forward(self, img, y):
        b, _, h, w = img.shape
        y_emb = self.embed(y)                       # (B, cond_ch)
        y_map = y_emb.view(b, self.cond_ch, 1, 1)   # (B, cond_ch,1,1)
        y_map = y_map.expand(b, self.cond_ch, h, w) # (B, cond_ch,H,W)
        x = torch.cat([img, y_map], dim=1)          # (B,1+cond_ch,H,W)
        x = self.net(x)
        return self.fc(x)


# =========================
# PREVIEW
# =========================
def sacuvaj_preview(G, epoha):
    G.eval()
    with torch.no_grad():
        for lab, naziv in [(0, "benign"), (1, "malignant"), (2, "normal")]:
            z = torch.randn(64, LATENT, device=DEVICE)
            y = torch.full((64,), lab, dtype=torch.long, device=DEVICE)
            fake = G(z, y)
            save_image((fake + 1) / 2, os.path.join(IZLAZ_DIR, f"{naziv}_epoha_{epoha:03d}.png"), nrow=8)


# =========================
# TRENING
# =========================
def main():
    ds = BUSIDataset(ROOT, transform)

    # Windows-safe: ako ti pravi probleme, stavi num_workers=0
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True)

    G = Generator(LATENT, 3).to(DEVICE)
    D = Discriminator(3, cond_ch=32).to(DEVICE)

    optG = optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    optD = optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    bce = nn.BCEWithLogitsLoss()

    for e in range(1, EPOHE + 1):
        G.train()
        D.train()

        pbar = tqdm(dl, desc=f"Epoha {e}/{EPOHE}")
        for real, y in pbar:
            real = real.to(DEVICE)
            y = y.to(DEVICE)
            b = real.size(0)

            # =====================
            # D update
            # =====================
            optD.zero_grad()

            lbl_real_D = torch.full((b, 1), REAL_LABEL_D, device=DEVICE)
            lbl_fake = torch.zeros((b, 1), device=DEVICE)

            # šum na real slike
            real_noisy = real + NOISE_STD * torch.randn_like(real)
            real_noisy = torch.clamp(real_noisy, -1, 1)

            out_real = D(real_noisy, y)
            loss_real = bce(out_real, lbl_real_D)

            z = torch.randn(b, LATENT, device=DEVICE)
            fake = G(z, y)
            out_fake = D(fake.detach(), y)
            loss_fake = bce(out_fake, lbl_fake)

            lossD = loss_real + loss_fake
            lossD.backward()
            optD.step()

            # =====================
            # G update
            # =====================
            optG.zero_grad()
            lbl_real_G = torch.ones((b, 1), device=DEVICE)  # cilj = 1.0
            out_fake2 = D(fake, y)
            lossG = bce(out_fake2, lbl_real_G)
            lossG.backward()
            optG.step()

            pbar.set_postfix(lossD=float(lossD.item()), lossG=float(lossG.item()))

        sacuvaj_preview(G, e)
        torch.save(G.state_dict(), os.path.join(IZLAZ_DIR, f"G_epoha_{e:03d}.pt"))

    print("Gotovo. Preview slike i checkpointovi su u folderu:", IZLAZ_DIR)


if __name__ == "__main__":
    main()
