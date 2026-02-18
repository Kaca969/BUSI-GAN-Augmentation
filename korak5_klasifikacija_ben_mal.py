import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

@dataclass
class Konfig:
    root: str = "Dataset_BUSI_with_GT"
    klase: Tuple[str, str] = ("benign", "malignant")  # 2 klase
    img_size: int = 128

    batch: int = 64
    epohe: int = 8
    lr: float = 1e-3
    seed: int = 42

    gan_ckpt: str = os.path.join("izlaz_gan", "G_epoha_015.pt")
    latent: int = 128

    synth_po_klasi: int = 100


CFG = Konfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nadji_slike_2klase(root: str, klase: Tuple[str, str]) -> List[Tuple[str, int]]:
    mapa = {klase[0]: 0, klase[1]: 1}
    uzorci = []
    for k in klase:
        folder = os.path.join(root, k)
        if not os.path.isdir(folder):
            raise RuntimeError(f"Ne nalazim folder: {folder}")
        for f in os.listdir(folder):
            fl = f.lower()
            if not fl.endswith((".png", ".jpg", ".jpeg")):
                continue
            if "mask" in fl:
                continue
            uzorci.append((os.path.join(folder, f), mapa[k]))
    return uzorci

class RealDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("L")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)


class TensorDatasetSimple(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return self.xs.size(0)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class Generator3(nn.Module):
    def __init__(self, latent_dim=128, broj_klasa=3):
        super().__init__()
        self.embed = nn.Embedding(broj_klasa, broj_klasa)
        ulaz_dim = latent_dim + broj_klasa

        self.fc = nn.Sequential(
            nn.Linear(ulaz_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z, y):
        y_emb = self.embed(y)
        x = torch.cat([z, y_emb], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        return self.deconv(x)


def load_generator_3klase(ckpt_path: str, latent: int) -> Generator3:
    G = Generator3(latent_dim=latent, broj_klasa=3).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)
    G.load_state_dict(sd)
    G.eval()
    return G


@torch.no_grad()
def generisi_sintetiku(G: Generator3, latent: int, po_klasi: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs_all = []
    ys_all = []
    for lab in [0, 1]:  # 0 benign, 1 malignant
        preostalo = po_klasi
        while preostalo > 0:
            b = min(64, preostalo)
            z = torch.randn(b, latent, device=DEVICE)
            y = torch.full((b,), lab, dtype=torch.long, device=DEVICE)
            fake = G(z, y)  # [-1,1]
            xs_all.append(fake.cpu())
            ys_all.append(torch.full((b,), lab, dtype=torch.long))
            preostalo -= b

    xs = torch.cat(xs_all, dim=0)
    ys = torch.cat(ys_all, dim=0)
    return xs, ys

def train_one(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    total = 0
    correct = 0
    ys = []
    ps = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1)

        total += x.size(0)
        correct += (pred == y).sum().item()

        ys.append(y.cpu())
        ps.append(pred.cpu())

    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()
    return correct / total, ys, ps


def run_experiment(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset, tag: str):
    train_loader = DataLoader(train_ds, batch_size=CFG.batch, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch, shuffle=False, num_workers=2)

    model = SimpleCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=CFG.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_sd = None

    for e in range(1, CFG.epohe + 1):
        tr_loss, tr_acc = train_one(model, train_loader, opt, loss_fn)
        val_acc, _, _ = eval_model(model, val_loader)
        print(f"[{tag}] Epoha {e:02d}/{CFG.epohe} | train loss={tr_loss:.4f} acc={tr_acc:.3f} | val acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    test_acc, y_true, y_pred = eval_model(model, test_loader)
    print(f"\n[{tag}] TEST accuracy: {test_acc:.4f}")
    print(f"[{tag}] Confusion matrix (rows=true, cols=pred):\n{confusion_matrix(y_true, y_pred)}")
    print(f"[{tag}] Report:\n{classification_report(y_true, y_pred, target_names=['benign','malignant'])}")

    return test_acc


def main():
    set_seed(CFG.seed)

    transform = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [-1,1]
    ])

    samples = nadji_slike_2klase(CFG.root, CFG.klase)
    if len(samples) == 0:
        raise RuntimeError("Nema slika. Proveri CFG.root i strukturu foldera.")

    y_all = [lab for _, lab in samples]

    # Split 70/15/15 (image-level)
    idx = list(range(len(samples)))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=CFG.seed, stratify=y_all)
    y_temp = [y_all[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=CFG.seed, stratify=y_temp)

    tr_samples = [samples[i] for i in idx_train]
    va_samples = [samples[i] for i in idx_val]
    te_samples = [samples[i] for i in idx_test]

    train_real = RealDataset(tr_samples, transform)
    val_ds = RealDataset(va_samples, transform)
    test_ds = RealDataset(te_samples, transform)

    print(f"Real counts: train={len(train_real)} val={len(val_ds)} test={len(test_ds)}")

    acc_real = run_experiment(train_real, val_ds, test_ds, tag="REAL_ONLY")

    if not os.path.isfile(CFG.gan_ckpt):
        raise RuntimeError(f"Ne nalazim GAN checkpoint: {CFG.gan_ckpt}")

    print("\nUčitavam GAN generator i generišem sintetiku...")
    G = load_generator_3klase(CFG.gan_ckpt, CFG.latent)

    xs_s, ys_s = generisi_sintetiku(G, CFG.latent, CFG.synth_po_klasi)
    print("Sintetika:", xs_s.shape, ys_s.shape)

    # real -> tenzori (jednom)
    real_loader_tmp = DataLoader(train_real, batch_size=CFG.batch, shuffle=False, num_workers=2)
    xs_r = []
    ys_r = []
    for x, yb in real_loader_tmp:
        xs_r.append(x)
        ys_r.append(yb)
    xs_r = torch.cat(xs_r, dim=0)
    ys_r = torch.cat(ys_r, dim=0)

    xs_aug = torch.cat([xs_r, xs_s], dim=0)
    ys_aug = torch.cat([ys_r, ys_s], dim=0)

    train_aug = TensorDatasetSimple(xs_aug, ys_aug)
    print(f"Aug train size: {len(train_aug)} (real {len(train_real)} + synth {xs_s.size(0)})")

    acc_aug = run_experiment(train_aug, val_ds, test_ds, tag="REAL_PLUS_SYNTH")

    print("\n===== REZIME =====")
    print(f"Accuracy (real only):      {acc_real:.4f}")
    print(f"Accuracy (real + synth):   {acc_aug:.4f}")
    print(f"Delta:                    {acc_aug - acc_real:+.4f}")


if __name__ == "__main__":
    main()
