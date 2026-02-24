import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F
import tqdm
from architecture import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
LEARNING_RATE = 1e-4
BATCH_SIZE_NB = 16

# --- CONFIGURATION DES CHEMINS ---
# Dossiers TRAIN :
IMG_TRAIN = "/bdd10k/img/train"
MASK_TRAIN = "/abels/train"

# Dossiers VAL :
IMG_VAL = "/bdd10k/img/val"
MASK_VAL = "/bdd10k/labels/val"

class BDD10kDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(384, 640)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size # Format (H, W)

        # On crée l'intersection pour éviter les erreurs
        raw_imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        raw_masks = {f.replace('_train_id.png', ''): f for f in os.listdir(mask_dir) if f.endswith('.png')}

        self.valid_pairs = []
        for img_file in raw_imgs:
            img_id = img_file.split('.')[0]
            if img_id in raw_masks:
                self.valid_pairs.append((img_file, raw_masks[img_id]))

        # Transformation standard ImageNet pour MobileNetV2
        self.img_transform = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.Compose([
            T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]

        # Chargement
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name))

        # Extraction de la ROUTE (ID 0 dans les labels sémantiques)
        mask_np = np.array(mask)
        road_mask = np.where(mask_np == 0, 1.0, 0.0).astype(np.float32)
        road_mask = Image.fromarray(road_mask)

        return self.img_transform(image), self.mask_transform(road_mask)
    

train_dataset = BDD10kDataset(IMG_TRAIN, MASK_TRAIN)
val_dataset = BDD10kDataset(IMG_VAL, MASK_VAL)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_NB, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_NB, shuffle=False)

# 2. Instancier ton modèle (n_class=1 pour la route)
model = MobileNetV2_UNet(n_class=1).to(DEVICE)

# 3. Fonction de perte et Optimiseur
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_fn(model, loader, optimizer, criterion):
    model.train()
    loop = tqdm.tqdm(loader)
    total_loss = 0

    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def validate_fn(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- LANCEMENT DE L'ENTRAÎNEMENT ---
for epoch in range(EPOCHS):
    train_loss = train_fn(model, train_loader, optimizer, criterion)
    val_loss = validate_fn(model, val_loader, criterion)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Sauvegarde du meilleur modèle
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")