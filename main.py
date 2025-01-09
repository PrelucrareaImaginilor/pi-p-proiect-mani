import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from segmentation_models_pytorch import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.colors import ListedColormap
from tqdm import tqdm


newsize = (256, 256)
batch_size = 32
num_workers = 4
num_classes = 10  # 9 vertebre + 1 clasa background
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # folosim gpu daca se poate
epochs = 20
learning_rate = 1e-3
TRAIN = False
output_dir = "C:\\Users\\maria\\Downloads\\archive\\data"
im_dir = os.path.join(output_dir, "images")
mask_dir = os.path.join(output_dir, "masks")

items = list(Path(im_dir).glob("*.png"))
image_names = [o.name for o in items]
images = list(set([o.split('_')[0] for o in image_names]))

fold_df = pd.DataFrame({"image_name": images})
np.random.seed(42)

# 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, v_ind) in enumerate(kf.split(fold_df)):
    fold_df.loc[v_ind, 'fold'] = i+1

def get_fold(fn, df):
    image_name = fn.name.split("_")[0]
    return df.loc[df.image_name==image_name, 'fold'].values[0]

olds = [get_fold(o, fold_df) for o in items]
df = pd.DataFrame({"image": image_names, "fold": olds})

class SEGDataset(Dataset):
    def __init__(self, df, mode, transforms=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = os.path.join(im_dir, row.image)
        mask_path = os.path.join(mask_dir, row.image)


        image = Image.open(image_path).convert('L')
        image = np.asarray(image)
        if (image > 1).any():  # normalizam
            image = image / 255.0

        mask = Image.open(mask_path)
        mask = np.asarray(mask)

        mask = np.where(mask <= 9, mask, 0)
        assert mask.max() < num_classes, f"Mask value {mask.max()} exceeds number of classes {num_classes}"

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]


        mask = torch.as_tensor(mask).long()
        mask = torch.nn.functional.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


        image = torch.as_tensor(image).float()

        return image, mask


transforms_train = A.Compose([
    A.Resize(newsize[0], newsize[1]),
    A.HorizontalFlip(),
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2()
])

transforms_valid = A.Compose([
    A.Resize(newsize[0], newsize[1]),
    A.Normalize(mean=[0.485], std=[0.229]),
    ToTensorV2()
])

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.4, weight_iou=0.6):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_iou = weight_iou
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets):
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy_loss(inputs, targets.argmax(dim=1))

        # IoU Loss
        probs = F.softmax(inputs, dim=1)
        intersection = torch.sum(probs * targets, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean(dim=1)

        # Combinam losses
        loss = self.weight_ce * ce_loss + self.weight_iou * iou_loss.mean()
        return loss


train_ = df[df['fold'] != 5].reset_index(drop=True)
valid_ = df[df['fold'] == 5].reset_index(drop=True)

dataset_train = SEGDataset(train_, 'train', transforms_train)
dataset_valid = SEGDataset(valid_, 'valid', transforms_valid)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)

model = Unet(
    encoder_name="resnet34",
    classes=num_classes,
    in_channels=1
)


criterion = CombinedLoss()
model.to(device)


def run(train_loader, val_loader, model, learning_rate, criterion, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}...")
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            print(f"Processing batch {train_loader.batch_size}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        val_loss /= len(val_loader)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './simple_unet.pth')

        scheduler.step(val_loss)
        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == '__main__':
    if TRAIN:
        run(train_loader, val_loader, model, learning_rate, criterion, epochs, device)
    else:
        model.load_state_dict(torch.load("./simple_unet.pth"))

def segment_spine(image_path, model, device):
    image = Image.open(image_path).convert('L')
    image = np.asarray(image)
    image = image / 255.0

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2()
    ])
    image = transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)

    print(f"Original image size: {image.shape}")
    output = output.argmax(1).squeeze().cpu().numpy()

    return output

image_path = "C:\\Users\\maria\\Desktop\\9_08.png"
output_mask = segment_spine(image_path, model, device)

def display_segmentation(image_path, output_mask):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    output_mask = (output_mask > 0.5).astype(int)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    colors = ['black', 'pink']
    custom_cmap = ListedColormap(colors)

    plt.imshow(output_mask, cmap=custom_cmap)
    plt.title("Segmented Image")
    plt.axis("off")
    plt.show()


display_segmentation(image_path, output_mask)
print(f"Max class value in segmented mask: {np.max(output_mask)}")
print(f"Segmented image size: {output_mask.shape}")
