import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

image_directory = '/Users/emilyyip/Desktop/MRI/Anatomical_mag_echo5/'
mask_directory = '/Users/emilyyip/Desktop/MRI/whole_liver_segmentation/'
output_path = '/Users/emilyyip/Desktop/MRI/output.txt'
save_dir = '/Users/emilyyip/Desktop/MRI/preds/'
pred_dir = os.path.join(save_dir, 'predictions')
img_dir = os.path.join(save_dir, 'images')

def get_file_paths(image_directory, mask_directory):
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.nii')]
    mask_paths = [os.path.join(mask_directory, filename) for filename in os.listdir(mask_directory) if filename.endswith('.nii')]
    divide = len(image_paths) //5
    return image_paths [:divide],mask_paths[:divide]

image_paths, mask_paths = get_file_paths(image_directory, mask_directory)


class NiftiDataset(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask using nibabel
        image = nib.load(self.image_files[idx]).get_fdata()
        mask = nib.load(self.mask_files[idx]).get_fdata()

        # Pad depth to 40 if not already
        if image.shape[2] != 40:
            image = self.pad_to_depth(image, 40)
            mask = self.pad_to_depth(mask, 40)

        # Convert numpy arrays to PyTorch tensors with dtype float32
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # Add channel dimension

        return image_tensor, mask_tensor

    def pad_to_depth(self, volume, target_depth):
        current_depth = volume.shape[2]
        if current_depth < target_depth:
            pad_size = (target_depth - current_depth) // 2
            pad_extra = (target_depth - current_depth) % 2
            volume = np.pad(volume, ((0, 0), (0, 0), (pad_size, pad_size + pad_extra)), mode='constant', constant_values=0)
        return volume

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, res_connect=False):
        super(ConvBlock, self).__init__()
        self.res_connect = res_connect
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(dropout)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout3d(dropout)

        if res_connect:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        if self.res_connect:
            identity = self.residual(identity)
            x += identity  # Element-wise addition for residual connection
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, res_connect=False):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout=dropout, res_connect=res_connect)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Ensure that the upsampled x has the same dimensions as skip before concatenating
        if x.size() != skip.size():
            # Optionally add a print statement here to debug sizes
            # print("Adjusting size from", x.size(), "to", skip.size())
            x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        # Adjust the filters to start with 1 if the input images are single-channel
        filters = [1, 64, 128, 256, 256]  # Starting with 1 channel now
        self.encoders = nn.ModuleList([
            EncoderBlock(filters[i], filters[i+1], dropout=0.1, res_connect=True)
            for i in range(len(filters)-1)
        ])

        self.bottleneck = ConvBlock(filters[-1], filters[-1], dropout=0.1, res_connect=True)

        # Assuming symmetrical structure for the decoder
        self.decoders = nn.ModuleList([
            EncoderBlock(filters[i+1], filters[i], dropout=0.1, res_connect=True)
            for i in reversed(range(len(filters)-1))
        ])

        self.final_conv = nn.Conv3d(filters[0], 1, kernel_size=1)  # Output segmentation map

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, pooled = encoder(x)
            skips.append(x)
            x = pooled

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x, _ = decoder(x)  # Ignore pooling in decoder
            x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=False)

        x = self.final_conv(x)
        return torch.sigmoid(x)  # Assuming binary classification

# Model instantiation
model = VNet()
print(model)
dataset = NiftiDataset(image_paths, mask_paths)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Helper function to calculate Dice Coefficient
# Ensure directories exist
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# Dice Coefficient, TPR, and FPR Functions
def dice_coef(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def tprf(y_true, y_pred, threshold=0.5):
    tp = ((y_pred >= threshold) & (y_true == 1)).float().sum().item()
    fn = ((y_pred < threshold) & (y_true == 1)).float().sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else -1

def fprf(y_true, y_pred, threshold=0.5):
    fp = ((y_pred >= threshold) & (y_true == 0)).float().sum().item()
    tn = ((y_pred < threshold) & (y_true == 0)).float().sum().item()
    return fp / (fp + tn) if (fp + tn) > 0 else -1

# Training and evaluation loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        print("hi")
        outputs = model(images)
        preds = (outputs > 0.5).float()
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("train")
    #TPRs = []
    #FPRs = []
    model.eval()
    with torch.no_grad():
        dice_scores = []
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()
            dice_scores.append(dice_coef(preds, masks).item())

    # Calculate average scores
    avg_dice = np.mean(dice_scores)
    #avg_tpr = np.mean(TPRs)
    #avg_fpr = np.mean(FPRs)
    print(f'Average Dice: {avg_dice:.4f}')
          
 #TPR: {avg_tpr:.4f}, FPR: {avg_fpr:.4f}'

    with open(output_path, "a") as f:
        print(f'Epoch {epoch + 1} Results:', file=f)
        print(f'Average Dice Score: {avg_dice:.4f}', file=f)
        #print(f'Average TPR: {avg_tpr:.4f}', file=f)
        #print(f'Average FPR: {avg_fpr:.4f}', file=f)

print("Training complete.")
