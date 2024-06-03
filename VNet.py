import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, random_split, Subset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

image_directory = '/Users/emilyyip/Desktop/MRI/Anatomical_mag_echo5/'
mask_directory = '/Users/emilyyip/Desktop/MRI/whole_liver_segmentation/'
output_path = '/Users/emilyyip/Desktop/MRI/preds/output.txt'
save_dir = '/Users/emilyyip/Desktop/preds/'
pred_dir = os.path.join(save_dir, 'predictions')
img_dir = os.path.join(save_dir, 'images')
from skimage.transform import resize

def get_file_paths(image_directory, mask_directory):
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith('.nii')]
    mask_paths = [os.path.join(mask_directory, filename) for filename in os.listdir(mask_directory) if filename.endswith('.nii')]
    return image_paths, mask_paths

image_paths, mask_paths = get_file_paths(image_directory, mask_directory)

import cv2
def resize_image(image, target_dims):
    rows, cols = image.shape[:2]
    target_rows, target_cols = target_dims

    pad_vert = target_rows - rows
    pad_top = pad_vert // 2
    pad_bot = pad_vert - pad_top

    pad_horz = target_cols - cols
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    img_padded = cv2.copyMakeBorder(image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT)
    return img_padded

class NiftiDataset(Dataset):
    def __init__(self, image_files, mask_files):
        self.image_files = image_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = nib.load(self.image_files[idx]).get_fdata()
        mask = nib.load(self.mask_files[idx]).get_fdata()

        image = resize_image(image, (256,256))
        mask = resize_image(mask, (256,256)) 

        image = resize(image, (64,64))
        mask = resize(mask, (64,64))

        # pad depth
        if image.shape[2] != 40:
            image = self.pad_to_depth(image, 40)
            mask = self.pad_to_depth(mask, 40)

        # convert arrays to pytorch tensors
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
            x += identity 
        
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
            x = F.interpolate(x, size=skip.size()[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        #filters
        filters = [1, 64, 128, 256, 256]  
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
        return torch.sigmoid(x)  # binary classification

kf = KFold(n_splits=3, shuffle=True, random_state=123)
# make model
model = VNet()
print(model)
batch_size = 1

dataset = NiftiDataset(image_paths, mask_paths)
# Assuming you have already created 'dataset' as an instance of NiftiDataset
total_size = len(dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for both train and test datasets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Dice Coefficient, TPR, and FPR Functions
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = y_true.contiguous().view(-1)
    y_pred_f = y_pred.contiguous().view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def tprf(y_true, y_pred, threshold=0.5):
    tp = ((y_pred >= threshold) & (y_true == 1)).float().sum().item()
    fn = ((y_pred < threshold) & (y_true == 1)).float().sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else -1

def fprf(y_true, y_pred, threshold=0.5):
    fp = ((y_pred >= threshold) & (y_true == 0)).float().sum().item()
    tn = ((y_pred < threshold) & (y_true == 0)).float().sum().item()
    return fp / (fp + tn) if (fp + tn) > 0 else -1

loss_data = []
dice_data = []
fpr_data = []
tpr_data = []

def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()
    
from datetime import datetime


def train_model(model, loader, optimizer):
    to_device(model.train())
    criterion = nn.BCELoss()

    running_loss = 0.0
    running_Dice = 0.0
    running_fprf = 0.0
    running_tprf = 0.0
    running_samples = 0
    batch_idx = 0
    
    for inputs, targets in loader:
        batch_idx = batch_idx + 1
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)

        outputs = model(inputs)
        outputs = outputs.squeeze(dim=1)

        outputs = outputs.to(torch.float)

        targets = targets.squeeze(dim=1)
        loss = criterion(outputs, targets)
        
        TPR = tprf(outputs, targets)
        FPR = fprf(outputs,targets)
        Dice = dice_coef(outputs, targets)
        loss.backward()
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()
        running_Dice += Dice.item()
        running_fprf += FPR
        running_tprf += TPR

    fpr_data.append(running_fprf / (batch_idx)) 
    tpr_data.append(running_tprf / (batch_idx))   
    loss_data.append(running_loss / (batch_idx))
    dice_data.append(running_Dice / (batch_idx))

    print("Trained {} samples, Loss: {:.4f}, DiceCoef: {:.4f}, FPR: {:.4f}, TPR: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx),
        running_Dice / (batch_idx),
        running_fprf / (batch_idx),
        running_tprf / (batch_idx)
    ))


def prediction_accuracy(ground_truth_labels, predicted_labels):
    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()
    

to_device(model)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)

def train_loop(model, loader, test_data, epochs, optimizer, scheduler):

    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        current_time = datetime.now()
        print("Current time:", current_time.strftime("%H:%M:%S"))
        train_model(model, loader, optimizer)
        torch.save(model.state_dict(), "/Users/emilyyip/Desktop/MRI/preds/model.pth")

        if scheduler is not None:
            scheduler.step()

        print("")

empty=[]


pred_dice = []
file_num = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold+1}/{kf.n_splits}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        train_loop(model, train_loader, empty, (1, 10), optimizer, scheduler)

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.plot(loss_data, color='r')
        plt.ylabel('Losses')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper right')
        plt.subplot(1,2,2)
        plt.plot(dice_data, color='r')
        plt.ylabel('dice_coef')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(f'/Users/emilyyip/Desktop/MRI/preds/process.png')
        plt.close()

        model.load_state_dict(torch.load("/Users/emilyyip/Desktop/MRI/preds/model.pth"))
        to_device(model.eval())


        pred_dice = []
        file_num = 0

        # use test loader here, add fpr and tpr
        for inputs, targets in test_loader:
            inputs = to_device(inputs)
            targets = to_device(targets)

            outputs = model(inputs)
            outputs = outputs.to(torch.float)
            test_img = inputs.squeeze(dim=1)
            test_mask = targets.squeeze(dim=1)
            pred_mask = outputs.squeeze(dim=1)

            test_img = test_img.cpu().detach().numpy()
            test_mask = test_mask.cpu().detach().numpy()
            pred_mask = pred_mask.cpu().detach().numpy()

            Dice = dice_coef(outputs, targets)
            
            pred_dice.append(Dice.item())

            for batch in range(batch_size):
                for slice in range(0,40,10):
                            plt.figure(figsize=(15, 10))
                            plt.subplot(1, 3, 1)
                            plt.imshow(test_img[batch,:,:,slice], cmap='binary')
                            plt.title('Original Image')
                            plt.axis('off')
                            plt.subplot(1, 3, 2)
                            plt.imshow(test_mask[batch,:,:,slice], cmap='binary')
                            plt.title('Ground Truth')
                            plt.axis('off')
                            plt.subplot(1, 3, 3)
                            plt.imshow(pred_mask[batch,:,:,slice], cmap='binary')
                            plt.title('Prediction')
                            plt.axis('off')
                            plt.savefig(f'/Users/emilyyip/Desktop/MRI/preds/images/img{file_num}_slice{slice}.png')
                            plt.close()
                            file_num += 1
                            train = np.array(dice_data)
                            test = np.array(pred_dice)
                            test_dice = np.mean(test)

                            f = open("/Users/emilyyip/Desktop/MRI/preds/output.txt", "a")
                            print('Best training dice score:', file=f)
                            print(np.max(train), file=f)
                            print('Average prediction dice score:', file=f)
                            print(test_dice, file=f)
                            f.close()
                            print('Average prediction dice score:')
                            print(test_dice)

