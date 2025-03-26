import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 硬件配置部分
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2023)  # 固定随机种子保证可重复性

# 数据参数配置
class Config:
    data_path = '/path/to/BraTS2020'  # 需修改为实际路径
    batch_size = 2  # 根据GPU显存调整
    num_workers = 4
    lr = 1e-4
    epochs = 100
    img_size = (128, 128, 128)  # 统一缩放后的尺寸
    num_classes = 3  # 背景、肿瘤核心、增强肿瘤

# 自定义数据集加载器
class BraTSDataset(Dataset):
    def __init__(self, scan_ids, mode='train'):
        self.scan_ids = scan_ids
        self.mode = mode
        self.crop_size = Config.img_size
        
    def __len__(self):
        return len(self.scan_ids)
    
    def _load_nii(self, path):
        # 读取nii文件并预处理
        scan = nib.load(path).get_fdata()
        scan = self._normalize(scan)
        return torch.tensor(scan, dtype=torch.float32)
    
    def _normalize(self, scan):
        # 各模态独立标准化
        mask = scan > 0
        mean = scan[mask].mean()
        std = scan[mask].std()
        return (scan - mean) / std if std > 0 else scan
    
    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        
        # 加载四个模态数据 [H, W, D, C]
        modalities = []
        for mod in ['t1', 't1ce', 't2', 'flair']:
            path = os.path.join(Config.data_path, scan_id, f'{scan_id}_{mod}.nii.gz')
            modalities.append(self._load_nii(path))
        image = torch.stack(modalities, dim=-1)  # 合并通道
        
        # 处理标签
        if self.mode != 'test':
            label_path = os.path.join(Config.data_path, scan_id, f'{scan_id}_seg.nii.gz')
            label = self._load_nii(label_path)
            label = self._process_labels(label)  # 合并标签类别
        else:
            label = torch.zeros_like(image[..., 0])
            
        # 随机数据增强
        if self.mode == 'train':
            image, label = self._random_augment(image, label)
            
        # 调整维度顺序为 [C, H, W, D]
        image = image.permute(3, 0, 1, 2)
        return image, label.long()
    
    def _process_labels(self, label):
        # 将BraTS标签合并为三类：背景(0), 肿瘤核心(1), 增强肿瘤(2)
        processed = torch.zeros_like(label)
        processed[label == 1] = 1   # 坏死和非增强肿瘤
        processed[label == 2] = 1   # 水肿
        processed[label == 4] = 2    # 增强肿瘤
        return processed
    
    def _random_augment(self, image, label):
        # 简单3D数据增强
        if np.random.rand() > 0.5:
            # 随机翻转
            axis = np.random.choice([0, 1, 2])
            image = torch.flip(image, [axis])
            label = torch.flip(label, [axis])
        return image, label

# 定义3D UNet模型
class UNet3D(nn.Module):
    def __init__(self, in_channels=4, num_classes=3):
        super().__init__()
        
        # 编码器部分
        self.enc1 = self._block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(128, 256)
        
        # 解码器部分
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(64, 32)
        
        # 输出层
        self.final = nn.Conv3d(32, num_classes, kernel_size=1)
        
    def _block(self, in_channels, features):
        # 基础卷积块
        return nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1),
            nn.InstanceNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1),
            nn.InstanceNorm3d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # 解码器
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

# 损失函数：Dice + CrossEntropy
class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, pred, target):
        # Dice损失计算
        smooth = 1.0
        pred_soft = torch.softmax(pred, dim=1)
        target_onehot = torch.eye(pred.shape[1]).to(DEVICE)[target.squeeze(1)]
        target_onehot = target_onehot.permute(0,4,1,2,3).contiguous()
        
        intersection = torch.sum(pred_soft * target_onehot)
        union = torch.sum(pred_soft) + torch.sum(target_onehot)
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return (1 - dice) + self.ce(pred, target.squeeze(1))

# 训练函数
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    
    with tqdm(loader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
    
    return running_loss / len(loader)

# 验证函数（带指标计算）
def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    total_dice = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 计算Dice系数
            preds = torch.argmax(outputs, dim=1)
            dice = compute_dice(preds, labels)
            total_dice += dice
    
    avg_loss = val_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice

def compute_dice(pred, target):
    # 计算各类别Dice系数
    dice_scores = []
    for cls in range(1, Config.num_classes):  # 忽略背景
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = pred_mask.sum() + target_mask.sum()
        dice = (2. * intersection) / (union + 1e-8)
        dice_scores.append(dice.item())
    return np.mean(dice_scores)

# 主训练流程
def main():
    # 初始化数据
    all_scans = [f for f in os.listdir(Config.data_path) if os.path.isdir(os.path.join(Config.data_path, f))]
    train_ids, val_ids = train_test_split(all_scans, test_size=0.2, random_state=42)
    
    train_set = BraTSDataset(train_ids, 'train')
    val_set = BraTSDataset(val_ids, 'val')
    
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, 
                            shuffle=True, num_workers=Config.num_workers)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size,
                          num_workers=Config.num_workers)
    
    # 初始化模型
    model = UNet3D(in_channels=4, num_classes=Config.num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = DiceCELoss()
    
    # 训练准备
    best_dice = 0.0
    checkpoint_path = 'best_model.pth'
    
    # 开始训练循环
    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        
        # 验证阶段
        val_loss, val_dice = validate(model, val_loader, criterion)
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, checkpoint_path)
            print(f"New best model saved with Dice {best_dice:.4f}")

if __name__ == "__main__":
    main()
