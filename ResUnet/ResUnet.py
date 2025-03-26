import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
from tqdm import tqdm


# 配置参数类
class Config:
    def __init__(self):
        # 路径设置
        self.data_root = "../data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
        self.checkpoint_dir = "../first"

        # 训练参数
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.epochs = 100
        self.num_workers = 4
        self.train_ratio = 0.8

        # 模型参数
        self.in_channels = 4  # 4个模态
        self.num_classes = 4  # 0:背景 1:坏死 2:水肿 3:增强肿瘤
        self.drop_rate = 0.2


# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 残差瓶颈模块
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch // 4)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch // 4, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch // 4)
        self.conv3 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = self.dropout(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ResUnet模型
class ResUnet(nn.Module):
    def __init__(self, config):
            super().__init__()

            self.enc1 = self._make_layer(config.in_channels, 64, 2)  # [B,64,200,200]
            self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 200→100

            self.enc2 = self._make_layer(64, 128, 3)  # [B,128,100,100]
            self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 100→50

            self.enc3 = self._make_layer(128, 256, 3)  # [B,256,50,50]
            self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 50→25

            self.enc4 = self._make_layer(256, 512, 5)  # [B,512,25,25]
            self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 25→13

            self.enc5 = self._make_layer(512, 1024, 14)  # [B,1024,13,13]

            # Bridge
            self.bridge = self._make_layer(1024, 1024, 4)  # [B,1024,13,13]

            # Decoder
            self.up5 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, output_padding=0)
            self.dec5 = self._make_layer(1024, 512, 2)  # 512(up5) + 512(enc4) = 1024

            self.up4 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
            self.dec4 = self._make_layer(512, 256, 2)  # 256(up4) + 256(enc3) = 512

            self.up3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)
            self.dec3 = self._make_layer(256, 128, 2)  # 128(up3) + 128(enc2) = 256

            self.up2 = nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)
            self.dec2 = self._make_layer(128, 64, 2)  # 64(up2) + 64(enc1) = 128

            self.final = nn.Conv2d(64, config.num_classes, 1)

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(  # 修正此处括号闭合
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )  # 添加闭合括号
        layers = [Bottleneck(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_ch, out_ch))

        return nn.Sequential(*layers)  # 保持正确缩进

    def forward(self, x):
        # Encoder
        # Encoder
        e1 = self.enc1(x)  # [B,64,200,200]
        e2 = self.enc2(self.pool1(e1))  # [B,128,100,100]
        e3 = self.enc3(self.pool2(e2))  # [B,256,50,50]
        e4 = self.enc4(self.pool3(e3))  # [B,512,25,25]
        e5 = self.enc5(self.pool4(e4))  # [B,1024,13,13]

        # Bridge
        b = self.bridge(e5)  # [B,1024,13,13]

        # Decoder
        d5 = self.up5(b)  # [B,512,25,25]
        d5 = torch.cat([d5, e4], dim=1)  # 现在尺寸匹配
        d5 = self.dec5(d5)

        d4 = self.up4(d5)  # [B,256,50,50]
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # [B,128,100,100]
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # [B,64,200,200]
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.final(d2)  # [B,num_classes,200,200]

# 数据加载类
class BraTSDataset(Dataset):
    def __init__(self, data_dir, cases, slice_idx=78, augment=True):  # 155层取中间层
        self.data_dir = data_dir
        self.cases = cases
        self.slice_idx = slice_idx
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        # 加载3D数据并选择中间切片
        flair = nib.load(os.path.join(self.data_dir, case, f"{case}_flair.nii")).get_fdata()[:, :, self.slice_idx]
        t1 = nib.load(os.path.join(self.data_dir, case, f"{case}_t1.nii")).get_fdata()[:, :, self.slice_idx]
        t1ce = nib.load(os.path.join(self.data_dir, case, f"{case}_t1ce.nii")).get_fdata()[:, :, self.slice_idx]
        t2 = nib.load(os.path.join(self.data_dir, case, f"{case}_t2.nii")).get_fdata()[:, :, self.slice_idx]

        seg = nib.load(os.path.join(self.data_dir, case, f"{case}_seg.nii")).get_fdata()[:, :, self.slice_idx]
        seg[seg == 4] = 3

        # 转换为2D多通道图像
        image = np.stack([flair, t1, t1ce, t2], axis=-1)  # [H,W,C]
        image, seg = self._preprocess(image, seg)

        return (
            torch.FloatTensor(image).permute(2, 0, 1),  # [C,H,W]
            torch.LongTensor(seg)  # [H,W]
        )

    def _preprocess(self, image, seg):
        """处理2D切片"""
        # 输入image: [H,W,C]
        # 输入seg: [H,W]

        # 裁剪到128x128 => 200x200
        image = image[20:220, 20:220, :]
        seg = seg[20:220, 20:220]
        if self.augment:
            # 随机水平翻转
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
                seg = np.flip(seg, axis=0).copy()
            # 随机垂直翻转
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()
                seg = np.flip(seg, axis=1).copy()
            # 随机平移（最大平移10像素）
            dx, dy = np.random.randint(-10, 10, size=2)
            image = np.roll(image, dx, axis=0)
            image = np.roll(image, dy, axis=1)
            seg = np.roll(seg, dx, axis=0)
            seg = np.roll(seg, dy, axis=1)
        # 标准化每个模态
        for c in range(4):
            channel = image[..., c]
            image[..., c] = (channel - channel.mean()) / (channel.std() + 1e-8)

        return image, seg


# 混合损失函数
class DiceCELoss(nn.Module):
    def __init__(self, weight=0.6):
        super().__init__()
        self.weight = weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)

        # 计算Dice Loss
        pred_prob = torch.softmax(pred, dim=1)
        dice_loss = 0
        smooth = 1e-5

        # 对每个类别计算Dice
        for cls in range(1, pred_prob.shape[1]):  # 跳过背景
            pred_mask = pred_prob[:, cls]
            target_mask = (target == cls).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            dice_loss += 1 - (2 * intersection + smooth) / (union + smooth)

        dice_loss /= (pred_prob.shape[1] - 1)  # 平均各类别

        return self.weight * dice_loss + (1 - self.weight) * ce_loss


# 计算各区域Dice系数
def calculate_dice(pred, target):
    """
    返回:
    (wt_dice, tc_dice, et_dice)
    """
    # 转换为硬预测
    pred = torch.argmax(pred, dim=1)

    # 创建各区域mask
    wt_pred = (pred > 0).float()
    wt_target = (target > 0).float()

    tc_pred = ((pred == 1) | (pred == 3)).float()
    tc_target = ((target == 1) | (target == 3)).float()

    et_pred = (pred == 3).float()
    et_target = (target == 3).float()

    # 计算Dice
    smooth = 1e-5
    wt_inter = (wt_pred * wt_target).sum()
    wt_dice = (2 * wt_inter + smooth) / (wt_pred.sum() + wt_target.sum() + smooth)

    tc_inter = (tc_pred * tc_target).sum()
    tc_dice = (2 * tc_inter + smooth) / (tc_pred.sum() + tc_target.sum() + smooth)

    et_inter = (et_pred * et_target).sum()
    et_dice = (2 * et_inter + smooth) / (et_pred.sum() + et_target.sum() + smooth)

    return wt_dice.item(), tc_dice.item(), et_dice.item()


def main():
    # 初始化配置
    config = Config()
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 准备数据
    all_cases = [d for d in os.listdir(config.data_root) if d.startswith("BraTS20")]

    # 计算实际样本数
    n_total = len(all_cases)
    n_train = int(n_total * config.train_ratio)
    n_val = n_total - n_train

    train_cases, val_cases = random_split(
        all_cases,
        [n_train, n_val],  # 使用整数样本数
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = BraTSDataset(config.data_root, train_cases)
    val_dataset = BraTSDataset(config.data_root, val_cases)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUnet(config).to(device)
    criterion = DiceCELoss(weight=0.6)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    best_dice = 0
    for epoch in range(config.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]")
        for images, segs in pbar:
            images = images.to(device)
            segs = segs.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, segs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()
        val_loss = 0
        wt_dice, tc_dice, et_dice = 0, 0, 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Val]")
            for images, segs in pbar:
                images = images.to(device)
                segs = segs.to(device)

                outputs = model(images)
                loss = criterion(outputs, segs)
                val_loss += loss.item()

                # 计算Dice
                batch_wt, batch_tc, batch_et = calculate_dice(outputs, segs)
                wt_dice += batch_wt
                tc_dice += batch_tc
                et_dice += batch_et

                pbar.set_postfix({
                    "wt": f"{batch_wt:.3f}",
                    "tc": f"{batch_tc:.3f}",
                    "et": f"{batch_et:.3f}"
                })

        # 计算平均指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_wt = wt_dice / len(val_loader)
        avg_tc = tc_dice / len(val_loader)
        avg_et = et_dice / len(val_loader)

        # 打印结果
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Dice Scores:")
        print(f"  Whole Tumor (WT): {avg_wt:.4f}")
        print(f"  Tumor Core (TC): {avg_tc:.4f}")
        print(f"  Enhancing Tumor (ET): {avg_et:.4f}")

        # 保存最佳模型
        current_dice = (avg_wt + avg_tc + avg_et) / 3
        if current_dice > best_dice:
            best_dice = current_dice
            torch.save(model.state_dict(),
                       os.path.join(config.checkpoint_dir, "best_model.pth"))
            print(f"Saved new best model with avg dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
