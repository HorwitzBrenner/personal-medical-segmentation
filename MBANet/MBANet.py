import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ---------------------------
# SE模块，用来做通道注意力
# ---------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ---------------------------
# 残差块，带SE模块
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------
# RFIN模块，用来把先验分支的特征注入到领域分支
# ---------------------------
class RFIN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RFIN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.in_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.relu(self.in_norm(self.conv(x)))

# ---------------------------
# DKIN模块，用来将领域分支的特征反馈给先验分支
# ---------------------------
class DKIN(nn.Module):
    def __init__(self, feature_dim):
        super(DKIN, self).__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.layer_norm(out)
        return out

# ---------------------------
# 领域分支：对低分辨率输入进行处理，输出特征图
# ---------------------------
class DomainBranch(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=8):
        super(DomainBranch, self).__init__()
        layers = []
        # 前两层降采样，把256x256降到64x64
        layers.append(nn.Conv2d(in_channels, base_channels, 3, 2, 1))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(base_channels, base_channels, 3, 2, 1))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        # 后续的残差块
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels, base_channels))
        self.domain_net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.domain_net(x)

# ---------------------------
# 先验分支：这里假设加载了预训练的SAM编码器，下面用个简单的Dummy替代
# ---------------------------
class DummySAMEncoder(nn.Module):
    def __init__(self):
        super(DummySAMEncoder, self).__init__()
        self.conv = nn.Conv2d(3, 256, 3, padding=1)
    def forward(self, x):
        # 返回一个列表，实际中可能有多层特征
        return [self.conv(x)]

# ---------------------------
# Dummy的mask decoder，同样作为占位
# ---------------------------
class DummyMaskDecoder(nn.Module):
    def __init__(self):
        super(DummyMaskDecoder, self).__init__()
        self.conv = nn.Conv2d(64, 1, 1)
    def forward(self, x, prompt):
        return torch.sigmoid(self.conv(x))

# ---------------------------
# 先验分支封装，调用SAM编码器
# ---------------------------
class PriorBranch(nn.Module):
    def __init__(self, sam_encoder):
        super(PriorBranch, self).__init__()
        self.sam_encoder = sam_encoder
    def forward(self, x):
        features = self.sam_encoder(x)
        return features

# ---------------------------
# MBA-Net整体模型，整合先验分支、领域分支和双向特征聚合
# ---------------------------
class MBA_Net(nn.Module):
    def __init__(self, sam_encoder, mask_decoder, num_rfin=3, num_dkin=3):
        super(MBA_Net, self).__init__()
        self.prior_branch = PriorBranch(sam_encoder)
        self.domain_branch = DomainBranch(3, 64, 8)
        self.rfin_modules = nn.ModuleList([RFIN(256, 64) for _ in range(num_rfin)])
        self.dkin_modules = nn.ModuleList([DKIN(768) for _ in range(num_dkin)])
        self.mask_decoder = mask_decoder
    
    def forward(self, x_high, x_low, box_prompt):
        # 高分辨率输入进入先验分支
        prior_features = self.prior_branch(x_high)
        # 低分辨率输入进入领域分支
        domain_features = self.domain_branch(x_low)
        # 用RFIN模块处理先验分支返回的第一层特征（示例）
        injected_feature = self.rfin_modules[0](prior_features[0].unsqueeze(-1).unsqueeze(-1))
        # 简单相加得到融合后的特征（实际中要保证尺寸匹配）
        fused_feature = domain_features + injected_feature
        # DKIN模块示例，这里将领域分支的特征展平后送入DKIN
        b, c, h, w = domain_features.size()
        domain_flat = domain_features.view(b, -1)
        integrated_feature = self.dkin_modules[0](domain_flat)
        # 最后用mask decoder预测分割结果
        mask = self.mask_decoder(fused_feature, box_prompt)
        return mask

# ---------------------------
# 自制一个简单数据集，用来做示例训练
# ---------------------------
class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 随机生成数据，模拟真实图像和标签
        x_high = torch.randn(3, 1024, 1024)   # 高分辨率图像
        x_low = torch.randn(3, 256, 256)        # 低分辨率图像
        box_prompt = torch.tensor([100, 100, 924, 924])  # 固定框提示
        label = (torch.rand(1, 256, 256) > 0.5).float()   # 分割标签，值为0或1
        return x_high, x_low, box_prompt, label

# ---------------------------
# 简单函数：计算二分类分割的像素级准确率
# ---------------------------
def compute_accuracy(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    correct = (pred_bin == target).float().sum()
    total = torch.numel(target)
    return correct / total

# ---------------------------
# 主训练函数
# ---------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=10, save_path='best_model.pth'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.99, weight_decay=1e-4)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # 用 tqdm 显示本轮训练进度
        for x_high, x_low, box_prompt, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_high = x_high.to(device)
            x_low = x_low.to(device)
            box_prompt = box_prompt.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(x_high, x_low, box_prompt)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        
        # 验证阶段，不计算梯度
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for x_high, x_low, box_prompt, label in val_loader:
                x_high = x_high.to(device)
                x_low = x_low.to(device)
                box_prompt = box_prompt.to(device)
                label = label.to(device)
                output = model(x_high, x_low, box_prompt)
                acc = compute_accuracy(output, label)
                val_acc += acc.item()
        val_acc /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # 保存验证准确率最高的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"新最佳模型已保存，验证准确率：{best_val_acc:.4f}")

if __name__ == '__main__':
    # 初始化模型，加载预训练SAM编码器和mask decoder的Dummy版本
    sam_encoder = DummySAMEncoder()
    mask_decoder = DummyMaskDecoder()
    model = MBA_Net(sam_encoder, mask_decoder)
    
    # 构造训练和验证数据集
    train_dataset = DummyDataset(20)
    val_dataset = DummyDataset(5)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, device, num_epochs=10, save_path='best_model.pth')
