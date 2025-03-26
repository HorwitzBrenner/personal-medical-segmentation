import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------------------------
# 从官方 SAM 库加载预训练模型
# segment-anything 库： pip install git+https://github.com/facebookresearch/segment-anything.git
from segment_anything import sam_model_registry

# ---------------------------
# 定义一个简单的融合模块，将 SAM 编码器的特征和领域分支特征合并
# ---------------------------
class FusionModule(nn.Module):
    def __init__(self, prior_channels, domain_channels, out_channels):
        super(FusionModule, self).__init__()
        # 将两路特征通道拼接后用 1x1 卷积映射到 out_channels
        self.conv = nn.Conv2d(prior_channels + domain_channels, out_channels, kernel_size=1)
    def forward(self, prior, domain):
        # 先将低分辨率的领域分支特征上采样到与先验分支一致
        domain_upsampled = F.interpolate(domain, size=prior.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([prior, domain_upsampled], dim=1)
        fused = self.conv(fused)
        return fused

# ---------------------------
# 定义基于 SAM image_encoder 的先验分支
# 这里假设 SAM image_encoder 接受 (B, 3, H, W) 的输入，输出 (B, N, D)
# 为了方便后续融合，我们把输出转成 (B, D, H_patch, W_patch)，同时简单构造一个位置编码（置零）
# ---------------------------
class PriorBranch(nn.Module):
    def __init__(self, sam_model):
        super(PriorBranch, self).__init__()
        self.image_encoder = sam_model.image_encoder
    def forward(self, x):
        # x: 高分辨率输入（例如 1024×1024）
        # image_encoder 输出 shape (B, N, D)
        image_embeddings = self.image_encoder(x)  # 注意：此处不进行归一化等预处理，确保输入符合 SAM 要求
        B, N, D = image_embeddings.shape
        H = W = int(N ** 0.5)  # 假定 N 为完全平方数
        # 转换为 (B, D, H, W)
        image_embeddings = image_embeddings.transpose(1, 2).reshape(B, D, H, W)
        # 这里构造一个简单的位置编码，实际可用 SAM 内部计算的
        image_pe = torch.zeros_like(image_embeddings)
        return image_embeddings, image_pe

# ---------------------------
# 领域分支：对低分辨率图像（例如 256×256）进行特征提取
# ---------------------------
class DomainBranch(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_blocks=8):
        super(DomainBranch, self).__init__()
        layers = []
        # 前两层降采样，将 256x256 降到 64x64
        layers.append(nn.Conv2d(in_channels, base_channels, 3, 2, 1))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(nn.Conv2d(base_channels, base_channels, 3, 2, 1))
        layers.append(nn.BatchNorm2d(base_channels))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        # 后续多个残差块
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels, base_channels))
        self.domain_net = nn.Sequential(*layers)
    def forward(self, x):
        return self.domain_net(x)

# ---------------------------
# SE 模块和残差块（这里注释比较简洁）
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
# MBA-Net 整体模型：使用 SAM 预训练模型和领域分支，融合后调用 SAM 的 mask_decoder 进行预测
# ---------------------------
class MBA_Net(nn.Module):
    def __init__(self, sam_model, num_fusion_channels=256):
        super(MBA_Net, self).__init__()
        # 使用 SAM 的各个模块
        self.sam_model = sam_model  # 包含 image_encoder, prompt_encoder, mask_decoder
        self.prior_branch = PriorBranch(sam_model)
        self.domain_branch = DomainBranch(in_channels=3, base_channels=64, num_blocks=8)
        # 融合模块，设定 SAM image_encoder 输出的通道数（例如 256）和领域分支输出的通道（64）
        self.fusion_module = FusionModule(prior_channels=256, domain_channels=64, out_channels=num_fusion_channels)
    def forward(self, x_high, x_low, box_prompt):
        # x_high: 高分辨率输入（用于 SAM 编码器），x_low: 低分辨率输入（用于领域分支）
        # 先验分支：得到 SAM 编码器输出和位置编码（这里位置编码简单置零）
        prior_embeddings, image_pe = self.prior_branch(x_high)
        # 领域分支处理低分辨率图像
        domain_features = self.domain_branch(x_low)
        # 融合两路特征（先将领域分支特征上采样后拼接，再1x1卷积映射）
        fused_embeddings = self.fusion_module(prior_embeddings, domain_features)
        # 用 SAM 的 prompt_encoder 处理 box_prompt（注意 box_prompt 需归一化至 [0,1]，并转换成 SAM 要求的格式）
        # 这里假设 box_prompt 的尺寸与 x_high 一致，实际可能需要根据 SAM 的要求做调整
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(box_prompt)
        # 调用 SAM 的 mask_decoder，得到分割结果
        # mask_decoder 的输入：融合后的 image embeddings、位置编码、prompt 编码
        masks, iou_pred = self.sam_model.mask_decoder(fused_embeddings, image_pe, sparse_embeddings, dense_embeddings, multimask_output=False)
        # 返回单个掩膜（可根据需要选择第一个结果）
        return masks[:, 0, :, :]

# ---------------------------
# 自定义数据集：从文件夹中加载图像和标签
# 数据集下载后分别放到 images/ 和 masks/ 文件夹下，确保文件名对应
# ---------------------------
class OvarianTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))  # 或者jpg，根据数据实际情况
        self.masks_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform_img = transform_img
        self.transform_mask = transform_mask
    def __len__(self):
        return len(self.images_paths)
    def __getitem__(self, idx):
        img = Image.open(self.images_paths[idx]).convert("RGB")
        mask = Image.open(self.masks_paths[idx]).convert("L")
        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        # 这里我们以高分辨率（例如1024×1024）和低分辨率（例如256×256）分别提供给模型
        x_high = img  # 假设 img 已经 resize 到 1024×1024
        x_low = F.interpolate(x_high.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        # box_prompt 需要归一化到 [0,1]，假设给出全图框：[xmin, ymin, xmax, ymax]
        # 例如：若图像为1024×1024，则归一化后为 [0,0,1,1]
        box_prompt = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        # mask 转换为二值（0或1）
        mask = (mask > 128).float()
        return x_high, x_low, box_prompt, mask

# ---------------------------
# 计算像素级准确率的简单函数
# ---------------------------
def compute_accuracy(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    correct = (pred_bin == target).float().sum()
    total = torch.numel(target)
    return correct / total

# ---------------------------
# 训练函数：每个 epoch 验证并保存最佳模型
# ---------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=10, save_path='best_model.pth'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.99, weight_decay=1e-4)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
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
        
        # 验证阶段
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
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with Val Acc: {best_val_acc:.4f}")

# ---------------------------
# 主：加载 SAM 模型、构造数据集和数据加载器，开始训练
# ---------------------------
if __name__ == '__main__':
    # 设定 SAM 模型的 checkpoint 路径（下载官方权重后填写正确路径）
    sam_checkpoint = "path/to/sam_vit_b_01ec64.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载 SAM 模型（使用 vit_b 版本）
    sam_model = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam_model.to(device)
    sam_model.eval()  # 冻结 SAM 模型参数
    
    # 构造 MBA-Net 模型，内部会使用 SAM 模型的各个模块
    model = MBA_Net(sam_model, num_fusion_channels=256)
    
    # 定义数据变换
    transform_img = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    transform_mask = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    # 数据集路径，根据实际情况修改
    train_images_dir = "./data/train/images"
    train_masks_dir = "./data/train/masks"
    val_images_dir = "./data/val/images"
    val_masks_dir = "./data/val/masks"
    
    train_dataset = OvarianTumorDataset(train_images_dir, train_masks_dir, transform_img, transform_mask)
    val_dataset = OvarianTumorDataset(val_images_dir, val_masks_dir, transform_img, transform_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # 开始训练，保存最佳模型到 best_model.pth
    train_model(model, train_loader, val_loader, device, num_epochs=10, save_path='best_model.pth')
