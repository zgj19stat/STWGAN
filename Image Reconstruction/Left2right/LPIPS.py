import os
import torch
import lpips
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# 设置文件夹路径
real_images_folder = './T1-TO/true'  # 真实图像的文件夹
fake_images_folder = './T1-TO/fake'  # 生成图像的文件夹

# 定义图像的转换操作
transform = transforms.Compose([
    transforms.Resize(299),  # 适应网络输入大小，LPIPS一般使用299x299
    transforms.Grayscale(3),  # MNIST是单通道图像，转为3通道
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化，跟InceptionV3一致
])

# 自定义数据集类
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # 转换为RGB图像
        if self.transform:
            image = self.transform(image)
        return image

# 创建数据加载器
real_images = CustomImageDataset(real_images_folder, transform=transform)
fake_images = CustomImageDataset(fake_images_folder, transform=transform)

real_loader = DataLoader(real_images, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_images, batch_size=32, shuffle=False)

# 加载LPIPS模型
lpips_model = lpips.LPIPS(net='alex')  # 使用AlexNet作为基础网络
lpips_model.eval()

# 获取设备（GPU如果可用，否则使用CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model.to(device)

# 计算LPIPS的平均值
def calculate_lpips(real_loader, fake_loader):
    lpips_scores = []

    with torch.no_grad():
        for real_images, fake_images in zip(real_loader, fake_loader):
            real_images = real_images.to(device)  # 将数据移到GPU上
            fake_images = fake_images.to(device)

            # 计算LPIPS
            lpips_score = lpips_model(real_images, fake_images)
            lpips_scores.append(lpips_score.cpu().numpy())  # 保存计算结果

    return np.concatenate(lpips_scores, axis=0)

# 执行LPIPS计算
lpips_scores = calculate_lpips(real_loader, fake_loader)

# 打印LPIPS值的平均值
average_lpips = np.mean(lpips_scores)
print(f"Average LPIPS: {average_lpips}")
