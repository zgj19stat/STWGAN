real_images_folder = './T1-TO/true'  # 真实图像的文件夹
fake_images_folder = './T1-TO/fake'  # 生成图像的文件夹

import os
import numpy as np
import torch
from torchvision import datasets, transforms, models
from scipy.linalg import sqrtm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def calculate_fid(real_images_folder, fake_images_folder, batch_size=32, image_size=28):
    # Define image transformation with 299x299 resize (InceptionV3 input size)
    transform = transforms.Compose([
        transforms.Resize(299),  # Resize images to 299x299 for InceptionV3
        transforms.Grayscale(3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Custom dataset to load PNG images from folder
    class CustomImageDataset(torch.utils.data.Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = folder_path
            self.transform = transform
            self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.png')]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
            if self.transform:
                image = self.transform(image)
            return image, 0  # Return dummy label (0), since labels are not needed for FID calculation

    # Load images using the custom dataset
    real_images = CustomImageDataset(real_images_folder, transform=transform)
    fake_images = CustomImageDataset(fake_images_folder, transform=transform)
    
    real_loader = DataLoader(real_images, batch_size=batch_size, shuffle=False)
    fake_loader = DataLoader(fake_images, batch_size=batch_size, shuffle=False)
    
    # Load the InceptionV3 model pre-trained
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    # Get the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move the model to the appropriate device
    inception_model.to(device)

    def get_features(loader):
        features = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)  # Move images to the same device as the model
                features.append(inception_model(images))
        return torch.cat(features, dim=0).cpu().numpy()

    # Get features for real and fake images
    real_features = get_features(real_loader)
    fake_features = get_features(fake_loader)

    # Calculate mean and covariance for real and fake features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    mu_diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    # Numerical stability fix for the square root of covariance matrix
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = mu_diff.dot(mu_diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

# Call the function to calculate FID
fid_score = calculate_fid(real_images_folder, fake_images_folder)
print(f"FID Score: {fid_score}")
