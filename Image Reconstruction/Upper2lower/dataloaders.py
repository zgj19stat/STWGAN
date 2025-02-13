from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
import torch

class MNISTHalfDataset(Dataset):
    def __init__(self, train=True, total_samples=10000, class_distribution=None, m=100):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = datasets.MNIST(root='mnist_data/', train=train, transform=transform, download=True)
        
        if class_distribution is None:
            class_distribution = np.ones(10) / 10.0

        assert len(class_distribution) == 10, "class_distribution must have 10 elements"
        class_distribution = np.array(class_distribution)
        class_distribution = class_distribution / class_distribution.sum()

        targets = np.array(self.dataset.targets)
        sampled_indices = []
        num_samples_list = []
        
        for label in range(10):
            label_indices = np.where(targets == label)[0]
            num_samples = round(total_samples * class_distribution[label])
            num_samples_list.append(num_samples)
            sampled_label_indices = np.random.choice(label_indices, num_samples, replace=False)
            sampled_indices.extend(sampled_label_indices)

        np.random.shuffle(sampled_indices)
        self.dataset = Subset(self.dataset, sampled_indices)
        print(num_samples_list)
        self.Eta = torch.randn(total_samples, m)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        Eta = self.Eta[idx]
        upper_half = img[:, :14, :]
        lower_half = img[:, 14:, :]
        return upper_half, lower_half, Eta

if __name__ == "__main__":
    class_distribution = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.2, 0.1]
    train_loader = DataLoader(MNISTHalfDataset(train=True, total_samples=10000, class_distribution=class_distribution), batch_size=64, shuffle=True, num_workers=12)

