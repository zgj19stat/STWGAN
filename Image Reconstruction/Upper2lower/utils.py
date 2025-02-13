import torch
import random
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def get_average(self):
        return self.average

    def update(self, value, num):
        self.value = value
        self.sum += value * num
        self.count += num
        self.average = self.sum / self.count

    def __repr__(self):
        return f"{self.get_average():.4f}"
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)