import argparse
import os
import torch
import matplotlib.pyplot as plt

from dataloaders import MNISTHalfDataset
from FNNmodels import Generator, Representation
from utils import set_seed

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train on simulationdata')
    parser.add_argument('--warmup', default=300, type=int, help='warmup')
    #test
    parser.add_argument('--result', default=None, type=str, required=True, help='result path')
    parser.add_argument('--n-images', default=10, type=int, help='number of showing images')
    parser.add_argument('--target-only', action='store_true', help='only train with target data')
    # Dimension
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--d', default=14*28, type=int)
    parser.add_argument('--q', default=14*28, type=int)
    parser.add_argument('--r', default=50, type=int)
    # Other
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    args = parser.parse_args()

    if args.target_only:
        args.r = args.q
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # if windows change 'mps' to 'cpu'
    set_seed(42) # pick the same test data

    generator = Generator(args.m, args.r, args.q).to(device)
    generator.load_state_dict(torch.load(args.result+'/generator.pth'))
    if not args.target_only:
        representation = Representation(args.d, args.r).to(device)
        representation = torch.load(args.result+'/epoch_{}_representation_weights.pth'.format(args.warmup))

    if not os.path.exists(args.result+'./test_S{}'.format(args.seed)):
        os.makedirs(args.result+'./test_S{}'.format(args.seed))

    # target_class_distribution = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    target_class_distribution = [0., 0., 0., 0., 0., 0.2, 0.2, 0.2, 0.2, 0.2] # D3
    test_loader = MNISTHalfDataset(train=False, total_samples=10, 
                                   class_distribution=target_class_distribution, m=args.m)

    with torch.no_grad():
        for i in range(args.n_images):
            upper_half, lower_half, Eta = test_loader[i]
            upper_half, lower_half, Eta = upper_half.unsqueeze(0).to(device), lower_half.unsqueeze(0).to(device), Eta.unsqueeze(0).to(device)
            
            set_seed(args.seed)
            Eta = torch.randn(1,args.m).to(device)
            if args.target_only:
                generated_lower_half = generator(Eta, upper_half.view(upper_half.size(0), -1))
            else:
                Z = representation(upper_half)
                generated_lower_half = generator(Eta, Z)

            background = - torch.ones_like(lower_half)
            upper_half_img = torch.cat((upper_half.cpu(), background.cpu()), dim=2)
            plt.imshow(upper_half_img.squeeze(), cmap='gray')
            plt.savefig(args.result+'./test_S{}/upper_half{}.png'.format(args.seed, i+1))
            plt.clf()

            fake_full_img = torch.cat((upper_half.cpu(), generated_lower_half.cpu()), dim=2)
            plt.imshow(fake_full_img.squeeze(), cmap='gray')
            plt.savefig(args.result+'./test_S{}/fake{}.png'.format(args.seed, i+1))
            plt.clf()

            true_full_img = torch.cat((upper_half.cpu(), lower_half.cpu()), dim=2)
            plt.imshow(true_full_img.squeeze(), cmap='gray')
            plt.savefig(args.result+'./test_S{}/true{}.png'.format(args.seed, i+1))
            plt.clf()