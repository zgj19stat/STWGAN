import argparse
import time
import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloaders import MNISTHalfDataset
from FNNmodels import Generator, Discriminator, Representation
from utils import set_seed
from train import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on simulationdata')
    # train
    parser.add_argument('--warmup', default=300, type=int, help='warmup')
    parser.add_argument('--epochs', default=600, type=int, help='train epochs')
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=64, type=int, help='basic batch size')
    parser.add_argument('--critic-iterations', default=5, type=int, help='The discriminator is updated five times per one update of the generator')
    parser.add_argument('--OT-weight', default=0.1, type=float, help='weight for OT_loss')
    parser.add_argument('--gp-weight', default=10., type=float, help='weight for wgan-gp')
    parser.add_argument('--target-only', action='store_true', help='only train with target data')
    # dataset
    parser.add_argument('--num-workers', default=0, type=int, help='num_workers')
    parser.add_argument('--target-num-examples', default=5000, type=int, help='number of simulation train target data')
    parser.add_argument('--source-num-examples', default=50000, type=int, help='number of simulation train source data')
    parser.add_argument('--sources', default=5, type=int, help='number of sources')
    parser.add_argument('--reliable-sources', default=1, type=int, help='number of reliable sources') # Oracle Algorithm
    # Dimension
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--d', default=14*28, type=int)
    parser.add_argument('--q', default=14*28, type=int)
    parser.add_argument('--r', default=50, type=int)
    # other
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--result', default=None, type=str, required=True, help='result path')
    
    args = parser.parse_args()

    if args.target_only:
        args.r = args.q
    args.u = args.source_num_examples // args.target_num_examples
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # if windows change 'mps' to 'cpu'

    if not os.path.exists(args.result+'./train'):
        os.makedirs(args.result+'./train')

    with open(args.result + '/training_record.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args])

    # Loading Dataset
    set_seed(42)
    target_class_distribution = [0., 0., 0., 0., 0., 0.2, 0.2, 0.2, 0.2, 0.2]
    target_train_loader = DataLoader(MNISTHalfDataset(train=True, total_samples=args.target_num_examples, 
                                                      class_distribution=target_class_distribution, m=args.m), 
                                     batch_size=args.bs, shuffle=True, num_workers=args.num_workers, drop_last=True)
    source_class_distribution = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    source_train_loader = DataLoader(MNISTHalfDataset(train=True, total_samples=args.source_num_examples, 
                                                      class_distribution=source_class_distribution, m=args.m), 
                                     batch_size=args.bs * args.u, shuffle=True, num_workers=args.num_workers, drop_last=True)

    if args.seed != 42 :
        set_seed(args.seed)
    
    # Stage 1
    is_stage1 = True

    # Create Model
    generator = Generator(args.m, args.r, args.q).to(device)
    discriminator = Discriminator(args.d, args.q).to(device)
    representation = Representation(args.d, args.r).to(device)
    print(generator)
    print(discriminator)
    print(representation)

    # Initialize optimizers
    G_optimizer = optim.RMSprop(generator.parameters(), lr=args.lr)
    D_optimizer = optim.RMSprop(discriminator.parameters(), lr=args.lr)
    R_optimizer = optim.RMSprop(representation.parameters(), lr=args.lr)

    # Train model
    trainer = Trainer(generator, discriminator, representation, G_optimizer, D_optimizer, R_optimizer, args,
                      is_stage1=is_stage1, device=device)
    trainer.train(source_train_loader, target_train_loader, args.warmup)

    if not args.target_only:
        
        torch.save(representation, args.result+'/epoch_{}_representation_weights.pth'.format(args.warmup))
        
        # Stage 2
        is_stage1 = False
        
        # Create Model
        generator = Generator(args.m, args.r, args.q).to(device)
        discriminator = Discriminator(args.r, args.q).to(device)
        representation = Representation(args.d, args.r).to(device)
        print(generator)
        print(discriminator)
        print(representation)

        representation = torch.load(args.result+'/epoch_{}_representation_weights.pth'.format(args.warmup))
        
        # Initialize optimizers
        G_optimizer = optim.RMSprop(generator.parameters(), lr=args.lr)
        D_optimizer = optim.RMSprop(discriminator.parameters(), lr=args.lr)
        R_optimizer = optim.RMSprop(representation.parameters(), lr=args.lr)
        # Train model
        trainer = Trainer(generator, discriminator, representation, G_optimizer, D_optimizer, R_optimizer, args,
                        is_stage1=is_stage1, device=device)
        trainer.train(source_train_loader, target_train_loader, args.epochs-args.warmup)

    torch.save(generator.state_dict(), args.result + '/generator.pth')
