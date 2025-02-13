import torch
import torch.optim as optim
import argparse
from torch.optim.lr_scheduler import MultiStepLR

from training import Trainer
from dataloaders import synthetic_data
from models import Generator, Discriminator, Representation
from utils import set_seed

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train on simulationdata')
    # train
    parser.add_argument('--warmup', default=300, type=int, help='warmup')
    parser.add_argument('--epochs', default=600, type=int, help='train epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=64, type=int, help='basic batch size')
    parser.add_argument('--critic-iterations', default=5, type=int, help='The discriminator is updated five times per one update of the generator')
    parser.add_argument('--OT-weight', default=0.1, type=float, help='weight for OT_loss')
    parser.add_argument('--gp-weight', default=10., type=float, help='weight for wgan-gp')
    parser.add_argument('--target-only', action='store_true', help='only train with target data')
    parser.add_argument('--pretrained', action='store_true', help='pretrained or not')
    # valid and test
    parser.add_argument('--j', default=10000, type=int, help='sample number')
    parser.add_argument('--valid-num-examples', default=20000, type=int, help='number of simulation valid data and test data')
    parser.add_argument('--is-test', action='store_true', help='stop validation')
    # dataset
    parser.add_argument('--target-num-examples', default=10000, type=int, help='number of simulation train target data')
    parser.add_argument('--source-num-examples', default=40000, type=int, help='number of simulation train source data')
    parser.add_argument('--sources', default=5, type=int, help='number of sources')
    parser.add_argument('--reliable-sources', default=1, type=int, help='number of reliable sources')
    parser.add_argument('--datasets', default=None, type=str, required=True, help='Generator Mode')
    # other
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--result', default=None, type=str, required=True, help='result path')
    
    args = parser.parse_args()

    # Define dimension
    if args.datasets == 'M1' or args.datasets == 'M2':
        if args.target_only or args.pretrained:
            args.m, args.d, args.q, args.r = 3, 100, 1, 100
        else:
            args.m, args.d, args.q, args.r = 3, 100, 1, 10
    elif args.datasets == 'M3':
        if args.target_only or args.pretrained:
            args.m, args.d, args.q, args.r = 1, 100, 1, 100
        else:
            args.m, args.d, args.q, args.r = 1, 100, 1, 5
    args.u = args.source_num_examples // args.target_num_examples # ratio for sample train
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') # if windows change 'mps' to 'cpu'
        
    # ready for record
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    
    # Loading Dataset
    set_seed(42)
    train_data, valid_data, valid_Eta, test_data, test_Eta = synthetic_data(args)

    if args.seed != 42 :
        set_seed(args.seed)
    
    # Stage 1
    is_stage1 = True
    if args.target_only or args.pretrained:
        with open(args.result + '/training_record.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args])
            writer.writerow(['epoch','lavalid_mse_mean', 'lavalid_mse_sd','mse_mean', 'mse_sd', 'valid_Y_mean_hat', 'valid_Y_mean'])
    else:
        with open(args.result + '/training_record1.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args])
            writer.writerow(['domain 1-wasserstein distance'])
    
    # Create Model
    generator = Generator(args.m, args.r, args.q).to(device)
    discriminator = Discriminator(args.d, args.q).to(device)
    representation = Representation(args.d, args.r).to(device)
    print(generator)
    print(discriminator)
    print(representation)
    
    # Initialize optimizers
    G_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(.9, .99))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.9, .99))
    R_optimizer = optim.Adam(representation.parameters(), lr=args.lr, betas=(.9, .99))

    # Train model
    trainer = Trainer(generator, discriminator, representation, G_optimizer, D_optimizer, R_optimizer, args,
                      is_stage1=is_stage1, device=device)
    mse_mean_record, mse_sd_record = trainer.train(train_data, valid_data, valid_Eta, test_data, test_Eta, args.warmup, args)

    if not args.target_only:
        
        torch.save(representation, args.result+'/epoch_{}_representation_weights.pth'.format(args.warmup))
        if args.pretrained:
            torch.save(generator, args.result+'/epoch_{}_generator_weights.pth'.format(args.warmup))
            torch.save(discriminator, args.result+'/epoch_{}_discriminator_weights.pth'.format(args.warmup))
        
        # Stage 2
        is_stage1 = False
        with open(args.result + '/training_record2.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args])
            writer.writerow(['epoch','lavalid_mse_mean', 'lavalid_mse_sd','mse_mean', 'mse_sd', 'valid_Y_mean_hat', 'valid_Y_mean'])
        
        # Create Model
        generator = Generator(args.m, args.r, args.q).to(device)
        discriminator = Discriminator(args.r, args.q).to(device)
        representation = Representation(args.d, args.r).to(device)
        print(generator)
        print(discriminator)
        print(representation)

        representation = torch.load(args.result+'/epoch_{}_representation_weights.pth'.format(args.warmup))
        if args.pretrained:
            generator = torch.load(args.result+'/epoch_{}_generator_weights.pth'.format(args.warmup))
            discriminator = torch.load(args.result+'/epoch_{}_discriminator_weights.pth'.format(args.warmup))
        
        # Initialize optimizers
        G_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(.9, .99))
        D_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.9, .99))
        R_optimizer = optim.Adam(representation.parameters(), lr=args.lr, betas=(.9, .99))

        # Train model
        trainer = Trainer(generator, discriminator, representation, G_optimizer, D_optimizer, R_optimizer, args,
                        is_stage1=is_stage1, device=device)
        mse_mean_record, mse_sd_record = trainer.train(train_data, valid_data, valid_Eta, test_data, test_Eta, args.epochs-args.warmup, args)

    # draw pic
    if args.target_only:
        epoch_record = torch.arange(0, args.warmup)
    else:
        epoch_record = torch.arange(0, args.epochs-args.warmup)

    if not args.is_test:
        plt.plot(epoch_record, mse_mean_record, label='mse_mean')
        plt.plot(epoch_record, mse_sd_record, label='mse_sd')
        plt.xlabel('epoch')
        plt.ylabel('mean & sd')
        plt.title('mse of mean and sd - Epoch {}'.format(args.epochs))
        plt.legend()
        plt.savefig(args.result + '/epoch_{}_mse_of_mean_and_sd.png')
        # clean for next pic
        plt.clf()
    