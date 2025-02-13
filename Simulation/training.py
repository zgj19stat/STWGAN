import random
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import time
import ot

import csv

from utils import AverageMeter

class Trainer():
    def __init__(self, generator, discriminator, representation, G_optimizer, D_optimizer, R_optimizer, args,
                 is_stage1=True, device='cuda'):
        # Initialize the Trainer object with generator, discriminator, optimizers, and hyperparameters
        # Set the generator, discriminator, and optimizers
        # Initialize lists to store losses during training
        # Initialize variables for tracking training progress
        self.G = generator
        self.G_opt = G_optimizer
        self.D = discriminator
        self.D_opt = D_optimizer
        self.R = representation
        self.R_opt = R_optimizer
        
        self.G_loss = AverageMeter()
        self.D_loss = AverageMeter()
        self.gradient_norm = AverageMeter()
        self.OT_loss = AverageMeter()
        self.OT_loss_domain = [AverageMeter() for _ in range(args.reliable_sources)]
        
        self.valid_num = args.valid_num_examples
        self.gp_weight = args.gp_weight
        self.OT_weight = args.OT_weight
        self.critic_iterations = args.critic_iterations
        self.bs = args.bs
        self.j = args.j
        self.u = args.u
        self.d = args.d
        self.m = args.m
        self.q = args.q
        self.is_stage1 = is_stage1
        self.target_only = args.target_only
        self.pretrained = args.pretrained
        self.datasets = args.datasets
        self.sources = args.sources
        self.reliable_sources = args.reliable_sources
        self.device = device

    def _critic_train_iteration(self, X, Y, Eta):
        """Perform one iteration of training for the discriminator (critic)"""
        X, Y, Eta = X.view(-1, self.d), Y.view(-1, self.q), Eta.view(-1, self.m)
        # Get generated data
        Z = self.R(X).clone().detach()
        generated_Y = self.G(Eta, Z).clone().detach()
        
        # Calculate probabilities on real and generated data
        if self.is_stage1:
            d_real = self.D(torch.concat([X, Y], dim=-1))
            d_generated = self.D(torch.concat([X, generated_Y], dim=-1))
            gradient_penalty = self._gradient_penalty(torch.concat([X, Y], dim=-1))
        else:
            d_real = self.D(torch.concat([Z, Y], dim=-1))
            d_generated = self.D(torch.concat([Z, generated_Y], dim=-1))
            gradient_penalty = self._gradient_penalty(torch.concat([Z, Y], dim=-1))

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = - d_generated.mean() + d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        # Record loss
        self.D_loss.update(d_loss.item(), len(X))

    def _critic_train_iteration_domain_only(self, X, Y, Eta):
        """Perform one iteration of training for the discriminator (critic)"""
        # Get generated data
        generated_Y = self.G(Eta, X).clone().detach()
        
        # Calculate probabilities on real and generated data
        d_real = self.D(torch.concat([X, Y], dim=-1))
        d_generated = self.D(torch.concat([X, generated_Y], dim=-1))
        gradient_penalty = self._gradient_penalty(torch.concat([X, Y], dim=-1))

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = - d_generated.mean() + d_real.mean() + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        # Record loss
        self.D_loss.update(d_loss.item(), len(X))
        
    def _gradient_penalty(self, real_data):
        """Calculate gradient penalty for Wasserstein GAN-GP"""
        # Calculate probability of examples
        real_data = Variable(real_data, requires_grad=True)
        prob = self.D(real_data)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob, inputs=real_data,
                               grad_outputs=torch.ones(prob.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(len(real_data), -1)
        self.gradient_norm.update(gradients.norm(2, dim=1).mean().item(), len(real_data))

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _generator_train_iteration_stage1(self, sources_X, sources_Y, sources_Eta, target_X, target_Y, target_Eta):
        """Perform one iteration of training for the generator in stage 1"""
        # Get source generated data
        sources_Z, d_sources_generated = [], []
        for i in range(self.reliable_sources):
            source_Z = self.R(sources_X[i])
            generated_Y = self.G(sources_Eta[i], source_Z)
            sources_Z.append(source_Z)
            d_sources_generated.append(self.D(torch.concat([sources_X[i], generated_Y], dim=-1)))

        sources_Z = torch.stack(sources_Z) # size:(sources, bs*u, r)
        d_sources_generated = torch.stack(d_sources_generated).view(-1, 1) # size:(sources*bs*u, 1)
        
        # Get target generated data
        target_Z = self.R(target_X)
        target_generated_Y = self.G(target_Eta, target_Z)
        d_target_generated = self.D(torch.concat([target_X, target_generated_Y], dim=-1))

        d_generated = torch.concat((d_sources_generated, d_target_generated), dim=0)
        
        # Get OT regularization
        OT_loss = []
        for i in range(self.reliable_sources):
            cost_matrix = torch.sum(torch.abs(sources_Z[i][:, None, :] - target_Z[None, :, :]), dim=2) + \
                torch.sum(torch.abs(sources_Y[i][:, None, :] - target_Y[None, :, :]), dim=2)
            with torch.no_grad():
                gamma = torch.tensor(ot.emd(ot.unif(self.bs*self.u), ot.unif(self.bs), cost_matrix.detach().cpu().numpy())).to(self.device)
            OT_loss.append(self.OT_weight * torch.sum(gamma.clone().detach() * cost_matrix))
        OT_loss = torch.stack(OT_loss).view(-1)

        # Calculate loss and optimize
        loss = d_generated.mean()  + OT_loss.sum()
        
        self.R_opt.zero_grad()
        self.G_opt.zero_grad()
        loss.backward()
        self.R_opt.step()
        self.G_opt.step()

        # Record loss
        self.G_loss.update(d_generated.mean().item(), self.bs*(self.u*self.reliable_sources+1))
        self.OT_loss.update(OT_loss.sum().item(), 1)
        for i in range(self.reliable_sources):
            self.OT_loss_domain[i].update(OT_loss[i].item(), 1)

    def _generator_train_iteration_stage2(self, X, Y, Eta):
        """Perform one iteration of training for the generator in stage 2"""
        # Get generated data
        Z = self.R(X).detach().clone()
        generated_Y = self.G(Eta, Z)

        # Calculate probabilities on generated data
        d_generated = self.D(torch.concat([Z, generated_Y], dim=-1))

        # Create total loss and optimize
        self.G_opt.zero_grad()
        g_loss = d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss.update(g_loss.item(), len(X))

    def _generator_train_iteration_domain_only(self, X, Y, Eta):
        """Perform one iteration of training for the generator in stage 2"""
        # Get generated data
        generated_Y = self.G(Eta, X)

        # Calculate probabilities on generated data
        d_generated = self.D(torch.concat([X, generated_Y], dim=-1))

        # Create total loss and optimize
        self.G_opt.zero_grad()
        g_loss = d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss.update(g_loss.item(), len(X))

    def _train_data_iter(self, data):
        sources_X, sources_Y, sources_Eta = data[0], data[1], data[2]
        target_X, target_Y, target_Eta = data[3], data[4], data[5]
        
        # source_num >> target_num
        sources_num = len(sources_X[0])
        target_num = len(target_X)
        
        sources_indices = [list(range(sources_num)) for _ in range(self.sources)]
        for indices in sources_indices:
            random.shuffle(indices)
        target_indices = list(range(target_num))
        random.shuffle(target_indices)
        
        # every domain give half of samples
        for i in range(0, target_num, self.bs):
            if i + self.bs > target_num:
                break
            sources_batch_indices = torch.tensor([sources_indices[j][i:i+self.bs*self.u] for j in range(self.sources)])
            target_batch_indices = torch.tensor(target_indices[i: min(i + self.bs, target_num)])
            sources_batch_X = torch.gather(sources_X, 1, sources_batch_indices.unsqueeze(-1).expand(-1, -1, sources_X.size(2)))
            sources_batch_Y = torch.gather(sources_Y, 1, sources_batch_indices.unsqueeze(-1).expand(-1, -1, sources_Y.size(2)))
            sources_batch_Eta = torch.gather(sources_Eta, 1, sources_batch_indices.unsqueeze(-1).expand(-1, -1, sources_Eta.size(2)))
            
            yield sources_batch_X, sources_batch_Y, sources_batch_Eta, \
                target_X[target_batch_indices], target_Y[target_batch_indices], target_Eta[target_batch_indices]

    def _train_epoch(self, epoch, train_data):
        """Train the WGAN for one epoch"""
        self.G.train()
        self.D.train()
        self.R.train()
        
        start_time = time.time()
        
        self.num_steps = 0
        for sources_X, sources_Y, sources_Eta, target_X, target_Y, target_Eta in self._train_data_iter(train_data):
            sources_X, sources_Y, sources_Eta = sources_X.to(self.device), sources_Y.to(self.device), sources_Eta.to(self.device)
            target_X, target_Y, target_Eta = target_X.to(self.device), target_Y.to(self.device), target_Eta.to(self.device)

            X = torch.concat([sources_X[:self.reliable_sources].view(-1, self.d), target_X], dim=0)
            Y = torch.concat([sources_Y[:self.reliable_sources].view(-1, self.q), target_Y], dim=0)
            Eta = torch.concat([sources_Eta[:self.reliable_sources].view(-1, self.m), target_Eta], dim=0)
            
            if self.target_only:
                self._critic_train_iteration_domain_only(target_X, target_Y, target_Eta)
                # Only update generator every |critic_iterations| iterations
                if self.num_steps % self.critic_iterations == 0:
                    self._generator_train_iteration_domain_only(target_X, target_Y, target_Eta)
            elif self.pretrained:
                if self.is_stage1:
                    self._critic_train_iteration_domain_only(sources_X[:self.reliable_sources].view(-1, self.d), sources_Y[:self.reliable_sources].view(-1, self.q), sources_Eta[:self.reliable_sources].view(-1, self.m))
                    # Only update generator every |critic_iterations| iterations
                    if self.num_steps % self.critic_iterations == 0:
                        self._generator_train_iteration_domain_only(sources_X[:self.reliable_sources].view(-1, self.d), sources_Y[:self.reliable_sources].view(-1, self.q), sources_Eta[:self.reliable_sources].view(-1, self.m))
                else:
                    self._critic_train_iteration_domain_only(target_X, target_Y, target_Eta)
                    # Only update generator every |critic_iterations| iterations
                    if self.num_steps % self.critic_iterations == 0:
                        self._generator_train_iteration_domain_only(target_X, target_Y, target_Eta)
            else:
                self._critic_train_iteration(X, Y, Eta)
                # Only update generator every |critic_iterations| iterations
                if self.num_steps % self.critic_iterations == 0:
                    if self.is_stage1:
                        self._generator_train_iteration_stage1(sources_X, sources_Y, sources_Eta, target_X, target_Y, target_Eta)
                    else:
                        self._generator_train_iteration_stage2(X, Y, Eta)
                    
            self.num_steps += 1
            
        train_time = time.time() - start_time

        print('train time: {}'.format(train_time))
        print("G: {} | D: {} | OT: {} | Gradient norm: {} ".format(
            self.G_loss.get_average(), self.D_loss.get_average(),
            self.OT_loss.get_average(), self.gradient_norm.get_average()))
    
    def _valid_epoch(self, epoch, valid_data, valid_Eta):
        """Valid the Generator in one epoch"""
        self.G.eval()
        self.D.eval()
        self.R.eval()
        
        valid_X, valid_Y_mean, valid_Y_sd, valid_Eta = valid_data[0].to(self.device), valid_data[1].to(self.device), valid_data[2].to(self.device), valid_Eta.to(self.device)
        valid_Y = torch.zeros(len(valid_X), self.j).to(self.device)
        
        start_time = time.time()
        valid_Z = self.R(valid_X)
        for i in range(self.j):
            Eta = valid_Eta[i].repeat(self.valid_num, 1)
            if self.target_only:
                valid_Y[:, i] = self.G(Eta, valid_X).detach().view(-1, 1).squeeze()
            else:
                valid_Y[:, i] = self.G(Eta, valid_Z).detach().view(-1, 1).squeeze()
        valid_time = time.time() - start_time
        
        valid_Y_mean_hat = torch.mean(valid_Y, dim=-1)
        valid_Y_sd_hat = torch.std(valid_Y, dim=-1)
        mse_mean = torch.mean((valid_Y_mean_hat-valid_Y_mean.view(-1))**2)
        mse_sd = torch.mean((valid_Y_sd_hat-valid_Y_sd.view(-1))**2)
        
        print('valid time: {}'.format(valid_time))
        print('mse_mean: {} | mse_sd: {}'.format(mse_mean, mse_sd))
        
        return mse_mean, mse_sd, valid_Y_mean_hat, valid_Y_mean.view(-1)

    def train(self, train_data, valid_data, valid_Eta, test_data, test_Eta, epochs, args):
        """Train the GAN over multiple epochs"""
        mse_mean_record = torch.zeros(epochs)
        mse_sd_record = torch.zeros(epochs)
        latest_mse_mean, latest_mse_sd = torch.tensor(0.), torch.tensor(0.)

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            self.G_loss.reset()
            self.D_loss.reset()
            self.gradient_norm.reset()
            self.OT_loss.reset()
            for i in range(self.reliable_sources):
                self.OT_loss_domain[i].reset()
            
            self._train_epoch(epoch, train_data)

            if not args.is_test:
                mse_mean, mse_sd, valid_Y_mean_hat, valid_Y_mean = self._valid_epoch(epoch, valid_data, valid_Eta)

                mse_mean_record[epoch], mse_sd_record[epoch] = mse_mean, mse_sd
                if epoch>=20:
                    latest_mse_mean = mse_mean_record[epoch-20:epoch].mean()
                    latest_mse_sd = mse_sd_record[epoch-20:epoch].mean()
                    
                if self.is_stage1:
                    if self.target_only:
                        with open(args.result + '/training_record.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([epoch+1,latest_mse_mean.item(), latest_mse_sd.item(), mse_mean.item(), mse_sd.item(), 
                                            valid_Y_mean_hat[:5].detach().cpu().numpy(), valid_Y_mean[:5].detach().cpu().numpy()])
                    else:
                        with open(args.result + '/training_record1.csv', mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([self.OT_loss_domain[i].get_average() for i in range(self.reliable_sources)])
                else:
                    with open(args.result + '/training_record2.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch+1,latest_mse_mean.item(), latest_mse_sd.item(), mse_mean.item(), mse_sd.item(), 
                                        valid_Y_mean_hat[:5].detach().cpu().numpy(), valid_Y_mean[:5].detach().cpu().numpy()])

                print("latest mse mean and sd: {}, {}".format(latest_mse_mean, latest_mse_sd))
        
        # test data
        if args.is_test:
            test_mse_mean, test_mse_sd, _, _ = self._valid_epoch(epoch, test_data, test_Eta)
            if self.is_stage1 and self.target_only:
                with open(args.result + '/training_record.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([test_mse_mean.item(), test_mse_sd.item()])
            elif not self.is_stage1:
                with open(args.result + '/training_record2.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([test_mse_mean.item(), test_mse_sd.item()])
                
        return mse_mean_record, mse_sd_record
