import random
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import time
import ot
import matplotlib.pyplot as plt

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
        
        self.gp_weight = args.gp_weight
        self.OT_weight = args.OT_weight
        self.critic_iterations = args.critic_iterations
        self.target_only = args.target_only
        self.result = args.result
        self.bs = args.bs
        self.u = args.u
        self.d = args.d
        self.m = args.m
        self.q = args.q
        self.is_stage1 = is_stage1
        self.device = device

    def _critic_train_iteration(self, X, Y, Eta):
        """Perform one iteration of training for the discriminator (critic)"""
        # Get generated data
        Z = self.R(X).clone().detach()
        generated_Y = self.G(Eta, Z).clone().detach()
        
        # Calculate probabilities on real and generated data
        if self.is_stage1:
            d_real = self.D(torch.concat([X, Y], dim=3).view(X.size(0), -1))
            d_generated = self.D(torch.concat([X, generated_Y], dim=3).view(X.size(0), -1))
            gradient_penalty = self._gradient_penalty(torch.concat([X, Y], dim=3).view(X.size(0), -1))
        else:
            d_real = self.D(torch.concat([Z, Y.view(Y.size(0), -1)], dim=-1))
            d_generated = self.D(torch.concat([Z, generated_Y.view(generated_Y.size(0), -1)], dim=-1))
            gradient_penalty = self._gradient_penalty(torch.concat([Z, Y.view(Y.size(0), -1)], dim=-1))

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
        generated_Y = self.G(Eta, X.view(X.size(0), -1)).clone().detach()
        
        # Calculate probabilities on real and generated data
        d_real = self.D(torch.concat([X, Y], dim=3).view(X.size(0), -1))
        d_generated = self.D(torch.concat([X, generated_Y], dim=3).view(X.size(0), -1))
        gradient_penalty = self._gradient_penalty(torch.concat([X, Y], dim=3).view(X.size(0), -1))

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

    def _generator_train_iteration_stage1(self, source_X, source_Y, source_Eta, target_X, target_Y, target_Eta):
        """Perform one iteration of training for the generator in stage 1"""
        # Get source generated data
        source_Z = self.R(source_X)
        source_generated_Y = self.G(source_Eta, source_Z)
        d_source_generated = self.D(torch.concat([source_X, source_generated_Y], dim=3).view(source_X.size(0), -1))
        
        # Get target generated data
        target_Z = self.R(target_X)
        target_generated_Y = self.G(target_Eta, target_Z)
        d_target_generated = self.D(torch.concat([target_X, target_generated_Y], dim=3).view(target_X.size(0), -1))

        d_generated = torch.concat((d_source_generated, d_target_generated), dim=0)
        
        # Get OT regularization
        cost_matrix = torch.sum(torch.abs(source_Z[:, None, :] - target_Z[None, :, :]), dim=2) + \
                torch.sum(torch.abs(source_Y[:, None, :, :, :] - target_Y[None, :, :, :, :]), dim=(2, 3, 4))
        with torch.no_grad():
            gamma = torch.tensor(ot.emd(ot.unif(self.bs*self.u), ot.unif(self.bs), cost_matrix.detach().cpu().numpy())).to(self.device)
        OT_loss = self.OT_weight * torch.sum(gamma.clone().detach() * cost_matrix)

        # Calculate loss and optimize
        loss = d_generated.mean() + OT_loss
        
        self.R_opt.zero_grad()
        self.G_opt.zero_grad()
        loss.backward()
        self.R_opt.step()
        self.G_opt.step()

        # Record loss
        self.G_loss.update(d_generated.mean().item(), self.bs*(self.u+1))
        self.OT_loss.update(OT_loss.item(), 1)

    def _generator_train_iteration_stage2(self, X, Y, Eta):
        """Perform one iteration of training for the generator in stage 2"""
        # Get generated data
        Z = self.R(X).detach().clone()
        generated_Y = self.G(Eta, Z)

        # Calculate probabilities on generated data
        d_generated = self.D(torch.concat([Z, generated_Y.view(generated_Y.size(0), -1)], dim=-1))

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
        generated_Y = self.G(Eta, X.view(X.size(0), -1))

        # Calculate probabilities on generated data
        d_generated = self.D(torch.concat([X, generated_Y], dim=3).view(X.size(0), -1))

        # Create total loss and optimize
        self.G_opt.zero_grad()
        g_loss = d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.G_loss.update(g_loss.item(), len(X))

    def _train_epoch(self, epoch, source_train_loader, target_train_loader):
        """Train the WGAN for one epoch"""
        start_time = time.time()
        
        self.num_steps = 0
        for (source_X, source_Y, source_Eta), (target_X, target_Y, target_Eta) in zip(source_train_loader, target_train_loader):
            source_X, source_Y, source_Eta = source_X.to(self.device), source_Y.to(self.device), source_Eta.to(self.device)
            target_X, target_Y, target_Eta = target_X.to(self.device), target_Y.to(self.device), target_Eta.to(self.device)

            X = torch.concat([source_X, target_X], dim=0)
            Y = torch.concat([source_Y, target_Y], dim=0)
            Eta = torch.concat([source_Eta, target_Eta], dim=0)
            
            if self.target_only:
                self._critic_train_iteration_domain_only(target_X, target_Y, target_Eta)
                # Only update generator every |critic_iterations| iterations
                if self.num_steps % self.critic_iterations == 0:
                    self._generator_train_iteration_domain_only(target_X, target_Y, target_Eta)
            else:
                self._critic_train_iteration(X, Y, Eta)
                # Only update generator every |critic_iterations| iterations
                if self.num_steps % self.critic_iterations == 0:
                    if self.is_stage1:
                        self._generator_train_iteration_stage1(source_X, source_Y, source_Eta, target_X, target_Y, target_Eta)
                    else:
                        self._generator_train_iteration_stage2(X, Y, Eta)

            self.num_steps += 1

            with torch.no_grad():
                if (epoch+1) % 20== 0 and self.num_steps % 50 == 0:
                    vis_X, vis_Y, vis_Eta = target_X[0].unsqueeze(0).to(self.device), target_Y[0].unsqueeze(0).to(self.device), target_Eta[0].unsqueeze(0).to(self.device)

                    if self.target_only:
                        generated_Y = self.G(vis_Eta, vis_Y.view(vis_Y.size(0), -1)).clone().detach()
                    else:
                        vis_Z = self.R(vis_Y)
                        generated_Y = self.G(vis_Eta, vis_Z)

                    fake_full_img = torch.cat((vis_X.cpu(), generated_Y.cpu()), dim=3)
                    plt.imshow(fake_full_img.squeeze(), cmap='gray')
                    plt.savefig(self.result+'./train/epoch{}.png'.format(epoch+1))
                    plt.clf()
            
        train_time = time.time() - start_time

        print('train time: {}'.format(train_time))
        print("G: {} | D: {} | OT: {} | Gradient norm: {} ".format(
            self.G_loss.get_average(), self.D_loss.get_average(),
            self.OT_loss.get_average(), self.gradient_norm.get_average()))

    def train(self, source_train_loader, target_train_loader, epochs):
        """Train the GAN over multiple epochs"""

        for epoch in range(epochs):
            print("\nEpoch {}/{}".format(epoch+1, epochs))
            self.G_loss.reset()
            self.D_loss.reset()
            self.gradient_norm.reset()
            self.OT_loss.reset()
            
            self._train_epoch(epoch, source_train_loader, target_train_loader)
