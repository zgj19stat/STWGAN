import torch
from .base_model import BaseModel
from . import networks
import ot
import time


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0)
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.target_only = opt.target_only
        self.loss_names = ['G_GAN', 'G_L1','D_real','D_fake']
        if opt.isTrain and opt.gan_mode == 'wgangp':
            self.loss_names.append('D_gradient')
        if not opt.target_only:
            self.loss_names.append('G_OT')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.gan_mode = opt.gan_mode
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
            if self.gan_mode == 'wgangp':
                self.critic_times = opt.critic_times
            self.ratio = opt.ratio
            if opt.OT_weight is not None:
                self.OT_weight = opt.OT_weight
            else:
                self.OT_weight = 1. - 1./(self.ratio+1.) # Adapative weight ready for multi-source setting
            self.OT_weight1 = opt.OT_weight1
            self.OT_weight2 = opt.OT_weight2
            
    def set_input(self, target_input, source_input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.target_real_A = target_input['A' if AtoB else 'B'].to(self.device)
        self.source_real_A = source_input['A' if AtoB else 'B'].to(self.device)
        self.target_real_B = target_input['B' if AtoB else 'A'].to(self.device)
        self.source_real_B = source_input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = target_input['A_paths' if AtoB else 'B_paths']
        if self.target_only:
            self.real_A = self.target_real_A
            self.real_B = self.target_real_B
        else:
            self.real_A = torch.concat([self.target_real_A, self.source_real_A], dim=0) # pool training
            self.real_B = torch.concat([self.target_real_B, self.source_real_B], dim=0) # pool training

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.target_fake_B, self.target_features = self.netG(self.target_real_A)  # G(A)
        if self.target_only:
            self.fake_B = self.target_fake_B
        else:
            self.source_fake_B, self.source_features = self.netG(self.source_real_A)  # G(source_A)
            self.fake_B = torch.concat([self.target_fake_B, self.source_fake_B], dim=0) # pool training

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        if self.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = self.loss_D_fake + self.loss_D_real
        elif self.gan_mode == 'wgangp':
            self.loss_D_gradient, _ = networks.cal_gradient_penalty(netD=self.netD, real_data=real_AB, fake_data=fake_AB, device=self.device, type='real')
            self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_gradient
        
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        if not self.target_only:
            # Third, For transfer, we make an OT regularization
            
            # self.cost_matrix = self.OT_weight1 * torch.sum(torch.abs(self.source_features.unsqueeze(1) - self.target_features.unsqueeze(0)), dim=2) + \
            #     self.OT_weight2 * torch.sum(torch.abs(self.source_real_B.unsqueeze(1)-self.target_real_B.unsqueeze(0)), dim=[2,3,4])
            if self.OT_weight1 is None and self.OT_weight2 is None:
                cost_matrix_A = torch.sum(torch.abs(self.source_features.unsqueeze(1) - self.target_features.unsqueeze(0)), dim=2)
                min_val_A = cost_matrix_A.min(dim=1, keepdim=True).values
                max_val_A = cost_matrix_A.max(dim=1, keepdim=True).values
                cost_matrix_A = (cost_matrix_A - min_val_A) / (max_val_A - min_val_A)
                
                cost_matrix_B = torch.sum(torch.abs(self.source_real_B.unsqueeze(1)-self.target_real_B.unsqueeze(0)), dim=[2,3,4])
                min_val_B = cost_matrix_B.min(dim=1, keepdim=True).values
                max_val_B = cost_matrix_B.max(dim=1, keepdim=True).values
                cost_matrix_B = (cost_matrix_B - min_val_B) / (max_val_B - min_val_B)
            
                self.cost_matrix = cost_matrix_A + cost_matrix_B
            else:
                self.cost_matrix = self.OT_weight1 * torch.sum(torch.abs(self.source_features.unsqueeze(1) - self.target_features.unsqueeze(0)), dim=2) + \
                    + self.OT_weight2 * torch.sum(torch.abs(self.source_real_B.unsqueeze(1)-self.target_real_B.unsqueeze(0)), dim=[2,3,4])
            
            with torch.no_grad():      
                self.gamma = torch.tensor(ot.emd(ot.unif(len(self.source_real_A)), ot.unif(len(self.target_real_A)), self.cost_matrix.detach().cpu().numpy()))
                self.gamma = self.gamma.to(dtype=torch.float32, device=self.device)
            self.loss_G_OT = self.OT_weight * torch.sum(self.gamma * self.cost_matrix)
        
        # combine loss and calculate gradients
        if self.target_only:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_OT
        self.loss_G.backward()

    def optimize_parameters(self, generator_update_flag):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        if (not self.gan_mode=='wgangp') or generator_update_flag:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # update G's weights
