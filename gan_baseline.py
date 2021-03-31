import torch, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
import torch_mimicry as mmc, argparse, os, sys, imgutils, re
from torch_mimicry.nets import sngan
from options import get_options
from torch import autograd

from torch_mimicry.nets.wgan_gp import wgan_gp_base
from torch_mimicry.nets.wgan_gp.wgan_gp_resblocks import DBlockOptimized, DBlock, GBlock

def main():
    parser = argparse.ArgumentParser(description='WGAN-GP baseline entry point')
    parser.add_argument('options_choice',
                type = str,
                help = 'Name of options set (ignored for vis)')
    parser.add_argument('outdir',
                type = str,
                help = 'Path to output folder')
    parser.add_argument('--eval', 
                action = 'store_true')
    parser.add_argument('--vis', 
                action = 'store_true')
    parser.add_argument('--model_dir_vis', 
                type = str, 
                help = 'folder with "log" and model file')
    parser.add_argument('--name_vis',
                type = str,
                help = 'name of image output')
    parser.add_argument('--n_fid_samples', 
                default = 50000, 
                type = int)
    args = parser.parse_args()

    if not args.vis:
        options = get_options( args.options_choice )

        OUTDIR = os.path.join('/gpfs-volume/genren-output/', args.outdir) #SNGAN-Baseline/'
        LOGDIR = os.path.join(OUTDIR, 'log/')
        NUM_STEPS = 50000 # 100000 # For training
        B = 64
        NW = 4
        NUM_CRITIC_STEPS_PER_GEN_STEP = 5

    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    def _get_latest_checkpoint(ckpt_dir):
        def _get_step_number(k):
            search = re.search(r'(\d+)_steps', k)
            if search:
                return int(search.groups()[0])
            else:
                return -float('inf')
        if not os.path.exists(ckpt_dir):
            return None
        files = os.listdir(ckpt_dir)
        if len(files) == 0:
            return None
        ckpt_file = max(files, key=lambda x: _get_step_number(x))
        return os.path.join(ckpt_dir, ckpt_file)

    #### For visualization ####

    if args.vis:
        
        assert not args.model_dir_vis is None
        assert not args.name_vis      is None

        ncols = 15
        B = ncols * 10
        netG = WGANGPGenerator64().to(device)

        netG_ckpt_file = os.path.join(args.model_dir_vis, 'generator.statedict.pt') 
        #'log', 'checkpoints', 'netG')
        #netG_ckpt_file = _get_latest_checkpoint(netG_ckpt_dir)  # can be None
        print('Loading from', netG_ckpt_file)
        netG.load_state_dict( torch.load(netG_ckpt_file) )
        #global_step_G = netG.restore_checkpoint(ckpt_file=netG_ckpt_file)
        #print('Global step:', global_step_G)
        print('\tDone')
        z = torch.randn(B, netG.nz).to(device)
        imgs = netG(z)
        #alpha  = imgs[:,-1,:,:]
        #imgsna = imgs[:,0:3,:,:]
        #imgsna = (imgsna + 1.0) / 2.0 # ( ((imgsa + 1.0) / 2.0)*255 ).byte()
        #imgs = torch.cat( (imgsna, alpha), dim = 1 )

        # Save images
        from imgutils import imswrite_t
        os.makedirs(args.outdir, exist_ok = True)
        print('Saving to', args.outdir)
        outpath = os.path.join(args.outdir, 'GAN_BASELINE-%s.png' % args.name_vis)
        imswrite_t(imgs, outpath, ncols = ncols, no_grid = True)

        sys.exit(0)

    ############################ If running evaluation ############################
    if args.eval:

        # Path to real images
        path_to_real_imgs = options['TEST_img_data_dir']
        assert os.path.isdir(path_to_real_imgs)

        netG = WGANGPGenerator64().to(device)

        netG_ckpt_dir = os.path.join(LOGDIR, 'checkpoints', 'netG')
        netG_ckpt_file = _get_latest_checkpoint(netG_ckpt_dir)  # can be None
        print('Loading from', netG_ckpt_file)
        global_step_G = netG.restore_checkpoint(ckpt_file=netG_ckpt_file)
        print('\tDone')

        import torch_fidelity
        NUM_SAMPLES = args.n_fid_samples
        class GanBaselineDataset(torch.utils.data.Dataset):
            """docstring for GanBaselineDataset"""
            def __init__(self, model, device, precompute = None):
                super(GanBaselineDataset, self).__init__()
                self.model = model
                self.do_precompute = not precompute is None
                self.remove_alpha = True # TODO setting this to false will fail
                if self.do_precompute:
                    with torch.no_grad():
                        assert type(precompute) is int
                        batch_size = 128
                        n_batches = ( (precompute // batch_size) + 
                                      (0 if precompute % batch_size == 0 else 1) )
                        print('Precomputing %d batches of size %d' % (n_batches, batch_size))
                        self.pre_data = []
                        for i in range(n_batches):
                            if i % 10 == 0:
                                print('\t', i, '/', n_batches)
                            z = torch.randn(batch_size, model.nz).to(device)
                            imgs = model(z)
                            if self.remove_alpha:
                                imgsa = imgs[:,0:3,:,:]
                            imgs = ( ((imgsa + 1.0) / 2.0)*255 ).byte()
                            if i == 0:
                                print("Pre-min/max", imgsa.min(-1)[0].min(-1)[0], 
                                        imgsa.max(-1)[0].max(-1)[0]) 
                                print("Post-min/max", imgs.min(-1)[0].min(-1)[0], 
                                        imgs.max(-1)[0].max(-1)[0]) 
                            self.pre_data.append(imgs.cpu())
                        print('Combining...')
                        self.pre_data = torch.cat(self.pre_data, dim = 0)[ 0 : precompute ]
                        print('\tObtained a preloaded set of shape', self.pre_data.shape)

            def __len__(self):
                return len(self.pre_data) if self.do_precompute else None 

            def __getitem__(self, i):
                if self.do_precompute:
                    return self.pre_data[i]

            # TODO inefficient... it's already batched!

        print('Building fake dataset')
        fake_dataset = GanBaselineDataset(model = netG, device = device, precompute = args.n_fid_samples)
        print('\tFinished building fake dataset')

        # Calculating metrics
        # Each input can be either a string (path to images, registered input), or a Dataset instance
        USE_FID = True
        USE_KID = True if args.n_fid_samples > 1000 else False
        USE_IS  = True
        from torch_fidelity import calculate_metrics
        metrics_dict = calculate_metrics(fake_dataset, path_to_real_imgs, cuda=True, 
                                         isc=USE_IS, fid=USE_FID, kid=USE_KID, 
                                         verbose=True)

        print('METRICS')
        print('For logdir =', LOGDIR)
        print(metrics_dict)
        sys.exit(0)

    ###############################################################################

    print('Reading dataset')
    img_dataset = imgutils.SinglePreloadedDirImageDataset(
                                          options['img_data_dir'],
                                          use_alpha          = options['use_alpha'],
                                          resize             = options['img_size'],
                                          load_gt_file       = options['USE_GT'],
                                          num_fixed_to_store = options['B_imgs'],
                                          take_subset        = options['img_data_subset'] )
    dataloader = img_dataset.get_dataloader(B, NW=NW, shuffle=True)

    #dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
    #dataloader = torch.utils.data.DataLoader(
    #    dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define models and optimizers
    print('Building GANs and optimizer')
    netG = WGANGPGenerator64().to(device)
    #netG = sngan.SNGANGenerator64().to(device)
    netD = WGANGPDiscriminator64().to(device)
    #netD = WGANGP_net(inchannels = 4)
    #netD = sngan.SNGANDiscriminator64().to(device)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    batchtest = next(dataloader.__iter__()).to(device)
    z_batch = torch.randn(B, 128).to(device)
    Gout = netG(z_batch)
    Dout = netD(batchtest)

    # Start training
    print('Entering trainer')
    #trainer = mmc.training.Trainer(
    trainer = MyTrainer(
        netD = netD,
        netG = netG,
        optD = optD,
        optG = optG,
        n_dis = NUM_CRITIC_STEPS_PER_GEN_STEP,
        num_steps = NUM_STEPS,
        lr_decay = 'linear',
        dataloader = dataloader,
        log_dir = LOGDIR, # './log/example',
        device = device)
    trainer.train()

    print('Saving generator')
    torch.save(netG.state_dict(), os.path.join(OUTDIR, 'generator.statedict.pt'))
    torch.save(netG, os.path.join(OUTDIR, 'generator.model.pt'))

#-------------------------------------------------------------------------------------#


class WGANGPGenerator32(wgan_gp_base.WGANGPBaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, outchannels=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        print('Building WGANGP generator with cout = %d' % outchannels)

        # Build the layers
        self.l1     = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        #self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b6     = nn.BatchNorm2d(self.ngf >> 3)
        self.c6     = nn.Conv2d(self.ngf >> 3, outchannels, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        #print('rr', x.shape)
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        #h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))

        #print('hh', h.shape)

        return h

class WGANGPGenerator64(wgan_gp_base.WGANGPBaseGenerator):
    r"""
    ResNet backbone generator for WGAN-GP.
    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, outchannels=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        print('Building WGANGP generator with cout = %d' % outchannels)

        # Build the layers
        self.l1     = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.block5 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b6     = nn.BatchNorm2d(self.ngf >> 4)
        self.c6     = nn.Conv2d(self.ngf >> 4, outchannels, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.
        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        #print('rr', x.shape)
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.b6(h)
        h = self.activation(h)
        h = torch.tanh(self.c6(h))

        #print('hh', h.shape)

        return h

class WGANGPDiscriminator64(wgan_gp_base.WGANGPBaseDiscriminator):
    r"""
    ResNet backbone discriminator for WGAN-GP.
    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        gp_scale (float): Lamda parameter for gradient penalty.        
    """
    def __init__(self, ndf=1024, inchannels=4, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        print('Building WGANGP critic with Cin = %d' % inchannels)

        # Build layers
        self.block1 = DBlockOptimized(inchannels, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.l6 = nn.Linear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l6.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        #print(x.shape,'x')
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)

        # Global average pooling
        h = torch.mean(h, dim=(2, 3))  # WGAN uses mean pooling
        output = self.l6(h)

        return output

    def compute_gradient_penalty_loss(self,
                                      real_images,
                                      fake_images,
                                      gp_scale=10.0):
        r"""
        Computes gradient penalty loss, as based on:
        https://github.com/jalola/improved-wgan-pytorch/blob/master/gan_train.py
        
        Args:
            real_images (Tensor): A batch of real images of shape (N, 3, H, W).
            fake_images (Tensor): A batch of fake images of shape (N, 3, H, W).
            gp_scale (float): Gradient penalty lamda parameter.
        Returns:
            Tensor: Scalar gradient penalty loss.
        """
        # Obtain parameters
        N, _, H, W = real_images.shape
        device = real_images.device

        # Randomly sample some alpha between 0 and 1 for interpolation
        # where alpha is of the same shape for elementwise multiplication.
        alpha = torch.rand(N, 1)
        alpha = alpha.expand(N, int(real_images.nelement() / N)).contiguous()
        alpha = alpha.view(N, 4, H, W)
        alpha = alpha.to(device)

        # Obtain interpolates on line between real/fake images.
        interpolates = alpha * real_images.detach() \
            + ((1 - alpha) * fake_images.detach())
        interpolates = interpolates.to(device)
        interpolates.requires_grad_(True)

        # Get gradients of interpolates
        disc_interpolates = self.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(device),
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute GP loss
        gradient_penalty = (
            (gradients.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gradient_penalty

class MyTrainer(mmc.training.Trainer):
    def _fetch_data(self, iter_dataloader):
        """
        Fetches the next set of data and refresh the iterator when it is exhausted.
        Follows python EAFP, so no iterator.hasNext() is used.
        """
        try:
            real_batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(self.dataloader)
            real_batch = next(iter_dataloader)

        #real_batch = (real_batch[0].to(self.device),
        #              real_batch[1].to(self.device))
	
        real_batch = (real_batch.to(self.device), None)

        return iter_dataloader, real_batch


#-------------------------#
if __name__ == "__main__":
        main()
#-------------------------#



#
