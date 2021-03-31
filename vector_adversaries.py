import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, numpy.random as npr
import logging, math, mathutils, utils
from torch.autograd import Variable
#from networks.networks import LBA_stack, Unfolder, Reshaper
from networks.networks import *
from networks.wgan_helpers import *
import kornia


class VaeKL(nn.Module):
    def __init__(self):
        super(VaeKL, self).__init__()

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())

class Wasserstein3dPoseAdversary(nn.Module):
    """
    Computes the Sinkhorn EMD between a buffer of translations and rotations.
    """
    def __init__(self, buffer_size):
        super(Wasserstein3dPoseAdversary, self).__init__()
        logging.info('Constructing Sinkhorn-based pose distribution matcher')
        self.dim_t = 3
        self.t_loss_scaling = np.pi # t in ~[-0.5,0.5], unlike r (in [0,pi]).
        # Buffers for storing pose hypotheses
        self.buffer_size = buffer_size
        r_buffer = torch.eye(3,3).view(1,1,3,3).expand(-1,buffer_size,-1,-1).clone()
        t_buffer = torch.zeros(1, buffer_size, self.dim_t)
        self.register_buffer('r_buffer', r_buffer)
        self.register_buffer('t_buffer', t_buffer)
        # Initialize EMD loss for t
        import geomloss
        self.t_emd = geomloss.SamplesLoss(loss='sinkhorn', 
                                          p=2, #2, 
                                          blur=0.05, # recommended for the unit cube
                                          diameter=2.0) # ~max distance
        # Initialize EMD loss for r
        self.r_emd = geomloss.SamplesLoss(loss='sinkhorn', 
                                          p=2, #2, 
                                          blur=0.05,
                                          cost=RotationSpaceCostMatrix(),
                                          diameter=3.141) # ~max distance
        logging.info('\tDone')

    def forward(self, for_gen, R_fake=None, R_real=None, t_fake=None, t_real=None):
        ### For the generator: return the chamfer loss to the reals
        if for_gen: return self.compute_loss_for_generator(R_fake=R_fake, t_fake=t_fake)
        ### Training the "critic" simply means updating the buffer
        else:       self.update_buffers(R_real=R_real, t_real=t_real)

    def compute_loss_for_generator(self, R_fake, t_fake):
        """ Sinkhorn distance between input and buffer """
        assert len(t_fake.shape) == 2     and t_fake.shape[-1]  == 3
        assert R_fake.shape[-2:] == (3,3) and len(R_fake.shape) == 3
        # Translation loss (inputs: 1 x B x 3)
        tloss = self.t_emd(self.t_buffer, t_fake.unsqueeze(0)).mean()
        # Rotation loss (inputs: 1 x B x 3 x 3 -> 1 x B x 9)
        rloss = self.r_emd(self.r_buffer.view(1,self.buffer_size,9), 
                           R_fake.view(1,R_fake.shape[0],9)
                           ).mean()
        # Scale the losses and return
        return rloss + (tloss * self.t_loss_scaling)

    def update_buffers(self, R_real, t_real):
        """ Notice: r,t are treated independently. """
        assert len(t_real.shape) == 2 and t_real.shape[-1] == 3
        r = R_real.detach()
        t = t_real.detach()
        n = r.shape[0]
        self.r_buffer[0, npr.choice(self.buffer_size, size=(n), replace=False), :, :] = r
        self.t_buffer[0, npr.choice(self.buffer_size, size=(n), replace=False), :] = t
    
class RotationSpaceCostMatrix(nn.Module):
    """
    Used to compute EMD cost matrix
    Batch size is always 1 -> matching the buffer distribution (sample) and the 
    """
    def __init__(self):
        super(RotationSpaceCostMatrix, self).__init__()
        from graphicsutils import MinAngleComposedRotationLoss
        self.R_distance = MinAngleComposedRotationLoss()
        
    def forward(self, R1, R2):
        # Input shapes R: [1 x] B x 9 (a single point cloud of matrices)
        s1 = R1.shape[1]
        s2 = R2.shape[1]
        R1 = R1.view(s1, 3, 3)
        R2 = R2.view(s2, 3, 3)
        R1_expanded = R1.unsqueeze(1).expand(-1, s2, -1, -1).reshape(s1*s2, 3, 3)
        R2_expanded = R2.unsqueeze(0).expand(s1, -1, -1, -1).reshape(s1*s2, 3, 3)
        dists_all = self.R_distance(R1_expanded, R2_expanded, mean_out=False).view(s1, s2)
        return dists_all.unsqueeze(0)

class ChamferBasedIndependent3dPoseAdv(nn.Module):
    """
    Use the Chamfer distance to match the distributions of r and t.
    """
    def __init__(self, buffer_size):
        super(ChamferBasedIndependent3dPoseAdv, self).__init__()
        self.dim_t = 3
        self.t_loss_scaling = np.pi # t in ~[-0.5,0.5], unlike r.
        # Buffers for storing pose hypotheses
        self.buffer_size = buffer_size
        t_buffer = torch.zeros(1, buffer_size, self.dim_t)
        self.register_buffer('r_buffer', r_buffer)
        self.register_buffer('t_buffer', t_buffer)
        # Initialize Chamfer loss for t
        from losses import chamfer_loss_object
        self.chamdist = chamfer_loss_object()
        # Initialize rotation loss for R
        from graphicsutils import MinAngleComposedRotationLoss
        self.R_distance = MinAngleComposedRotationLoss()

    def forward(self, for_gen, R_fake=None, R_real=None, t_fake=None, t_real=None):
        ### For the generator: return the chamfer loss to the reals
        if for_gen: return self.compute_loss_for_generator(R_fake=R_fake, t_fake=t_fake)
        ### Training the "critic" simply means updating the buffer
        else:       self.update_buffers(R_real=R_real, t_real=t_real)

    def compute_loss_for_generator(self, R_fake, t_fake):
        """ Chamfer distance between input and buffer """
        assert len(t_fake.shape) == 2 and t_fake.shape[-1] == 3
        # Translation loss
        td1, td2, _, _ = self.chamdist(self.t_buffer, t_fake.unsqueeze(0))
        tloss = (td1 + 1e-5).sqrt().mean() + (td2 + 1e-5).sqrt().mean()
        # Rotation loss
        nF = R_fake.shape[0]
        nR = self.buffer_size
        real_expanded = self.r_buffer.unsqueeze(1).expand(-1, nF, -1, -1).reshape(nR*nF, 3, 3)
        fake_expanded =        R_fake.unsqueeze(0).expand(nR, -1, -1, -1).reshape(nR*nF, 3, 3)
        dists_all = self.R_distance(fake_expanded, real_expanded, mean_out=False).view(nR, nF)
        rd1 = dists_all.min(dim=0)[0].mean()
        rd2 = dists_all.min(dim=1)[0].mean()
        rloss = rd1 + rd2
        # Scale the losses and return
        return rloss + (tloss * self.t_loss_scaling)

    def update_buffers(self, R_real, t_real):
        """ Notice: r,t are treated independently. """
        assert len(t_real.shape) == 2 and t_real.shape[-1] == 3
        r = R_real.detach()
        t = t_real.detach()
        n = r.shape[0]
        self.r_buffer[npr.choice(self.buffer_size, size=(n), replace=False), :, :] = r
        self.t_buffer[0, npr.choice(self.buffer_size, size=(n), replace=False), :] = t

class PoseRrtAdversary(nn.Module):
    def __init__(self, dim_r):
        super(PoseRrtAdversary, self).__init__()
        self.dim_r = dim_r
        self.dim_t = 3
        # Three adversaries: one for each of r, R, and t
        # (1) adversary on vector rotation
        self.f_r = LBA_stack( sizes = [self.dim_r, 64, 32, 1],
                               norm_type = 'spectral',
                               act_type = 'lrelu',
                               end_with_lin = True )
        # (2) adversary on vector translation
        self.f_t = LBA_stack( sizes = [self.dim_t, 32, 16, 1],
                               norm_type = 'spectral',
                               act_type = 'lrelu',
                               end_with_lin = True )
        # MSE loss
        self.mse = torch.nn.MSELoss()
        # Combiner
        self.f = Avg2(self.f_r, self.f_t, self.dim_r, self.dim_t)

    def forward(self, for_gen, v_fake, v_real=None):
        if for_gen:
            return self.compute_loss_for_generator(x = v_fake) 
        else:
            device    = v_fake.device
            valid     = Variable(torch.Tensor(v_real.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake      = Variable(torch.Tensor(v_fake.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( self.f(v_real), valid)
            fake_loss = self.mse( self.f(v_fake.detach()), fake)
            d_loss    = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, x):
        return (self.f(x) - 1.0).pow(2).mean() * 0.5

class Avg2(nn.Module):
    def __init__(self, m1, m2, s1, s2):
        super(Avg2, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1 
        self.s2 = s2
    def forward(self, x):
        x1 = x[:, 0 : self.s1]
        x2 = x[:, self.s1 : ]
        return 0.5 * (self.m1(x1) + self.m2(x2))

class GlobalHistogramFixedSamplesAB(nn.Module):
    """
    Expects the input to be 2D, e.g. the UV from YUV
    """

    def __init__(self, 
                 num_template_vertices, 
                 num_fixed_samples, 
                 batch_size,
                 lower_bound            = 0.0, # in colours space
                 upper_bound            = 1.0,
                 min_sigma              = None # 0.1
                ):
        super(GlobalHistogramFixedSamplesAB, self).__init__()
        logging.info('\tInitializing global histogram critic')
        self.nV  = num_template_vertices
        self.nS  = num_fixed_samples
        self.npd = round( math.pow(self.nS, 1.0 / 2.0) ) # num samples per dimension
        self.L   = lower_bound
        self.U   = upper_bound
        self.bd  = (self.U - self.L) / (self.npd - 1) # distance between bins
        M = torch.meshgrid(torch.linspace(self.L, self.U, self.npd), 
                           torch.linspace(self.L, self.U, self.npd) )
        # Points in RGB space to sample at
        self.B = batch_size
        #print(M[0].shape, self.npd, torch.stack(M, -1).shape)
        S = torch.stack(M, dim = -1).view(self.npd**2, 2).unsqueeze(0).expand(self.B,-1,-1).contiguous() # B x nS x 3
        self.register_buffer('S', S)
        # Precompute some bandwidth related quantities
        # Silverman's rule with an isotropic bandwidth matrix
        # Kernel:
        #   K_H(x) = (2 pi)^(-d/2) |H|^(-1/2) exp(-x^T H^-1 x / 2)
        #          = c_f sigma^(-d) exp(c_n sigma^-2 d^2)
        #   where d = x^Tx, H = c^2 sigma^2 I_d, c given by Silverman's rule
        self.d = 2.0 # dimensionality of space
        # Bandwidth matrix H = c^2 sigma^2 I_3
        self.c  = (4.0 / (self.d + 2))**(1 / (self.d + 4)) * self.nV**(-1 / (self.d + 4))
        # Frontal coefficient (outside the exp): c_f * sigma^(-d)
        self.cf = ( (2.0 * math.pi)**(-self.d / 2) ) / (self.c ** self.d)
        # Inner coefficient (inside the exp): exp( c_n dist_squared / sigma^2 )
        self.cn = (-1.0 / (2.0 * self.c**2))
        # Minimum standard deviation (linearly related to the bandwith) allowed
        self.min_sigma = self.bd if min_sigma is None else min_sigma
        # Should we detach sigma?
        self.detach_sigma = False
        # Logging
        logging.info('\tL = %.3f, U = %.3f, nS = %d, npd = %d, BD = %.3f, min_sigma = %.3f', 
                     self.L, self.U, self.nS, self.npd, self.bd, self.min_sigma )
        logging.info('\tDetaching sigma? ' + str(self.detach_sigma))

    def forward(self, T):
        assert len(T.shape) == 3 and T.shape[1 : ] == (self.nV, 2) # B x nV x 3
        B = T.shape[0]
        # Average std deviations (across channels) per texture vector
        sigma = T.std(dim = 1).mean(dim = -1).clamp(min = self.min_sigma).view(B, 1, 1) # Shape: B x 1 x 1
        if self.detach_sigma: 
            sigma = sigma.detach()
        # Squared pairwise distances between all the fixed samples and the input textures
        D = mathutils.batched_cdist_l2sq(self.S, T)
        # Compute the exponential term in the Gaussian KDE
        E = torch.exp(self.cn * D / (sigma**2))
        # KDE density estimate at each fixed position in colour space
        K_H = self.cf * ( sigma.squeeze(-1) ** (-self.d) ) * E.mean(dim = -1) # B x nS 
        return K_H / K_H.sum(dim = 1, keepdim = True).clamp(1e-5)


class GlobalHistogramFixedSamples(nn.Module):

    def __init__(self, 
                 num_template_vertices, 
                 num_fixed_samples, 
                 #fixed_bandwidth        = None,
                 lower_bound            = 0.0, # in colours space
                 upper_bound            = 1.0,
                 min_sigma              = 0.1,
                ):
        super(GlobalHistogramFixedSamples, self).__init__()
        logging.info('\tInitializing global histogram critic')
        self.nV  = num_template_vertices
        self.nS  = num_fixed_samples
        self.npd = round( math.pow(self.nS, 1.0 / 3.0) ) # num samples per dimension
        self.L   = lower_bound
        self.U   = upper_bound
        self.bd  = (self.U - self.L) / (self.npd - 1) # distance between bins
        logging.info('\tL = %.3f, U = %.3f, nS = %d, npd = %d, BD = %.3f', 
                self.L, self.U, self.nS, self.npd, self.bd )
        M = torch.meshgrid(torch.linspace(self.L, self.U, self.npd), 
                           torch.linspace(self.L, self.U, self.npd),
                           torch.linspace(self.L, self.U, self.npd) )
        # Points in RGB space to sample at
        S = torch.stack(M, dim = -1).view(self.npd**3, 3).unsqueeze(0).contiguous() # 1 x nS x 3
        self.register_buffer('S', S)
        # Precompute some bandwidth related quantities
        # Silverman's rule with an isotropic bandwidth matrix
        self.d = 3 # dimensionality of space
        # Bandwidth matrix H = c^2 sigma^2 I_3
        self.c  = (4.0 / (self.d + 2))**(1 / (self.d + 4)) * self.nV**(-1 / (self.d + 4))
        # Frontal coefficient (outside the exp): c_f * sigma^(-3)
        self.cf = ((2.0 * math.pi)**(-1.5)) / (self.c**3)
        # Inner coefficient (inside the exp): exp( c_n dist_squared / sigma^2 )
        self.cn = (-1.0 / (2.0 * self.c**2))
        # Minimum standard deviation (linear related to the bandwith) allowed
        self.min_sigma = min_sigma

    def forward(self, T):
        assert len(T.shape) == 3 and T.shape[1 : ] == (self.nV, 3) # B x nV x 3
        B = T.shape[0]
        # Average std deviations (across channels) per texture vector
        sigma = T.std(dim = 1).mean(dim = -1).clamp(min = self.min_sigma).view(B, 1, 1) # Shape: B x 1 x 1
        # Fixed colour samples
        S = self.S.expand(B, -1, -1).contiguous() # B x nS x 3
        # Squared pairwise distances between all the fixed samples and the input textures
        D = mathutils.batched_cdist_l2sq(S, T)
        #print('kk', S.shape, T.shape)
        #D = torch.cdist(S, T.contiguous(), p = 2) # B x nS x nV
        # Compute the exponential term in the Gaussian KDE
        E = torch.exp(self.cn * D / (sigma**2))
        # KDE density estimate at each fixed position in colour space
        K_H = self.cf * (sigma.squeeze(-1)**(-3.0)) * E.mean(dim = -1) # B x nS 
        return K_H / K_H.sum(dim = 1, keepdim = True).clamp(1e-5)

class VectorAdversaryLinWGANGP(nn.Module):
    def __init__(self, dim_input, hidden_sizes, wgan_gp_pen_weight, drift_mag_weight):
        super(VectorAdversaryLinWGANGP, self).__init__()
        self.D = dim_input
        self.sizes = [ dim_input ] + list(hidden_sizes) + [ 1 ]
        self.f = LBA_stack( sizes = self.sizes,
                               norm_type = 'layer',
                               act_type = 'lrelu',
                               end_with_lin = True )
        self.wgan_gp_pen_weight = wgan_gp_pen_weight
        self.drift_mag_weight = drift_mag_weight # As in progressively growing GANs
        self.learned = True
        self.expected_dim_len = 2

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False):
        if len(v_fake.shape) == 3:
            v_fake = v_fake.view(-1, self.D)
            v_real = None if v_real is None else v_real.view(-1, self.D)
        assert len(v_fake.shape) == self.expected_dim_len, str(v_fake.shape) #+ " " + str(v_real.shape)
        if for_gen:
            return self.compute_loss_for_generator(
                                x              = v_fake, 
                                prob_weights   = prob_weights, 
                                max_probs_only = max_probs_only)
        else:
            assert v_fake.shape == v_real.shape
            gp = compute_gradient_penalty_loss_vecs(self.f, v_real, v_fake,
                                                    self.wgan_gp_pen_weight)
            output_real = self.f(v_real)
            output_fake = self.f(v_fake)
            loss = wasserstein_loss_dis(output_real, 
                                        output_fake, 
                                        drift_mag_weight = self.drift_mag_weight)
            return loss + gp

    def compute_loss_for_generator(self, x, prob_weights=None, max_probs_only=False):
        return wasserstein_loss_gen( self.f(x) )

class TextureGraphAdversaryWGANGP(nn.Module):
    def __init__(self, per_node_input_dim, hidden_dims, GSB, wgan_gp_pen_weight, drift_mag_weight):
        super(TextureGraphAdversaryWGANGP, self).__init__()
        from networks.DglGCN import GcnPatchCritic
        self.f = GcnPatchCritic(in_dim                = per_node_input_dim, 
                                hidden_dims           = hidden_dims, 
                                graph_structure_batch = GSB)
        self.wgan_gp_pen_weight = wgan_gp_pen_weight
        self.drift_mag_weight   = drift_mag_weight # As in progressively growing GANs
        self.learned = True

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False):
        assert len(v_fake.shape) == 3
        if for_gen:
            return self.compute_loss_for_generator(
                                x              = v_fake, 
                                prob_weights   = prob_weights, 
                                max_probs_only = max_probs_only)
        else:
            assert v_fake.shape == v_real.shape
            gp = compute_gradient_penalty_loss_texture(self.f, v_real, v_fake,
                                                       self.wgan_gp_pen_weight)
            output_real = self.f(v_real)
            output_fake = self.f(v_fake)
            loss = wasserstein_loss_dis(output_real, 
                                        output_fake, 
                                        drift_mag_weight = self.drift_mag_weight)
            return loss + gp

    def compute_loss_for_generator(self, x, prob_weights=None, max_probs_only=False):
        return wasserstein_loss_gen( self.f(x) )

class ConditionalUvTextureImagePlus2dHistoCritic(nn.Module):
    def __init__(self, 
                 ## Histogram critic settings
                 num_template_vertices, 
                 hidden_sizes, 
                 num_fixed_samples = None,
                 batch_size = None,
                 ## UV texture image 
                 inchannels = None,
                 critic_type = None,
                 ## Shared parameters
                 wgan_gp_pen_weight = None, 
                 drift_mag_weight = None, 
                 ## Meta-parameters
                 texture_critic_type = None,
                 options = None,
                 ):
        super(ConditionalUvTextureImagePlus2dHistoCritic, self).__init__()

        assert texture_critic_type in ['histo', 'uvimage', 'uvimage+histo']
        self.texture_critic_type = texture_critic_type
        self.use_histo      = (self.texture_critic_type in ['histo',   'uvimage+histo'])
        self.use_img_critic = (self.texture_critic_type in ['uvimage', 'uvimage+histo'])
        self.use_both       = (self.texture_critic_type == 'uvimage+histo')

        # Histogram critic
        if texture_critic_type in ['histo', 'uvimage+histo']:
            if options['histo_type'] == 'uv':
                options['num_histo_samples_per_dim'] = 25
                self.histo_critic = GlobalHistogram2DFixedSamplesLinWGANGP(
                                num_template_vertices = num_template_vertices,
                                hidden_sizes          = hidden_sizes,
                                wgan_gp_pen_weight    = wgan_gp_pen_weight,
                                drift_mag_weight      = 0.01,
                                num_fixed_samples     = options['num_histo_samples_per_dim'] ** 2,
                                batch_size            = batch_size, )
            if options['histo_type'] == 'rgb': 
                options['num_histo_samples_per_dim'] = 10
                self.histo_critic = GlobalHistogramFixedSamplesLinWGANGP(
                                 num_template_vertices = num_template_vertices, 
                                 hidden_sizes          = hidden_sizes, 
                                 wgan_gp_pen_weight    = wgan_gp_pen_weight, 
                                 drift_mag_weight      = 0.01, 
                                 num_fixed_samples     = options['num_histo_samples_per_dim']**3)

            # self.histo_critic = GlobalHistogram2DFixedSamplesLinWGANGP(num_template_vertices, 
            #                                                            hidden_sizes, 
            #                                                            wgan_gp_pen_weight, 
            #                                                            drift_mag_weight, 
            #                                                            num_fixed_samples,
            #                                                            batch_size, )
        else:
            self.histo_critic = None 

        # Texture image critic
        if texture_critic_type in ['uvimage', 'uvimage+histo']:

            ### Pre-processing for conv critic
            nV = num_template_vertices
            outdim_m_reduction = 8
            self.tex_img_size  = 64
            self.M_reducer = nn.Sequential(
                                Unfolder(),
                                LBA_stack_to_reshape(
                                    [ 3 * nV, 256, outdim_m_reduction ],
                                    #[ 3 * nV, 512, 128, outdim_m_reduction ],
                                    [ outdim_m_reduction ],
                                    #[ 3 * nV, 3 * nV // 2, 128 ],
                                    #[ 128 ],
                                    norm_type = 'bn',
                                    end_with_lin = False) # Next network starts with linear/conv
                             )
            # Generator for the image form of the reduced template
            #from gan_baseline import WGANGPGenerator64
            M_c = outdim_m_reduction # 3
            #self.M_to_img_mapping = WGANGPGenerator64(nz = 128, outchannels = M_c)

            ### Conv critic
            from image_adversaries import ImageAdversarySimpleWganGp 
            self.tex_img_critic = ImageAdversarySimpleWganGp(lambda_weight    = wgan_gp_pen_weight, 
                                                             drift_mag_weight = drift_mag_weight,
                                                             inchannels       = inchannels + M_c,
                                                             critic_type      = critic_type )
        else:
            self.tex_img_critic = None

    def M_to_img_mapping(self, M_reduced):
        B, dimMR = M_reduced.shape
        return M_reduced.view(B, dimMR, 1, 1).expand(-1, -1, self.tex_img_size, self.tex_img_size)

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False, 
                return_histogram=False):
        """
        Feed in the tuple (nodal_texture_vector, texture_image)
        """
        if v_real is None: v_real = (None, None, None)

        # Histogram critic
        if self.use_histo:
            histo_out = self.histo_critic(for_gen, v_fake[0], v_real[0], prob_weights, max_probs_only, 
                                          return_histogram = return_histogram)
        # Image critic
        if self.use_img_critic:
            fake_tex_img = v_fake[1]
            fake_M_pe    = v_fake[2]
            fakeci = torch.cat(
                        (fake_tex_img, self.M_to_img_mapping(self.M_reducer(fake_M_pe))), 
                        dim = 1
                     )

            real_tex_img = v_real[1]
            real_M_pe    = v_real[2]
            if real_tex_img is None or real_M_pe is None:
                realci = None
            else:
                realci = torch.cat(
                            (real_tex_img, self.M_to_img_mapping(self.M_reducer(real_M_pe))), 
                            dim = 1
                        )

            image_out = self.tex_img_critic(for_gen, fakeci, realci, prob_weights, max_probs_only)

        # Combine output
        if   self.use_both:
            if return_histogram:
                return (histo_out[0] + image_out) / 2.0, histo_out[1] 
            return (histo_out + image_out) / 2.0
        elif self.use_histo:
            return histo_out
        elif self.use_img_critic:
            return image_out

class UvTextureImagePlus2dHistoCritic(nn.Module):
    def __init__(self, 
                 ## Histogram critic settings
                 num_template_vertices, 
                 hidden_sizes, 
                 num_fixed_samples = None,
                 batch_size = None,
                 ## UV texture image 
                 inchannels = None,
                 critic_type = None, # This is the conv critic arch
                 ## Shared parameters
                 wgan_gp_pen_weight = None, 
                 drift_mag_weight = None, 
                 ## Meta-parameters
                 texture_critic_type = None, # This is how/whether uv is with or without histo
                 options = None,
                 ):
        super(UvTextureImagePlus2dHistoCritic, self).__init__()

        assert texture_critic_type in ['histo', 'uvimage', 'uvimage+histo']
        self.texture_critic_type = texture_critic_type
        self.use_histo      = (self.texture_critic_type in ['histo',   'uvimage+histo'])
        self.use_img_critic = (self.texture_critic_type in ['uvimage', 'uvimage+histo'])
        self.use_both       = (self.texture_critic_type == 'uvimage+histo')

        # Histogram critic
        if texture_critic_type in ['histo', 'uvimage+histo']:
            if options['histo_type'] == 'uv':
                options['num_histo_samples_per_dim'] = 25
                self.histo_critic = GlobalHistogram2DFixedSamplesLinWGANGP(
                                num_template_vertices = num_template_vertices,
                                hidden_sizes          = hidden_sizes,
                                wgan_gp_pen_weight    = wgan_gp_pen_weight,
                                drift_mag_weight      = 0.01,
                                num_fixed_samples     = options['num_histo_samples_per_dim'] ** 2,
                                batch_size            = batch_size, )
            if options['histo_type'] == 'rgb': 
                options['num_histo_samples_per_dim'] = 10
                self.histo_critic = GlobalHistogramFixedSamplesLinWGANGP(
                                 num_template_vertices = num_template_vertices, 
                                 hidden_sizes          = hidden_sizes, 
                                 wgan_gp_pen_weight    = wgan_gp_pen_weight, 
                                 drift_mag_weight      = 0.01, 
                                 num_fixed_samples     = options['num_histo_samples_per_dim']**3)

            # self.histo_critic = GlobalHistogram2DFixedSamplesLinWGANGP(num_template_vertices, 
            #                                                            hidden_sizes, 
            #                                                            wgan_gp_pen_weight, 
            #                                                            drift_mag_weight, 
            #                                                            num_fixed_samples,
            #                                                            batch_size, )
        else:
            self.histo_critic = None 

        # Texture image critic
        if texture_critic_type in ['uvimage', 'uvimage+histo']:
            from image_adversaries import ImageAdversarySimpleWganGp 
            self.tex_img_critic = ImageAdversarySimpleWganGp(lambda_weight    = wgan_gp_pen_weight, 
                                                             drift_mag_weight = drift_mag_weight,
                                                             inchannels       = inchannels,
                                                             critic_type      = critic_type )
        else:
            self.tex_img_critic = None

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False, 
                return_histogram=False):
        """
        Feed in the tuple (nodal_texture_vector, texture_image)
        """
        if v_real is None: v_real = (None, None)

        # Histogram critic
        if self.use_histo:
            histo_out = self.histo_critic(for_gen, v_fake[0], v_real[0], prob_weights, max_probs_only, 
                                          return_histogram = return_histogram)
        # Image critic
        if self.use_img_critic:
            image_out = self.tex_img_critic(for_gen, v_fake[1], v_real[1], prob_weights, max_probs_only)

        # Combine output
        if   self.use_both:
            if return_histogram:
                return (histo_out[0] + image_out) / 2.0, histo_out[1] 
            return (histo_out + image_out) / 2.0
        elif self.use_histo:
            return histo_out
        elif self.use_img_critic:
            return image_out

class UvHistogram(nn.Module):
    def __init__(self, nV, nS, B):
        super(UvHistogram, self).__init__()
        self.rgb2yuv = kornia.color.RgbToYuv()
        self.colour_uv_normer = ColourUvNormer()
        self.h = GlobalHistogramFixedSamplesAB(num_template_vertices = nV,
                                               num_fixed_samples     = nS,
                                               batch_size            = B )

    def forward(self, nodal_texture):
        assert len(nodal_texture.shape) == 3 and nodal_texture.shape[-1] == 3 # B x |V| x 3
        nodal_texture = self.rgb2yuv(
                        nodal_texture.transpose(1,2).unsqueeze(-1) # B x 3 x |V| x 1
                    ).squeeze(-1).transpose(1,2)[:, :, 1:] # B x |V| x 2
        nodal_texture = self.colour_uv_normer(nodal_texture) # Normalize into [0,1]
        return self.h(nodal_texture)
        

class GlobalHistogram2DFixedSamplesLinWGANGP(VectorAdversaryLinWGANGP):
    """
    Defines a texture vector adversary on an unfolded (fixed position) histogram of colour values.
    """
    def __init__(self, 
                 num_template_vertices, 
                 hidden_sizes, 
                 wgan_gp_pen_weight, 
                 drift_mag_weight, 
                 num_fixed_samples,
                 batch_size,
                 ):

        # The histogram outputs a vector of probability densities (unfolded from 2D)
        super(GlobalHistogram2DFixedSamplesLinWGANGP, self).__init__(dim_input          = num_fixed_samples, 
                                                                     hidden_sizes       = hidden_sizes,
                                                                     wgan_gp_pen_weight = wgan_gp_pen_weight,
                                                                     drift_mag_weight   = drift_mag_weight )

        # Input: a B x |V| x 2 texture vector in [0,1] -> Output: B x nS histogram (fixed size, unfolded in 2D)
        self.h = GlobalHistogramFixedSamplesAB(num_template_vertices = num_template_vertices, 
                                               num_fixed_samples     = num_fixed_samples,
                                               batch_size            = batch_size)

        # Over-ride the linear layer to (1) convert T to a histogram, (2) unfold it, and (3) run it through the lba
        #self.f = nn.Sequential( Reshaper((num_template_vertices, 3)), self.h, Unfolder(), self.f )
        self.core_f = self.f 
        self.unfolder = Unfolder()
        self.f = nn.Sequential( self.h, self.unfolder, self.core_f )
        self.expected_dim_len = 3

        # Working in the hue part of YUV colour space
        self.rgb2yuv = kornia.color.RgbToYuv()
        self.colour_uv_normer = ColourUvNormer()

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False, 
                return_histogram = False):

        assert len(v_fake.shape) == 3 # B x |V| x 3

        v_fake = self.rgb2yuv( 
                            v_fake.transpose(1,2).unsqueeze(-1) # B x 3 x |V| x 1
                        ).squeeze(-1).transpose(1,2)[:, :, 1:] # B x |V| x 2
        v_fake = self.colour_uv_normer(v_fake)

        if for_gen:
            return self.compute_loss_for_generator(
                                x                = v_fake, 
                                prob_weights     = prob_weights, 
                                max_probs_only   = max_probs_only,
                                return_histogram = return_histogram )
        else:
            
            v_real = self.rgb2yuv( 
                            v_real.transpose(1,2).unsqueeze(-1) # B x 3 x |V| x 1
                        ).squeeze(-1).transpose(1,2)[:, :, 1:] # B x |V| x 2
            v_real = self.colour_uv_normer(v_real)

            assert v_fake.shape == v_real.shape
            gp = compute_gradient_penalty_loss_texture(self.f, v_real, v_fake,
                                                       self.wgan_gp_pen_weight)
            output_real = self.f(v_real)
            output_fake = self.f(v_fake)
            loss = wasserstein_loss_dis(output_real, 
                                        output_fake, 
                                        drift_mag_weight = self.drift_mag_weight)
            return loss + gp

    def compute_loss_for_generator(self, x, prob_weights=None, max_probs_only=False, return_histogram=False):
        if return_histogram:
            histogram = self.h(x)
            return wasserstein_loss_gen(self.core_f(self.unfolder(histogram))), histogram
        return wasserstein_loss_gen( self.f(x) )


class GlobalHistogramFixedSamplesLinWGANGP(VectorAdversaryLinWGANGP):
    """
    Defines a texture vector adversary on an unfolded (fixed position) histogram of colour values.
    """
    def __init__(self, 
                 num_template_vertices, 
                 hidden_sizes, 
                 wgan_gp_pen_weight, 
                 drift_mag_weight, 
                 num_fixed_samples):
        super(GlobalHistogramFixedSamplesLinWGANGP, self).__init__(dim_input          = num_fixed_samples, 
                                                                   hidden_sizes       = hidden_sizes,
                                                                   wgan_gp_pen_weight = wgan_gp_pen_weight,
                                                                   drift_mag_weight   = drift_mag_weight)
        # Input: a B x |V| x 3 texture vector in [0,1] -> Output: B x nS histogram (fixed size, unfolded in 3D)
        self.h = GlobalHistogramFixedSamples(num_template_vertices = num_template_vertices, 
                                             num_fixed_samples     = num_fixed_samples)
        # Over-ride the linear layer to (1) convert T to a histogram, (2) unfold it, and (3) run it through the lba
        #self.f = nn.Sequential( Reshaper((num_template_vertices, 3)), self.h, Unfolder(), self.f )
        self.u = Unfolder()
        self.f0 = self.f
        self.f = nn.Sequential( self.h, Unfolder(), self.f )
        self.expected_dim_len = 3

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False,
                return_histogram=False):
        assert len(v_fake.shape) == 3
        if for_gen:
            return self.compute_loss_for_generator(
                                x              = v_fake, 
                                prob_weights   = prob_weights, 
                                max_probs_only = max_probs_only,
                                return_histogram = return_histogram)
        else:
            assert v_fake.shape == v_real.shape
            gp = compute_gradient_penalty_loss_texture(self.f, v_real, v_fake,
                                                       self.wgan_gp_pen_weight)
            output_real = self.f(v_real)
            output_fake = self.f(v_fake)
            loss = wasserstein_loss_dis(output_real, 
                                        output_fake, 
                                        drift_mag_weight = self.drift_mag_weight)
            return loss + gp

    def compute_loss_for_generator(self, x, prob_weights=None, max_probs_only=False, return_histogram=False):
        h = self.h(x)
        x = self.u(h)
        x = self.f0(x)
        if return_histogram:
            return wasserstein_loss_gen( x ), h
        return wasserstein_loss_gen( x )


class VecAndHistoVecTextureCritic(nn.Module):
    def __init__(self, input_dim,
                       num_template_vertices, 
                       fixed_graph_structure_batch,
                       hidden_sizes_local,
                       hidden_sizes_global, 
                       num_fixed_samples_global,
                       wgan_gp_pen_weight, 
                       drift_mag_weight, 
                ):
        super(VecAndHistoVecTextureCritic, self).__init__()

        self.f_vec = texture_critic = VectorAdversaryLinWGANGP(num_template_vertices * 3, 
                                                               (512, 256, 128),
                                                               wgan_gp_pen_weight = wgan_gp_pen_weight,
                                                               drift_mag_weight   = drift_mag_weight )
        
        self.f_global = GlobalHistogramFixedSamplesLinWGANGP(num_template_vertices = num_template_vertices, 
                                                             hidden_sizes = hidden_sizes_global, 
                                                             wgan_gp_pen_weight = wgan_gp_pen_weight, 
                                                             drift_mag_weight = drift_mag_weight, 
                                                             num_fixed_samples = num_fixed_samples_global )
    def forward(self, *args, **kwargs):
        return 0.5 * ( self.f_vec(*args, **kwargs) + self.f_global(*args, **kwargs) )

class GlobalAndLocalTextureCritic(nn.Module):
    def __init__(self, input_dim,
                       num_template_vertices, 
                       fixed_graph_structure_batch,
                       hidden_sizes_local,
                       hidden_sizes_global, 
                       num_fixed_samples_global,
                       wgan_gp_pen_weight, 
                       drift_mag_weight, 
                ):
        super(GlobalAndLocalTextureCritic, self).__init__()

        self.f_local = TextureGraphAdversaryWGANGP(per_node_input_dim = input_dim, 
                                                   hidden_dims = hidden_sizes_local, 
                                                   GSB = fixed_graph_structure_batch, 
                                                   wgan_gp_pen_weight = wgan_gp_pen_weight, 
                                                   drift_mag_weight = drift_mag_weight )
        self.f_global = GlobalHistogramFixedSamplesLinWGANGP(num_template_vertices = num_template_vertices, 
                                                             hidden_sizes = hidden_sizes_global, 
                                                             wgan_gp_pen_weight = wgan_gp_pen_weight, 
                                                             drift_mag_weight = drift_mag_weight, 
                                                             num_fixed_samples = num_fixed_samples_global )
    def forward(self, *args, **kwargs):
        return 0.5 * ( self.f_local(*args, **kwargs) + self.f_global(*args, **kwargs) )

class VectorAdversaryLin(nn.Module):
    def __init__(self, dim_input, hidden_sizes):
        super(VectorAdversaryLin, self).__init__()
        self.D = dim_input
        self.sizes = [ dim_input ] + list(hidden_sizes) + [ 1 ]
        self.f = LBA_stack( sizes = self.sizes,
                            norm_type = 'spectral',
                            act_type = 'lrelu',
                            end_with_lin = True )
        self.mse = torch.nn.MSELoss()

    def forward(self, for_gen, v_fake, v_real=None, prob_weights=None, max_probs_only=False):
        if for_gen:
            return self.compute_loss_for_generator(
                                x              = v_fake, 
                                prob_weights   = prob_weights, 
                                max_probs_only = max_probs_only)
        else:
            device    = v_fake.device
            valid     = Variable(torch.Tensor(v_real.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake      = Variable(torch.Tensor(v_fake.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( self.f(v_real), valid)
            fake_loss = self.mse( self.f(v_fake.detach()), fake)
            d_loss    = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, x, prob_weights=None, max_probs_only=False):
        if prob_weights is None:
            return (self.f(x) - 1.0).pow(2).mean() * 0.5
        else:
            B, NH, dim_v = x.shape
            if max_probs_only:
                # To avoid pushing poor hypotheses xi_p into the N(0,I) area, let's penalize only the 
                # maximum probability ones. This is to avoid drawing them as samples in cycle 1.
                minds = prob_weights.detach().argmax(dim=1) # B
                v_ml_only = x.gather(dim = 1, # Best hyps: B x NH x dim(xi_p) -> B x dim(xi_p)
                               index = minds.view(B,1,1).expand(-1,-1,dim_v)
                            ).squeeze(1)
                return 0.5 * (self.f(v_ml_only) - 1.0).pow(2).mean()
            else:
                return 0.5 * ( (self.f(x).squeeze(-1) - 1.0).pow(2) * prob_weights ).sum(dim=1).mean(dim=0)

class StdNormalMmdImqKMatcher(nn.Module):
    """
    Following the Wasserstein Auto-encoders paper, we use the inverse multi-quadratics kernel to compute
        the kernel MMD.
    """
    def __init__(self, xi_T_dim):
        super(StdNormalMmdImqKMatcher, self).__init__()
        self.d = xi_T_dim
        # IMQ kernel parameter
        # Note that this ONLY works when matching z~N(0,I), since C depends on Var[z]
        self.C = 2.0 * self.d
        logging.info('Initializing IMQ-kernel MMD matcher for %d', self.d)
        if self.d > 64: logging.info('\tWarning: for dimensions above ~64, MMD does not appear to perform well in this form')

    def imq(self, x, y):
        """ 
        x,y: B x d --> imq(x,y): shape B, of kernel inner products 
        """
        return self.C / (self.C + (x - y).pow(2).sum(dim=-1))

    def upper_triangular_kernelized_sum(self, x):
        """
        Computes sum_i sum_{j =/= i} k(x_i, x_j) = 2 * sum(upper_triangular( [k(x_i,x_j)]_ij ))
        Args: 
            x: B x d
        """
        B, d = x.shape
        # Indices to the above-diagonal upper-triangular portion of the matrix
        UT_inds = torch.triu_indices(B, B, offset=1)
        # Compute the B x B symmetric matrix of kernel evaluations between all input vectors
        K_dist_mat = self.imq( x.view(1,B,d).expand(B,-1,-1).reshape(B*B, d), 
                               x.view(B,1,d).expand(-1,B,-1).reshape(B*B, d) ).reshape(B, B)
        # Upper triangular (above-diagonal) portion of the pairwise distance matrix
        dists = K_dist_mat[ UT_inds[0], UT_inds[1] ]
        # Sum over the upper triangular portion, and double the result
        return 2.0 * dists.sum()

    def complete_kernelized_sum(self, x, y):
        """
        Computes sum_{k,l} k(x_k, y_l), i.e., all pairwise kernel distances (inner products in Hilbert space)
            between the samples in x and y.

        Args:
            x: B1 x d
            y: B2 x d
        """
        B1, d = x.shape
        B2, d = y.shape
        # Compute the B1 x B2 matrix of kernel evaluations between all input vectors
        K_dist_mat = self.imq( x.view(1,B1,d).expand(B2,-1,-1).reshape(B1*B2, d), 
                               y.view(B2,1,d).expand(-1,B1,-1).reshape(B2*B1, d) ).reshape(B1, B2)
        # Simply take the sum of all the kernelized inner products
        return K_dist_mat.sum()

    def forward(self, x):
        """
        Compute the MMD between empirical_dist(x) and N(0,I).

        Recall the MMD is written:
            MMD^2(P,Q) = || E_{x~P} phi(x) - E_{y~Q} phi(y) ||_H^2
                       = E_{x1,x2~P}[k(x1,x2)] + E_{y1,y2~Q}[k(y1,y2)] 
                            - 2 E_{a~P,b~Q}[k(a,b)]
                       = sup_{f in H s.t. ||f||_H <= 1} 
                            E_{x~P}[f(x)] - E_{y~Q}[f(y)]
        where k(a,b) = < phi(a), phi(b) >_H computes the distance in the RKHS and
            f is a test function in the Hilbert space.
        Since we use the IMQ kernel, we have to use the sample estimate of the MMD:
            MMDE^2(P_hat,Q_hat) =   c_s sum_i sum_{j =/= i} k(x_i, x_j)
                                  + c_s sum_i sum_{j =/= i} k(y_i, y_j)
                                  - c_d sum_{k,l} k(x_k, y_l)
        where c_s = (1 / (n(n-1))), c_d = 2 / n^2, x_i in P_hat, and y_j in Q_hat.

        But, note that P is fixed (P_hat are samples from a fixed distribution (here N(0,I))), 
            so we can ignore that term. 
        """
        B, d = x.shape 
        prior_z = torch.randn(B, d).to(x.device) # The prior sample doesn't have to be length B
        c_s = 1.0 / (B * (B - 1.0))
        c_d = 2.0 / (B**2)
        return (   c_s * self.upper_triangular_kernelized_sum(x) 
                 - c_d * self.complete_kernelized_sum(x, prior_z) ).clamp(min=1e-8).sqrt()

class StdNormalMomentMatcher(nn.Module):
    def __init__(self):
        super(StdNormalMomentMatcher, self).__init__()

    def forward(self, x):
        """
        Computes a moment matching loss between the batch of samples and the expected moments
            from N(0,I).
        """
        B, d = x.shape
        mean_penalty = x.abs().sum(dim=-1).mean()
        eye = torch.eye(d,d).to(x.device)
        demeaned = x - torch.mean(x, dim=1, keepdim=True) # B x d
        emp_cov = (1.0 / (B - 1)) * (demeaned.t()).matmul(demeaned)
        cov_penalty = (emp_cov - eye).abs().sum().sqrt()
        print(mean_penalty, cov_penalty)
        return mean_penalty + cov_penalty

class StdNormalSinkhornMatcher(nn.Module):
    def __init__(self, prior_batch_size=64):
        super(StdNormalSinkhornMatcher, self).__init__()
        import geomloss
        self.emd = geomloss.SamplesLoss(loss='sinkhorn', 
                                          p=1, 
                                          blur=0.05) # ~max distance
                                        #  blur=0.05) # ~max distance
        self.B = int(prior_batch_size)
        logging.info('Constructing Sinkhorn-based StdNormal matcher (B = %d) for xi_T', self.B)

    def forward(self, x):
        B, d = x.shape 
        prior_z_sample = torch.randn(1, self.B, d).to(x.device)
        L = self.emd(prior_z_sample, x.unsqueeze(0))
        return L.mean()

from networks.swae_helpers import _sliced_wasserstein_distance

class StdNormalSlicedWassersteinMatcher(nn.Module):
    def __init__(self, num_projections, expected_dim):
        super(StdNormalSlicedWassersteinMatcher, self).__init__()
        self.NP = num_projections
        logging.info('Constructing sliced Wasserstein distance to N(0,I) Gaussian')
        logging.info('\tNumProjections = %d', self.NP)
        self.learned = False
        self.vdim = expected_dim

    def forward(self, x):
        B, d = x.shape
        assert self.vdim == d
        prior_z_sample = torch.randn(B, d).to(x.device)
        return _sliced_wasserstein_distance(encoded_samples = x,
                                            distribution_samples = prior_z_sample,
                                            num_projections = self.NP,
                                            p = 2)


class ColourUvNormer(nn.Module):
    """
    Input: B x |V| x 2

    Assumes the output is from kornia YUV, but only the UV colour components.
    Note: U in [-0.436, 0.436], V in [-0.615, 0.615].
    """
    def __init__(self):
        super(ColourUvNormer, self).__init__()
        c = torch.tensor([0.436, 0.615]).view(1,1,2)
        self.register_buffer('c', c)

    def forward(self, inval):
        # [-a,a] -> [-1,1] -> [0,1]
        return 0.5 * ((inval / self.c) + 1.0)

#------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    import torch, kornia, sys
    torch.set_printoptions(threshold=50000)
    
    ##
    #rrr = torch.rand(1, 3, 1000, 1000)
    p = 125
    rrr = torch.stack( torch.meshgrid(torch.linspace(0, 1, p),
                                      torch.linspace(0, 1, p),
                                      torch.linspace(0, 1, p)), 
                       0 ).unsqueeze(0).reshape(1, 3, p**2, p)
    print('rrr', rrr.shape)
    colorconv = kornia.color.RgbToYuv()
    rrr = colorconv(rrr).squeeze(0).view(3, p**3)
    print('Mins', rrr.min(-1)[0])
    print('Maxs', rrr.max(-1)[0])
    print('(0,0,0) ->', colorconv( torch.tensor([0.0,0.0,0.0]).view(1,3,1,1) ) )
    print('(1,1,1) ->', colorconv( torch.tensor([1.0,1.0,1.0]).view(1,3,1,1) ) )
    print(rrr.shape, rrr[1:,:].transpose(0,1).unsqueeze(0).shape)
    rrr = ColourUvNormer()( rrr[1:,:].transpose(0,1).unsqueeze(0) ).squeeze(0) # |V| x 2
    print(rrr.shape)
    print('Mins', rrr.min(0)[0])
    print('Maxs', rrr.max(0)[0])
    ##
    
    B = 2
    nV = 1000
    disc = 10
    T = (torch.randn(B, nV, 2) * 0.1 + 0.5).clamp(min=0.0, max=1.0)
    #T = ((torch.rand(B, nV, 3) * 0.1) + 0.5).clamp(min=0.0, max=1.0)
    nU = 100
    U = torch.rand(B, nU, 2)

    #T[:, 0 : nU, :] = U

    print('sigma', T.std(dim=1))

    C = GlobalHistogramFixedSamplesAB(nV, num_fixed_samples = disc**2, batch_size=B)

    vec = C(T)
    print('IOP', vec.shape)
    print(vec.view(B, disc, disc))
    sys.exit(0)

    q = GlobalHistogramFixedSamplesLinWGANGP(
                    num_template_vertices = nV, 
                    hidden_sizes = (500,100), 
                    wgan_gp_pen_weight = 10, 
                    drift_mag_weight = 0.01, 
                    num_fixed_samples = 8**2)

    ttt = q(for_gen = True, v_fake = T.view(B, nV*2))

    print('ttt', ttt)

    # import torch
    # d = 64
    # B = 40
    # #M1 = StdNormalMmdImqKMatcher(d)
    # #M3 = StdNormalMomentMatcher()
    # M3 = StdNormalSinkhornMatcher()
    # #x = torch.bmm( 500*torch.randn(B,d,d), torch.rand(B, d, 1) ).squeeze(-1) 
    # x1 = torch.randn(B, d) * 0.1
    # x2 = torch.randn(B, d) * 1.0
    # x3 = torch.randn(B, d) * 5.0
    
    # a = torch.diag( torch.rand(d) ).unsqueeze(0).expand(B,-1,-1)
    # mm = lambda a,x : torch.bmm(a, x.unsqueeze(-1)).squeeze(-1)

    # #x[0 : B//2, :] *= 10
    # #x[B//2 : ,  :] *= 0.1
    # #print('Imq', M1(x))
    # #print('mm', M2(x))
    # print('Sinkhorn 0.0001', M3(x1/1000.0))    
    # print('Sinkhorn 0.001',  M3(x1/100.0))    
    # print('Sinkhorn 0.01',   M3(x1/10))
    # print('Sinkhorn 0.1',    M3(x1))
    # print('Sinkhorn 1.0',    M3(x2))
    # print('Sinkhorn 10.0',   M3(x3))
    # print('Sinkhorn R.01',     M3(mm(0.01*a , x2)))
    # print('Sinkhorn R.1',     M3(mm(0.1*a , x2)))
    # print('Sinkhorn R1',     M3(mm(a , x2)))
    # print('Sinkhorn R2',     M3(mm(2*a , x2)))
    # print('Sinkhorn R10',     M3(mm(10*a , x2)))
#------------------------------------------------------------------------------------------#




#
