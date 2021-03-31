import torch, torch.nn as nn, torch.nn.functional as F
#import torch3d.models as models
from torch.autograd import Variable
from networks import GraphResNet
from networks.networks import *
from networks.helpers import GlobalMean

class ShapeAdversarySimplePC(nn.Module):

    def __init__(self, nfeats=3):
        super(ShapeAdversarySimplePC, self).__init__()
        self.f = PointNetT(nfeats, 1)
        self.mse = torch.nn.MSELoss()

    def forward(self, for_gen, pc_fake, pc_real=None):
        if for_gen:
            return self.compute_loss_for_generator(pc_fake)
        else:
            device = pc_fake.device
            valid = Variable(torch.Tensor(pc_fake.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(pc_real.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( self.f(pc_real), valid)
            fake_loss = self.mse( self.f(pc_fake.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, pc):
        return (self.f(pc) - 1.0).pow(2).mean() * 0.5

class FcTemplatePositionShapeAdversary(nn.Module):
    """
    Operates on deformed template vertex locations only.
    Due to the correspondence between template nodes,
        we can use simple FC networks.
    This does assume that the mapping from true_mesh to 
        deformed template is of decent quality.
    """
    def __init__(self, template_size):
        super(FcTemplatePositionShapeAdversary, self).__init__()
        
        self.nf = template_size * 3
        self.mse = torch.nn.MSELoss()
        self.f = nn.Sequential(
                    Unfolder(),
                    LBA_stack( sizes = (self.nf, 512, 256, 1),
                               norm_type = 'spectral', # 'batch',
                               act_type = 'elu',
                               end_with_lin = True )
                  )

    def forward(self, for_gen, fakes, reals=None):
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            B = real_scores.shape[0]
            device = fake_scores.device
            valid = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
            fake  = Variable(torch.Tensor(B, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( real_scores, valid)
            fake_loss = self.mse( fake_scores, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, fakes):
        return (self.f(fakes) - 1.0).pow(2).mean() * 0.5

class ComTwoStageShapeAdversary(nn.Module):
    def __init__(self, nTV, dim_lat_pert):
        super(ComTwoStageShapeAdversary, self).__init__()
        self.nf = nTV * 3
        self.dim_v = dim_lat_pert
        self.mse = torch.nn.MSELoss()
        self.intdim = 128
        # Critic on latent perturbations
        self.f_v = LBA_stack( sizes = (self.dim_v, 256, self.intdim),
                               norm_type = 'spectral', # 
                               act_type = 'elu',
                               end_with_lin = False )
        # Critic on template vertices
        self.f_tverts = nn.Sequential(
                             Unfolder(),
                             LBA_stack( sizes = (self.nf, 512, self.intdim),
                                norm_type = 'spectral', #
                                act_type = 'elu',
                                end_with_lin = False ) )
        # Combined critic
        self.com_crit = nn.Sequential(
                             LBA_stack( sizes = (self.intdim * 2, 128, 1),
                                norm_type = 'spectral', #
                                act_type = 'elu',
                                end_with_lin = True ) )
    def f(self, x):
        return self.com_crit( 
                torch.cat( 
                    ( self.f_v(x[0]), self.f_tverts(x[1]) ), 
                    dim = 1 ) 
        )

    def forward(self, for_gen, fakes, reals=None):
        """ Pass each input as a tuple (v, delta, M) """
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            Br = real_scores.shape[0]
            Bf = fake_scores.shape[0]
            device = fake_scores.device
            valid = Variable(torch.Tensor(Br, 1).fill_(1.0), requires_grad=False).to(device)
            fake  = Variable(torch.Tensor(Bf, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( real_scores, valid)
            fake_loss = self.mse( fake_scores, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, fakes):
        return (self.f(fakes) - 1.0).pow(2).mean() * 0.5

class SingleStageShapeAdversary(nn.Module):
    """
    Only apply the shape adversary to v, the latent shape.
    """
    def __init__(self, dim_lat_pert, layer_sizes, wgan_gp_pen_weight, drift_mag_weight):
        super(SingleStageShapeAdversary, self).__init__()
        from vector_adversaries import VectorAdversaryLinWGANGP
        self.critic = VectorAdversaryLinWGANGP(dim_lat_pert, layer_sizes, 
                                wgan_gp_pen_weight=wgan_gp_pen_weight, 
                                drift_mag_weight=drift_mag_weight)

    def forward(self, for_gen, fakes, reals=None):
        fakes = fakes[0]
        if not reals is None: reals = reals[0]
        return self.critic(for_gen=for_gen, v_fake=fakes, v_real=reals)


class ComSingleStageShapeAdversary(nn.Module):
    """
    Only apply the shape adversary to v, the latent shape.
    """
    def __init__(self, dim_lat_pert, nV, layer_sizes, wgan_gp_pen_weight, drift_mag_weight):
        super(ComSingleStageShapeAdversary, self).__init__()
        from vector_adversaries import VectorAdversaryLinWGANGP
        self.nV            = nV 
        self.reduced_M_dim = 128
        self.intermed_dim = 512
        self.M_preeuc_preproc = nn.Sequential(
                                    Unfolder(),
                                    LBA_stack_to_reshape(
                                        [ 3 * self.nV, self.intermed_dim, self.reduced_M_dim ],
                                        [ self.reduced_M_dim ],
                                        norm_type = 'bn',
                                        end_with_lin = False) # Next network starts with linear
                                 )
        self.critic = VectorAdversaryLinWGANGP(dim_lat_pert + self.reduced_M_dim, layer_sizes, 
                                wgan_gp_pen_weight = wgan_gp_pen_weight, 
                                drift_mag_weight   = drift_mag_weight)

    def forward(self, for_gen, fakes, reals=None):
        fakes = torch.cat( (fakes[0], self.M_preeuc_preproc(fakes[1])), dim = -1)
        if not reals is None: 
            reals = torch.cat( (reals[0], self.M_preeuc_preproc(reals[1])), dim = -1)
        return self.critic(for_gen=for_gen, v_fake=fakes, v_real=reals)



class TwoStageShapeAdversary(nn.Module):
    def __init__(self, nTV, dim_lat_pert):
        super(TwoStageShapeAdversary, self).__init__()
        self.nf = nTV * 3
        self.dim_v = dim_lat_pert
        self.mse = torch.nn.MSELoss()
        self.intdim = 1
        # Critic on latent perturbations
        self.f_v = LBA_stack( sizes = (self.dim_v, 256, 128, self.intdim),
                               norm_type = 'spectral', # 
                               act_type = 'elu',
                               end_with_lin = True )
        # Critic on template vertices
        self.f_tverts = nn.Sequential(
                             Unfolder(),
                             LBA_stack( sizes = (self.nf, 512, 256, self.intdim),
                                norm_type = 'spectral', #
                                act_type = 'elu',
                                end_with_lin = True ) )
        # Combiner
        self.f = DoubleAvg(self.f_v, self.f_tverts)

    def forward(self, for_gen, fakes, reals=None):
        """ Pass each input as a tuple (v, delta, M) """
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            Br = real_scores.shape[0]
            Bf = fake_scores.shape[0]
            device = fake_scores.device
            valid = Variable(torch.Tensor(Br, 1).fill_(1.0), requires_grad=False).to(device)
            fake  = Variable(torch.Tensor(Bf, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( real_scores, valid)
            fake_loss = self.mse( fake_scores, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, fakes):
        return (self.f(fakes) - 1.0).pow(2).mean() * 0.5

class MultiStageShapeAdversary(nn.Module):

    def __init__(self, nTV, dim_lat_pert, nts):
        super(MultiStageShapeAdversary, self).__init__()

        self.nf = nTV * 3
        self.dim_v = dim_lat_pert
        self.mse = torch.nn.MSELoss()
        self.intdim = 1
        self.d_nts = nts
        # Critic on latent perturbations
        self.f_v = LBA_stack( sizes = (self.dim_v, 128, self.intdim),
                               norm_type = 'spectral', # 
                               act_type = 'elu',
                               end_with_lin = True )
        # Critic on delta perturbations
        self.f_delta = nn.Sequential(
                            Unfolder(),
                            LBA_stack( sizes = (self.nf*self.d_nts, 512, 256, self.intdim),
                               norm_type = 'spectral', # 
                               act_type = 'elu',
                               end_with_lin = True ) )
        # Critic on template vertices
        self.f_tverts = nn.Sequential(
                             Unfolder(),
                             LBA_stack( sizes = (self.nf, 512, 256, self.intdim),
                                norm_type = 'spectral', #
                                act_type = 'elu',
                                end_with_lin = True ) )
        #
        if self.intdim == 1:
            self.f = TripletAvg(self.f_v, self.f_delta, self.f_tverts)
        else:
            self.f = TripletCom(self.intdim, self.f_v, self.f_delta, self.f_tverts)

    def forward(self, for_gen, fakes, reals=None):
        """ Pass each input as a tuple (v, delta, M) """
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            B = real_scores.shape[0]
            device = fake_scores.device
            valid = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
            fake  = Variable(torch.Tensor(B, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( real_scores, valid)
            fake_loss = self.mse( fake_scores, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, fakes):
        return (self.f(fakes) - 1.0).pow(2).mean() * 0.5
            

class DoubleAvg(nn.Module):
    def __init__(self, f1, f2):
        super(DoubleAvg, self).__init__()
        self.f1 = f1
        self.f2 = f2
    def forward(self, x):
        return (self.f1(x[0]) + self.f2(x[1])) / 2.0 

class TripletAvg(nn.Module):
    def __init__(self, f1, f2, f3):
        super(TripletAvg, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
    def forward(self, x):
        return (self.f1(x[0]) + self.f2(x[1]) + self.f3(x[2])) / 3.0 

class TripletCom(nn.Module):
    def __init__(self, D, f1, f2, f3):
        super(TripletCom, self).__init__()
        self.D = D
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f = nn.Sequential(
                        nn.ELU(),
                        nn.utils.spectral_norm( nn.Linear(self.D*3, 1) )
                )
    def forward(self, x):
        return self.f( torch.cat( (self.f1(x[0]), self.f2(x[1]), self.f3(x[2])), dim = -1 ) )    

class DglGcnSimpleMeshAdversary(nn.Module):
    """
    A simple GCNN critic operating on mesh inputs using the Deep graph library
    """
    def __init__(self, nfeats=3):
        super(DglGcnSimpleMeshAdversary, self).__init__()
        # f is a GCN that maps from a batched DGL graph set to a scalar (per graph)
        from networks.DglGCN import GCNClassifier
        self.f = GCNClassifier(in_dim = 3, hidden_dim = 64, outdim = 1)
        self.mse = torch.nn.MSELoss()

    def forward(self, for_gen, fakes, reals=None):
        """
        Inputs must be batched DGL graphs with nodal features in the entry named "features"
        """
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            B           = real_scores.shape[0]
            device      = fake_scores.device
            valid       = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
            fake        = Variable(torch.Tensor(B, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss   = self.mse( real_scores, valid)
            fake_loss   = self.mse( fake_scores, fake)
            d_loss      = 0.5 * (real_loss + fake_loss)
            return d_loss

    def compute_loss_for_generator(self, pc):
        return (self.f(pc) - 1.0).pow(2).mean() / 2

class SilhouetteAdversary(nn.Module):
    def __init__(self):
        super(SilhouetteAdversary, self).__init__()
        self.f = SilhouetteCriticNetwork()

    def forward(self, for_gen, fakes, reals=None, faces_fake=None, faces_real=None):
        
        # Here, fakes and reals are vertex positions
        
        if for_gen:
            return self.compute_loss_for_generator(fakes)
        else:
            real_scores = self.f(reals)
            fake_scores = self.f(fakes)
            B = real_scores.shape[0]
            device = fake_scores.device
            valid = Variable(torch.Tensor(B, 1).fill_(1.0), requires_grad=False).to(device)
            fake  = Variable(torch.Tensor(B, 1).fill_(0.0), requires_grad=False).to(device)
            real_loss = self.mse( real_scores, valid)
            fake_loss = self.mse( fake_scores, fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            return d_loss
 
    def compute_loss_for_generator(self, I):
        return (self.f(I) - 1.0).pow(2).mean() * 0.5

##################################################################################################

from networks.pc_archs import PointNet as LocPointNet, VaePointNet as VaePN

class PointNetT(nn.Module):
    def __init__(self, nfeats, dim_out, for_vae=False):
        super(PointNetT, self).__init__()
        #self.f = models.PointNet(nfeats, dim_out, dropout=dropout)
        if for_vae:
            self.f = VaePN(indim=nfeats, outdim=dim_out)
        else:
            self.f = LocPointNet(in_channels=nfeats, num_classes=dim_out)

    def forward(self, pc):
        return self.f( pc.transpose(1,2) )

##################################################################################################

# Consider the trivial approach of a simple image critic on the projections
# See "Synthesizing 3D shapes from Silhouette Image Collections", by Li et al [1]
# Also "SiCloPe: Silhouette-Based Clothed People", by Natsume et al [2]

class SilhouetteCriticNetwork(nn.Module): # Based on [1]
    def __init__(self):
        super(SilhouetteCriticNetwork, self).__init__()
        self.f = nn.Sequential(
                    nn.utils.spectral_norm( nn.Conv2d(1, 32, 3, stride=2, padding=1) ),
                    nn.LeakyReLU(0.2),
                    nn.utils.spectral_norm( nn.Conv2d(32, 64, 3, stride=2, padding=1) ),
                    nn.LeakyReLU(0.2),
                    nn.utils.spectral_norm( nn.Conv2d(64, 128, 3, stride=2, padding=1) ),
                    nn.LeakyReLU(0.2),
                    nn.utils.spectral_norm( nn.Conv2d(128, 256, 3, stride=2, padding=1) ),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 1, 1),
                    GlobalMean() )

    def forward(self, x):
        return self.f(x)


#
