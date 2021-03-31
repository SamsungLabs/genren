import torch, torch.nn as nn, torch.nn.functional as F
import meshutils, graphicsutils

try:
    import SoftRas.soft_renderer as sr
except ModuleNotFoundError:
    import soft_renderer as sr

class SrRenderer(nn.Module):

    def __init__(self, imsize, initial_eye=None, sigma_val=1e-5, cdir=[0,0,1], 
                       dist_eps=1e-4, gamma_val=1e-4, harden=False, orig_size=256):
        super(SrRenderer, self).__init__()
        if harden:
            self.renderer = sr.SoftRenderer(
                               image_size=imsize,
                               anti_aliasing=True,
                               orig_size=orig_size,
                               camera_mode='look_at',
                               camera_direction=cdir,
                               near=0.05,
                               far=100,
                               texture_type='vertex',
                               light_intensity_directionals=0.0,
                               light_directions=[0,1,0],
                               light_intensity_ambient=1.0,
                               sigma_val=sigma_val, # 1e-5 originally
                               dist_func='hard',      # <
                               dist_eps=dist_eps,
                               gamma_val=gamma_val, 
                               aggr_func_rgb='hard',  # < 
                               aggr_func_alpha='hard' # <
                         )
        else:
            self.renderer = sr.SoftRenderer(
                               image_size=imsize,
                               anti_aliasing=True,
                               orig_size=orig_size,
                               camera_mode='look_at',
                               camera_direction=cdir,
                               near=0.05,
                               far=100,
                               texture_type='vertex',
                               light_intensity_directionals=0.0,
                               light_directions=[0,1,0],
                               light_intensity_ambient=1.0,
                               sigma_val=sigma_val, # 1e-5 originally
                               dist_func='euclidean', 
                               dist_eps=dist_eps,
                               gamma_val=gamma_val, 
                               aggr_func_rgb='softmax', 
                               aggr_func_alpha='prod'
                         )

        # TODO not buffers?
        at = torch.tensor([0.0,0.0,0.0], requires_grad=False)
        up = torch.tensor([0.0,1.0,0.0], requires_grad=False)
        self.register_buffer('at', at)
        self.register_buffer('up', up)
        self.renderer.set_at_vec(self.at)
        self.renderer.set_up_vec(self.up)


        if not initial_eye is None:
            self.set_eye(initial_eye)

    def move_at_up(self, device):
        self.renderer.set_at_vec(self.at.to(device))
        self.renderer.set_up_vec(self.up.to(device))

    def set_eye(self, new_loc):
        self.renderer.set_eye(new_loc)

    @property
    def eye(self):
        return self.renderer.eye

    @eye.setter
    def eye(self, new_eye):
        self.set_eye(new_eye)

    def run_with_eye(self, V, F, T, eye):
        self.eye = eye
        return self.renderer(V, F, T, texture_type='vertex')

    def forward(self, V, F, T):
        assert V.shape[0] == F.shape[0] and T.shape[0] == V.shape[0]
        return self.renderer(V, F, T, texture_type='vertex')

    def forward_old(self, V, F, T=None, eye=None, azis=None, elevs=None, dists=None):
        """
        We are in radians here.
        Uses the current self.eye if None is given.
        """
        if T is None: T = torch.zeros(V.shape)
        # Case 1: Passing a list of (spherical) eye coordinates
        if (not azis is None) or (not elevs is None) or (not dists is None):
            eyes = graphicsutils.spherical_rads(azis, elevs, dists)
            return torch.cat([ 
                    self.run_with_eye(V, F, T, eye=eye.unsqueeze(0)) 
                    for eye in eyes], dim=0)
        # Case 2: Passing a single eye
        else:
            if not eye is None:
                assert V.shape[0] == 1 and F.shape[0] == 1 and T.shape[0] == 1
                assert eye.shape[0] == 1
                return self.run_with_eye(V, F, T, eye)
            else:
                assert V.shape[0] == F.shape[0] and T.shape[0] == V.shape[0]
                return self.renderer(V, F, T, texture_type='vertex')

############################################################################################

def main():
    device = torch.device('cuda:0')
    import meshzoo
    V, F = meshzoo.icosa_sphere(1)

def oldmain():
    device = torch.device('cuda:0')
    smesh = meshutils.read_surface_mesh('data/torus.obj', to_torch=True)
    V, F = smesh
    V = V.unsqueeze(0).to(device)
    F = F.unsqueeze(0).to(device)
    T = torch.rand(1, V.shape[1], 3).to(device)

    #azis0  = torch.tensor([0.0, 0.5,  1.0,  1.5, 2.0], requires_grad=True)
    #elevs0 = torch.tensor([0.0, 0.8, 1.75, 2.2, 3.14], requires_grad=True)
    #dists0 = torch.tensor([1.0, 2.2, 4.5, 8.75, 25.0],  requires_grad=True)
    azis0  = torch.tensor([0.0, 1.6], requires_grad=True)
    elevs0 = torch.tensor([0.0, 1.6], requires_grad=True)
    dists0 = torch.tensor([1.5, 4.0], requires_grad=False)

    azis  = azis0.to(device)
    elevs = elevs0.to(device)
    dists = dists0.to(device)

    R = SrRenderer().to(device)
    R.move_at_up(device)

    print('|VS|', V.shape)
    print('|FS|', F.shape)
    print('a,e,d:', azis.shape, elevs.shape, dists.shape)
    
    imgs = R(V, F, T, azis=azis, elevs=elevs, dists=dists)
    print('|I|', imgs.shape)

    display = False
    if display:
        from imgutils import mpl_show
        for i in range(imgs.shape[0]):
            mpl_show(imgs[i].cpu().detach())

    fake_target = torch.rand( imgs.shape ).to(device)
    L = ( (fake_target - imgs)**2 ).mean(0).sum()
    L.backward()

    print('Eye gradients')
    print('azis', azis0.grad)
    print('elevs', elevs0.grad)
    print('dists', dists0.grad)



#-------------------------#
if __name__ == "__main__":
    main()
#-------------------------#



