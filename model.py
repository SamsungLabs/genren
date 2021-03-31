import torch, torch.nn as nn, torch.nn.functional as F, math
from networks.networks import *
import graphicsutils, meshzoo, meshutils, dgl, utils
import logging, numpy as np, os, sys 
from networks.vae import VAE_encoder

class CyclicGenRen(nn.Module):
    """
    Model class for a 2D-3D inter-modality CycleGAN, the cyclic generative renderer.
    """
    def __init__(self, 
                 dim_xi_T,
                 dim_xi_p,
                 dim_lat_pert,
                 dim_backbone,
                 num_impulses,
                 num_hypotheses, # For multi-hypothesis pose prediction
                 use_alpha,
                 options,
                 using_PCs = True,
                 template_mesh = None, 
                 use_delta = True,
                 learn_template = False,
                 parallelize = None,
                 rotation_representation_mode = None,
                 FDR_pixel_distribution = None, # num_pixels x 3
                 pc_enc_arch = None,
                 renderer = None, # 'dgcnn', # None
        ):
        super(CyclicGenRen, self).__init__()
        logging.info('Initializing CyclicGenRen')

        ### Modes ###
        self.use_delta      = use_delta
        self.learn_template = learn_template
        self.use_alpha      = use_alpha
        self.sample_quats   = options['use_quat_sampling']
        self.generator_norm = options['generator_normalization'].lower().strip()
        assert self.generator_norm in ['sn', 'bn', 'sn+bn']
        assert use_delta # no alternative implemented
        
        # Pose probability mode
        self.pose_prob_method = options['pose_prob_method'].strip().lower() # 'allan'
        assert self.pose_prob_method in ['allan', 'simplex', 'softmax']
        logging.info('\tUsing pose prob calc method ' + self.pose_prob_method)

        if self.sample_quats:
            assert False
            logging.info('Sample quats is active --> dim(xi_p_rot) = 4 for rotation')
            logging.info('Specified dim(xi_p) = %d controls ONLY the translation component', dim_xi_p)

        # Rotation representation mode
        rotation_representation_mode = rotation_representation_mode.lower().strip()
        _rdims = { '6d' : 6, 'uq' : 4, '3a' : 3 }
        assert rotation_representation_mode in _rdims.keys()
        self.rot_mode = rotation_representation_mode

        ### Dimensionalities ###
        logging.info('\tDimensionalities')
        self.dim_xi_T            = _pq(dim_xi_T,       'xi_T', 2)
        self.dim_xi_p            = _pq(dim_xi_p,       'xi_p_trans' if self.sample_quats else 'xi_p', 2)
        self.dim_lat_pert        = _pq(dim_lat_pert,   'latent_pert', 2)
        self.dim_backbone        = _pq(dim_backbone,   'backbone', 2)
        self.num_impulses        = _pq(num_impulses,   'NTS', 2, nd=False) 
        self.num_pose_hypotheses = _pq(num_hypotheses, 'NPHs', 2, nd=False)
        self.rot_dim             = _rdims[self.rot_mode]
        self.euc_dim             = self.rot_dim + 3

        ### Learned template mesh ###
        # Default: use a sphere
        if template_mesh is None:
            # icosa 6  -> |V| = 362,  |F| = 720;  icosa 7  -> |V| = 492,  |F| = 980
            # icosa 8  -> |V| = 642,  |F| = 1280; icosa 9  -> |V| = 812,  |F| = 1620
            # icosa 10 -> |V| = 1002, |F| = 2000; icosa 11 -> |V| = 1212, |F| = 2420
            # icosa 12 -> |V| = 1442, |F| = 2880; icosa 13 -> |V| = 1692, |F| = 3380
            V, F = meshzoo.icosa_sphere(options['icosasphere_subdivs']) 
            logging.info('\tUsing SPHERE template (%s) [|V|=%d,|F|=%d]' 
                            % ('learned' if learn_template else 'fixed', V.shape[0], F.shape[0]))
        else:
            assert not (options['manual_template_UV'] is None)
            #assert False
            V, F = template_mesh
            logging.info('\tUsing external template (%s) [|V|=%d,|F|=%d]' 
                            % ('learned' if learn_template else 'fixed', V.shape[0], F.shape[0]))
        # Template vertices
        if learn_template:
            _V = torch.FloatTensor(V)
            _V = _V / _V.norm(dim = 1, keepdim = True)
            self.register_parameter( 'template_V', nn.Parameter(_V.detach().clone()) ) 
            #.to(parallelize[0])
            #self.template_V = nn.Parameter( _V.detach().clone() ) # |V| x 3
        else:
            fV = torch.FloatTensor(V)
            fV = fV / fV.norm(dim = 1, keepdim = True)
            self.register_buffer('template_V', fV)        

        # Template faces and edges
        F = torch.LongTensor(F)
        E = meshutils.F_to_E(F, both_directions=False)
        self.register_buffer('template_F', F) # |F| x 3
        self.register_buffer('template_E', E) # |E| x 3 = 3|F| x 3
        self.nV, self.nF = V.shape[0], F.shape[0]
        logging.info('\tE.shape = ' + str(E.shape))

        ### Compute UV mapping ###
        self.manual_UV_path = options['manual_template_UV']
        self.using_manual_UV = not (self.manual_UV_path is None)
        if self.using_manual_UV:
            print('Using manual BFF UVs')
            assert not (template_mesh is None), "Specify manual template if using manual UVs"
            U, V = meshutils.read_bff_uvs(self.manual_UV_path) # [-1,1]
            print( "\tObtained %d,%d BFF values" % (len(U),len(V)) )
            U = U.clamp(min = -1.0 + 1e-7, max = 1.0 - 1e-7)
            V = V.clamp(min = -1.0 + 1e-7, max = 1.0 - 1e-7)
        else:
            print('Using spherical UVs')
            assert template_mesh is None
            U = torch.atan2(self.template_V[:, 0], self.template_V[:, 2]) / np.pi # [-1,1], |V|
            V = torch.asin(self.template_V[:, 1]) * (-2.0 / np.pi) # [-1,1], |V|

        # Final tensor of UV mapping
        C_uv = torch.stack( (U,V), dim = -1).unsqueeze(0).unsqueeze(0) # 1 x 1 x |V| x 2 
        assert C_uv.shape == (1, 1, self.nV, 2), "Received unacceptable UV shape " + str(C_uv.shape)
        self.register_buffer('C_uv', C_uv.detach().clone())
        logging.info('\tC_uv shape = ' + str(C_uv.shape))

        ### Controllable rotation vector decoder ###
        if   self.rot_mode == '6d':
            logging.info('Using 6D rotation representation')
            self.rotv_decoder = graphicsutils.SixDimRotationDecoder()
        elif self.rot_mode == 'uq':
            logging.info('Using quaternionic rotation representation')
            self.rotv_decoder = graphicsutils.QuatRotationDecoder()
        elif self.rot_mode == '3a':
            logging.info('Using angle triplet representation')
            self.angle_limit = math.pi # / 2
            self.rotv_decoder = graphicsutils.AngleTripletDecoder(self.angle_limit)
        else:
            raise ValueError('Unknown rotation representation type')

        ### Image Inference Networks ###
        # Backbone network: I -> embedding q (note: includes BN and relu)
        self.backbone_inference_network = resnet18_backbone(dim_backbone) # I -> q
        # Map backbone output to latent deformation (latent deformation encoder)
        self.lat_deformation_inferrer = LBA_stack_to_reshape(
                                            [dim_backbone, dim_lat_pert], 
                                            [dim_lat_pert],
                                            init_lin_scale = 1e-2,
                                            end_with_lin=True)

        # Map backbone to pose/shape probabilities
        self.prob_inferrer = LBA_stack_to_reshape(
                                    [ dim_backbone, 
                                      32 * self.num_pose_hypotheses, # ~[100-200]
                                      self.num_pose_hypotheses],
                                    [ self.num_pose_hypotheses ],
                                    init_lin_scale = 1e-1,
                                    end_with_lin = True)

        # Map backbone to pose during inference (decoupled)        
        self.lat_pose_inferrer = LBA_stack_to_reshape(
                                            [  dim_backbone, 
                                               self.euc_dim * self.num_pose_hypotheses * 2, 
                                               self.euc_dim * self.num_pose_hypotheses ],
                                            [ self.num_pose_hypotheses, self.euc_dim ],
                                            init_lin_scale = 1e-1, #1.0,
                                            end_with_lin=True)

        # Map backbone output to latent texture (I -> xi_T)
        self.vae_xi_T = options['vae_for_xi_T']
        if self.vae_xi_T:
            assert False
            logging.info("\tUsing VAE for xi_T")
            self.lat_texture_inferrer = VAE_encoder(
                                            init_network   = LBA_stack_to_reshape(
                                                                [_total_texture_encoder_dim, 
                                                                    self.num_pose_hypotheses * dim_xi_T * 2, 
                                                                    dim_xi_T * self.num_pose_hypotheses ],
                                                                [ self.num_pose_hypotheses, dim_xi_T ],
                                                                norm_type = 'bn',
                                                                init_lin_scale = 1.0, 
                                                                end_with_lin=False), # Modified for VAE, 
                                            init_out_size  = dim_xi_T * 2, 
                                            final_out_size = dim_xi_T)
        else:
            # More complex inference network (q, v, M_E -> xi_T)
            from texture_inference_helpers import ImageShapePoseConditionedInferrer
            self.lat_texture_inferrer = ImageShapePoseConditionedInferrer(
                                            q_dim    = dim_backbone, 
                                            v_dim    = dim_lat_pert, 
                                            nV       = self.nV, 
                                            n_hyps   = self.num_pose_hypotheses, 
                                            dim_xi_T = dim_xi_T,
					                        renderer_obj = renderer,
                                            template_faces = self.template_F,
                                            options = options
                                            )

        ### Inferred Value Decoders ###

        #>> Latent deformation decoder <<#
        # Map latent deformation to nodal perturbation (v -> delta)
        self.deformation_decoder_network = LBA_stack_to_reshape(
                                            [dim_lat_pert, 600, 900, 1200, self.nV * self.num_impulses * 3],
                                            (self.num_impulses, self.nV, 3),
                                            #norm_type = 'sn+bn', # <<< decoder regularization for smoothness
                                            norm_type = 'bn', # <<< decoder regularization for smoothness
                                            init_lin_scale = 1e-1, # 1e-6,
                                            end_with_lin = True)

        #>> Texture Decoder <<#
        # Preprocess or project v before texture decoding
        self.v_feed = False
        if self.v_feed:
            assert False
            logging.info('Feeding latent shape v to the texture decoder')
            #-----------------------------------------#
            self.reduced_v_dim = 32 # 128
            self.detach_v_for_texture_decoder = options['detach_v_for_texture_decoder']
            #-----------------------------------------#
            logging.info('\tReduced v dim = %d. Detaching? %s', 
                         self.reduced_v_dim, str(self.detach_v_for_texture_decoder))
            self.v_reducer = LBA_stack_to_reshape(
                                            [ dim_lat_pert, 128, self.reduced_v_dim ], 
                                            #[ dim_lat_pert, self.reduced_v_dim ], 
                                            [ self.reduced_v_dim ],
                                            norm_type = options['tex_dec_norm_type'],
                                            end_with_lin = False)
        # Preprocess or project M_post_euc before texture decoding (or even detach it)
        else:
            #------------------------------------------#
            self.reduced_M_dim = 400 # 512
            self.detach_m_for_texture = True
            # self.linear_m_reducer = True
            #------------------------------------------#
            logging.info('Feeding deformed template (pre-euclidean) M to the texture decoder')
            self.M_reducer = nn.Sequential(
                                Unfolder(),
                                LBA_stack_to_reshape(
                                    # 3000,         1500,           750,                    400
                                    #[ 3 * self.nV, 3 * self.nV // 2, 3 * self.nV // 4, self.reduced_M_dim ],
                                    #[ 3 * self.nV, 3 * self.nV // 2, self.reduced_M_dim ],
                                    [ 3 * self.nV, self.reduced_M_dim * 2, self.reduced_M_dim ],
                                    [ self.reduced_M_dim ],
                                    norm_type = 'bn',
                                    end_with_lin = False) # Next network starts with linear
                             )
        # Map latent texture to nodal textures
        # Texture depends on shape but not pose
        _added_dim = self.reduced_v_dim if self.v_feed else self.reduced_M_dim
        from gan_baseline import WGANGPGenerator64, WGANGPGenerator32

        self.tex_img_gen_type = options['tex_img_gen_type']
        if self.tex_img_gen_type == '32':
            self.TID = 32
            self.texture_decoder = WGANGPGenerator32(nz = dim_xi_T + _added_dim, outchannels = 3)
        elif self.tex_img_gen_type == '64':
            self.TID = 64
            self.texture_decoder = WGANGPGenerator64(nz = dim_xi_T + _added_dim, outchannels = 3)


        # Map latent pose to intermediate value used to get actual pose (xi_p -> v_R, t)
        if self.sample_quats:
            assert False
            # xi_p is drawn as a uniformly random unit quat
            # Mapping from the input uniformly random unit quat (u -> r)
            map_4d_to_8d = LBA_stack_to_reshape([4, 32, 8],
                                                [8],
                                                norm_type = self.generator_norm, # 'sn+bn', 'bn'
                                                init_lin_scale = 1.0,
                                                end_with_lin = True)
            # Decodes the Gaussian xi_p into a translation vector (xi_p_translation -> t)
            # Again, note that here xi_p = xi_p_translation when sample_quats is True
            translation_decoder = LBA_stack_to_reshape(
                                                [dim_xi_p, 16, 3],
                                                [3],
                                                norm_type = self.generator_norm,
                                                init_lin_scale = 1.0,
                                                end_with_lin = True)
            from graphicsutils import RotationalTransformBasedEuclideanTransformGenerator
            # Mapping from (u, xi_p_t) --> (r, t)
            self.pre_pose_decoder = RotationalTransformBasedEuclideanTransformGenerator(
                                        map_4d_to_8d, translation_decoder)
        else:
            # Mapping from xi_p to (R,t)
            # xi_p is a vector from an arbtrary distribution
            self.pre_pose_decoder = LBA_stack_to_reshape(
                                                [dim_xi_p, 64, self.euc_dim],
                                                [3 + self.rot_dim],
                                                norm_type = self.generator_norm,
                                                init_lin_scale = 1e-1,
                                                end_with_lin = True)
        ### Mesh Processors ###
        # Map a mesh to latent deformation vector
        if using_PCs:
            _feature_length = 6 # coords + normals # TODO
            ## PointNet Case ##
            if pc_enc_arch is None or pc_enc_arch == 'pointnet':
                from shape_adversaries import PointNetT  
                # Option to use VAE for v or not
                self.use_vae_for_v = options['use_vae_for_v'] 
                if self.use_vae_for_v:
                    # Has dropout by default :/
                    self.mesh_template_pert_inferrer = VAE_encoder(
                                                        init_network   = PointNetT(_feature_length, 
                                                                                   dim_lat_pert * 2, 
                                                                                   for_vae = True), # Modified for VAE
                                                        init_out_size  = dim_lat_pert * 2, 
                                                        final_out_size = dim_lat_pert)
                    logging.info('Using Pointnet VAE as shape encoder')
                else:
                    self.mesh_template_pert_inferrer = NonVae( PointNetT(_feature_length, dim_lat_pert) )
                    logging.info('Using Pointnet (non-VAE) as shape encoder')
            ## DGCNN Case ##
            elif pc_enc_arch == 'dgcnn': 
                from networks.pc_archs import DGCNN
                self.mesh_template_pert_inferrer = DGCNN(_feature_length, dim_lat_pert, dropout=0.1)
                logging.info('Using DGCNN as shape encoder')
        else:
            raise ValueError('Unknown pc encoder option ' + str(pc_enc_arch))
        
        ### Parallelize if desired ###
        # note this may not be the most efficient, we may want to parallelize at a higher
        # level (i.e., at the full genren level), but then have to rewrite forward
        if not parallelize is None:
            self.backbone_inference_network  = nn.DataParallel( self.backbone_inference_network, 
                                                                device_ids = parallelize )
            self.lat_deformation_inferrer    = nn.DataParallel( self.lat_deformation_inferrer,
                                                                device_ids = parallelize )
            self.lat_pose_inferrer           = nn.DataParallel( self.lat_pose_inferrer,
                                                                device_ids = parallelize )
            self.lat_texture_inferrer        = nn.DataParallel( self.lat_texture_inferrer,
                                                                device_ids = parallelize )
            self.deformation_decoder_network = nn.DataParallel( self.deformation_decoder_network,
                                                                device_ids = parallelize )
            self.texture_decoder             = nn.DataParallel( self.texture_decoder,
                                                                device_ids = parallelize )
            self.pre_pose_decoder            = nn.DataParallel( self.pre_pose_decoder,
                                                                device_ids = parallelize )
            self.mesh_template_pert_inferrer = nn.DataParallel( self.mesh_template_pert_inferrer,
                                                                device_ids = parallelize )
            self.prob_inferrer               = nn.DataParallel( self.prob_inferrer,
                                                                device_ids = parallelize )
        ### DGL Precomputations ###
        self._dgl_template_graph = self.get_dgl_template_graph()

        ### Translation controls ###
        t_mins = torch.tensor(options['translation_mins'])
        t_maxs = torch.tensor(options['translation_maxs'])
        self.register_buffer('t_mins', t_mins)
        self.register_buffer('t_maxs', t_maxs)

        ### Pixel sampling handling, based on imgs ###
        random_pixels = FDR_pixel_distribution
        if random_pixels is None:
            random_pixels = torch.rand(50000, 3) #None 
        self.register_buffer('random_pixels', random_pixels)
        self.NRP = len(self.random_pixels)
        logging.info('Obtained random pixels: ' + str(self.random_pixels.shape))

    # </ End constructor /> #

    #------------------------------------------------------------------------------------------------#

    def activate_learn_template(self):
        self.learn_template = True
        self.template_V = nn.Parameter( torch.FloatTensor(self.template_V) ) # |V| x 3

    ######### Handling the MeshAE parameters (saving, loading, or printing) #########

    # https://stackoverflow.com/questions/53159427/pytorch-freeze-weights-and-update-param-groups
    def freeze_mesh_ae(self):
        for layer in [self.deformation_decoder_network, self.mesh_template_pert_inferrer]:
            for param in layer.parameters():
                param.requires_grad = False
                param.grad = None 

    def print_mesh_ae_params(self):
        for i, layer in enumerate([self.deformation_decoder_network, self.mesh_template_pert_inferrer]):
            for j, param in enumerate(layer.parameters()):
                print(i,j,param.sum(), param.std())

    def unfreeze_mesh_ae(self):
        for layer in [self.deformation_decoder_network, self.mesh_template_pert_inferrer]:
            for param in layer.parameters():
                param.requires_grad = True 

    def np_ae_mesh_params_list(self):
        return [ param.cpu().detach().numpy()
            for layer in [self.deformation_decoder_network, self.mesh_template_pert_inferrer] 
            for param in layer.parameters() ]

    def np_ae_mesh_params_diffs(self, prev):
        curr = self.np_ae_mesh_params_list()
        diffs = [ np.abs(c - p).sum() for c, p in zip(curr, prev) ]
        return diffs

    ######### Saves and Loads the Genren model #########

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def load_mesh_ae(self, path):
        # Load the pretrained weights
        loaded_state_dict = torch.load(path)
        module_names = [ "deformation_decoder_network", "mesh_template_pert_inferrer" ]
        # Access the current weights
        curr_state_dict = self.state_dict()
        # Iterate over the current weight sets
        for name in curr_state_dict.keys():
            # Only consider copying weights for mesh AE layers
            if any([ (cname in name) for cname in module_names ]):
                # Ensure the weight itself is present
                if name in loaded_state_dict.keys():
                    # Copy pretrained weight values
                    self.state_dict()[name].copy_(loaded_state_dict[name])

    def load_state(self, path, eval_mode=False):
        self.load_state_dict( torch.load(path) )
        if eval_mode: self.eval()

    @staticmethod
    def load_model(path, eval_mode=False):
        model = torch.load(path)
        if eval_mode: model.eval()
        return model

    ######### Encoding and decoding methods #########

    @staticmethod
    def _tanh_to_sigmoid(x):
        return (x + 1.0) / 2.0

    def decode_texture(self, xi_T, canon_template=None, v=None):
        """
        Mapping from xi_T to the texture tensor T.

        Return (UV_texture_image, texture_vector) with 
            shapes (B x [nH x] 3 x H x W, B x [nH x] x |V| x 3).

        If xi_T has hypotheses, then the output does as well.
        I.e., <B x [nH x ] dim(xi_T)> --> <B x [nH x] |V| x 3>
        """
        # Append v or M?
        if self.v_feed:
            assert False
            ### Handle v: detach, reduce
            if self.detach_v_for_texture_decoder: v = v.detach()
            assert len(v.shape) == 2
            extra = self.v_reducer(v)
        else: # M feed
            assert len(canon_template.shape) == 3
            m = self.M_reducer(canon_template)
            if self.detach_m_for_texture:
                m = m.detach()
            extra = m

        ### Case 1: no hypotheses
        if len(xi_T.shape) == 2:
            # B x 3 x H x W
            texture_image = self._tanh_to_sigmoid( self.texture_decoder( torch.cat( (xi_T, extra), dim=1 ) ) )
            # B x 3 x |T|
            nodal_textures = F.grid_sample( texture_image, 
                                            self.C_uv.expand(xi_T.shape[0], -1, -1, -1),
                                            align_corners = False,
                                            padding_mode = "border" 
                                           ).squeeze(-2)
            return texture_image, nodal_textures.transpose(1, 2)

        ### Case 2: hypotheses
        B, nH, dim_xi_T = xi_T.shape
        extra = extra.unsqueeze(1).expand(-1, nH, -1) # B x nH x dim_reduced_v
        reshaped_input = torch.cat( (xi_T, extra), dim = 2 ).reshape(B*nH, -1) # BnH x |z|
        # BnH x 3 x H x W
        texture_image = self._tanh_to_sigmoid( self.texture_decoder( reshaped_input ) )
        # BnH x |T| x 3
        nodal_textures = F.grid_sample(texture_image, 
                                       self.C_uv.expand(B*nH, -1, -1, -1),
                                       align_corners = False,
                                       padding_mode = "border" 
                                       ).squeeze(-2).transpose(1, 2)
        # Return the output, unfolded
        return ( texture_image.view(B, nH, 3, self.TID, self.TID),
                 nodal_textures.view(B, nH, -1, 3) )
        # else:
        #     assert len(canon_template.shape) == 3
        #     m = self.M_reducer(canon_template)
        #     if self.detach_m_for_texture:
        #         m = m.detach()
        #     return self.texture_decoder( torch.cat( (xi_T, m), dim = 1 ) )
        # [-1,1] -> renders into [-2,2] -> divide by 2 <<< Done post render now!!                

    def process_raw_translation(self, t):
        """
        Mapping from a raw translation vector t (B x 3) to a processed one (B x 3)
            with min/max values enforced.
        """
        assert len(t.shape) == 2 and t.shape[-1] == 3
        normedv = (torch.tanh(t) + 1.0) * 0.5 # B x 3
        return (normedv * (self.t_maxs - self.t_mins)) + self.t_mins

    def random_euclidean_pose(self, B, ret_intermed_pose=False):
        """ 
        Returns a random rigid pose (Euclidean transform) from our learned pose distribution.

        Performs: xi_p ~ P_pose -> decode_pose(x_p) = (R,t)

        if ret_intermed_pose:
            Returns (R,r,t)
        Otherwise:
            Returns (R,t), R = rotation matrices, t = translation 
        """
        return self.pose_decode(self.sample_xi_p(B), ret_intermed_pose=ret_intermed_pose)

    def infer_latent_texture(self, tup):
        """
        Perform latent texture inference.

        Input tuple: (I, q, v, M_hat)
        """
        if self.vae_xi_T:
            return self.lat_texture_inferrer(*tup)
        else:
            return self.lat_texture_inferrer(*tup), None, None

    def pose_decode(self, xi_p, ret_intermed_pose=False):
        """ 
        Performs a mapping from latent pose xi_p to Euclidean transform (R,t).

        Shapes: B x nH x |xi_p| --> (B x nH x 3 x 3, B x nH x 3, [B x nH x dim(r)]) 
        """
        if len(xi_p.shape) == 2: xi_p = xi_p.unsqueeze(dim=1)
        B, nH, dimxp = xi_p.shape
        dimr = self.rot_dim
        dimt = 3
        # Map latent pose to intermediate Euclidean pose
        temp = self.pre_pose_decoder(xi_p.view(B*nH, dimxp)) # B*nH x [dimr + 3]
        temp_unfolded = temp.view(B, nH, dimr + dimt)
        r_folded = temp[:, 0 : dimr] # This is NOT an angle, it is unscaled. (it is unbounded)
        R = self.rotv_decoder(r_folded + 1e-7) # Generate the rotation matrices
        r = temp_unfolded[:, :, 0 : dimr] # Unfolded and unscaled. Again, NOT an angle yet.
        # Post-process translation
        t = self.process_raw_translation( temp[:, dimr : ] ) # Warning! input is folded
        if ret_intermed_pose:
            return ( R.view(B,nH,3,3), t.view(B,nH,3,1), r )
        return R.view(B,nH,3,3), t.view(B,nH,3,1)

    def deformation_decoder(self, v):
        """
        Maps the latent deformation vector v to the pointwise template node deformation vector delta.

        v may come with or without hypotheses. We return it in the shape it came!
        """
        vs = v.shape
        shape_len = len(vs)
        if shape_len == 2: # B x dim(v)
            return self.deformation_decoder_network(v) # B x |I| x |V| x 3
        elif shape_len == 3: # B x nH x dim(v)
            B, nH, dimv = vs
            return self.deformation_decoder_network(v.reshape(B*nH, dimv)).reshape(B, nH, -1, self.nV, 3)

    def image_inference(self, I, detach_pose=False):
        """ 
        Performs inference on an image (extract pose, shape, and texture from a single input image).

        I -> (v, R_hat, t_hat, r_hat, pose_probs, xi_T_hat) 
        """
        # Obtain backbone (via resnet)
        q = self.backbone_inference_network(I) # 
        # Obtain inferred pose variables 
        r_t_p    = self.lat_pose_inferrer(q) # B x nH x (rot_dim + 3 + 1)
        B, nH, _ = r_t_p.shape
        # Intermediate rotation representation
        r = r_t_p[:, :, 0 : self.rot_dim] # This is NOT an angle, it is unscaled. (it is unbounded)
        # 3D translation
        t = self.process_raw_translation(
                r_t_p[:, :, self.rot_dim : self.rot_dim+3].view(B*nH, 3)
            ).view(B, nH, 3).unsqueeze(-1)
        # 3D rotation matrices
        R = self.rotv_decoder(r.view(B*nH, self.rot_dim) + 1e-7).view(B,nH,3,3)

        #-----#
        # Inferred probabilities as logits
        pose_probs = self.prob_inferrer(q)

        ### Map logits to probabilities ###
        #> Performs simple normalization into the simplex via division by the magnitude
        if self.pose_prob_method == 'simplex':
            # Simplex normalization -> allows multiple winners
            min_prob   = 0.005 # 1e-4 # Not exactly the min prob since there is normalization obviously
            pose_probs = ( torch.tanh(pose_probs) + 1.0 + min_prob ) # B x N_H, in [eps, 2.0 + eps]
            pose_probs = pose_probs / (pose_probs.sum(dim=1, keepdim=True) + 1e-6)
        #> Performs a "smeared" softmax 
        elif self.pose_prob_method == 'allan':
            # Allan's idea for the offset/smeared pose probs
            logits        = pose_probs
            beta_max      = 9.2 # ln(10^4), maximum offset to the logits
            max_logit     = torch.max(logits, dim=1)[0] # 
            median_logits = torch.median(logits, dim=1)[0]
            beta          = max_logit - median_logits # B
            # Beta is the distance from the max to the median
            # Anything between [max_logit - beta, max_logit] is shoved down to max_logit - beta
            beta[beta > beta_max] = beta_max # B, Limit the strength of beta
            # Now, any pose probs larger than L_max - beta get weakened
            modified_max     = max_logit - beta # B, smeared maximum logit (new upper bound, UB)
            mask             = pose_probs > modified_max.unsqueeze(1) # B x NH, mark any logit larger than the UB 
            pose_probs[mask] = modified_max.unsqueeze(1).expand(-1,nH)[mask] # replace logits that are too large
            pose_probs       = F.softmax(pose_probs, dim = 1)
        #> Performs a regular softmax
        elif self.pose_prob_method == 'softmax':
            # Softmax-based normalization -> encourage one winner
            # Result is almost always a Kronecker delta function in the end though
            pose_probs = F.softmax(pose_probs, dim = 1) # self.prob_inferrer(q), dim = 1)
        else:
            raise ValueError('Unrecognized pose prob computation method')
        #-----#

        # Latent deformation decoding (q -> v)
        v = self.lat_deformation_inferrer(q)

        # True deformation decoding (v -> delta)
        delta_hat = self.deformation_decoder(v) # multi-hypothesis shape
        M_hat, M_hat_preeuc, M_hat_intermeds = self.transformed_template(R, t, delta_hat, 
                                                                         detach_pose=detach_pose)

        return ( v, # Latent deformation v
                 R, # Rotation matrices
                 t, # Translation vectors
                 r, # Unscaled vector representation rotations
                 pose_probs, # Latent pose hypothesis probabilities (B x N_H) 
                 delta_hat, 
                 M_hat, M_hat_preeuc, M_hat_intermeds,
                 self.infer_latent_texture( (I, q, v, M_hat) )
               )

    def pre_euc_perturbed_template(self, delta, split=True): 
        """
        Maps a real-space perturbation (delta) to a perturbed template mesh (M_orig + delta).
        The output is in canonical coordinates (before a Euclidean transform).

        Expects: delta: B x |I| x |V| x 3
        """
        B = delta.shape[0]
        nts = self.num_impulses # num time steps
        dt = 0.01 # time step size
        delta = delta.view(B, self.num_impulses, self.nV, 3)
        undeformed_verts = self.template_V.unsqueeze(0).expand(B,-1,-1)
        if nts == 1:
            deformed_v = undeformed_verts + dt * delta[:,0,:,:]    
            if split:
                return deformed_v, None         
            else:
                return deformed_v.unsqueeze(1)
        all_deformed = torch.zeros(delta.shape).to(delta.device)
        for i in range(nts):
            verts = undeformed_verts if i == 0 else all_deformed[:,i-1,:,:]
            all_deformed[:,i,:,:] = delta[:,i,:,:] * dt + verts
        if split:
            pert_template = all_deformed[:, -1,   :, :]
            intermeds     = all_deformed[:, 0:-1, :, :]
            return pert_template, intermeds
        else:
            return all_deformed # B x num_impulses x |V| x 3

    def transformed_template(self, R, t, delta, detach_pose, deformed_template_pre_euc=None):
        """  
        Obtain the non-canonical transformed template (i.e., R(delta + M_template) + t).

        If deformed_template_pre_euc is passed, delta will not be used.

        Shapes:
            R: B x nH x 3 x 3
            t: B x nH x 3 x 1
            delta: B x [|I| = N_I x] |V| x 3

        Note that R is left multiplied when viewing the template verts as columns vectors,
            meaning it is transposed and right-multiplied when viewing the template verts
            as row vectors.

        Returns:
            if already deformed:
                Euclidean transformed templates
            else:
                (1) t + (template + delta) R^T [B x nH x |V| x 3]
                (2) perturbed template deform(template, delta) [B x nH x |V| x 3]
                (3) intermediates from template deformation 
        """
        if detach_pose:
            R = R.detach()
            t = t.detach()
        ### Case 1: already deformed by delta
        if not deformed_template_pre_euc is None:
            assert delta is None
            B, nV, _ = deformed_template_pre_euc.shape 
            B, nH, _, _ = R.shape
            pert_template = deformed_template_pre_euc.unsqueeze(1).expand(-1, nH, -1, -1)
            transformed_T = torch.matmul(
                                      pert_template, # Need this to be B x nH x |V| x 3
                                      #pert_template.unsqueeze(1).expand(-1, nH, -1, -1),
                                      R.permute(0,1,3,2) 
                            )  + t.squeeze(-1).unsqueeze(2)
            return transformed_T
        ### Case 2: deforming with delta
        B, nH, _, _ = t.shape
        delta_shape = delta.shape
        delta_dims = len(delta_shape)
        # Handle template learnability
        if self.learn_template:
            template_V = self.template_V 
        else:
            template_V = self.template_V.detach()
        # Obtain perturbed template
        if delta_dims == 4:
            # B x |I|=num_impulses x |V| x 3
            pert_template, intermeds = self.pre_euc_perturbed_template(delta) 
            # This will always return at least one hypothesis (i.e., nH >= 1)
            # When going shape -> shape, this should have nH = 1
            #pert_template = pert_template.unsqueeze(1).expand(-1, nH, -1, -1)
            # Now pert_template is B x |V| x 3 but R is B x NH x 3 x 3 (t is B x NH x 3 x 1)
            # So expand pert_template to B x nH x |V| x 3
            # mult(pert_template, R) is B x nH x |V| x 3, so we need to reshape 
            #   t to B x nH x 1 x 3 (the same translation is applied to every point) 
            # rather than B x nH x 3 x 1 as it currently is
            transformed_T = torch.matmul(
                                    pert_template.unsqueeze(1).expand(-1, nH, -1, -1), # Need this to be B x nH x |V| x 3
                                    #pert_template.unsqueeze(1).expand(-1, nH, -1, -1),
                                    R.permute(0,1,3,2) 
                            )  + t.squeeze(-1).unsqueeze(2)
        else:
            # B x nH x |I| x |V| x 3
            B, nH, nI, nV, _ = delta.shape
            pert_template, intermeds = self.pre_euc_perturbed_template(delta.reshape(B*nH,nI,nV,3)) 
            pert_template = pert_template.reshape(B, nH, nV, 3)
            # Now pert_template is B x |V| x 3 but R is B x NH x 3 x 3 (t is B x NH x 3 x 1)
            # So expand pert_template to B x nH x |V| x 3
            # mult(pert_template, R) is B x nH x |V| x 3, so we need to reshape 
            #   t to B x nH x 1 x 3 (the same translation is applied to every point) 
            # rather than B x nH x 3 x 1 as it currently is
            transformed_T = torch.matmul(
                                      pert_template, # Need this to be B x nH x |V| x 3
                                      #pert_template.unsqueeze(1).expand(-1, nH, -1, -1),
                                      R.permute(0,1,3,2) 
                            )  + t.squeeze(-1).unsqueeze(2)
        # Return post_euc_template, pre_euc_template, impulse_intermediates
        return transformed_T, pert_template, intermeds

    def sample_xi_p(self, B): 
        """
        Sample a random latent pose xi_p ~ P_pose.
        """
        xi_p_0 = torch.randn(B, self.dim_xi_p) # 
        #if self.sample_quats:
        #    return torch.cat( (graphicsutils.sample_uniform_rotation_quat(B), xi_p_0), dim = 1)
        return xi_p_0

    def sample_xi_T(self, B): 
        """
        Sample a random latent texture xi_T ~ P_texture
        """
        return torch.randn(B, self.dim_xi_T) # 

    def correct_renders(self, renderer_outputs_raw):
        #renderer_outputs_rgb   = (renderer_outputs_raw[:,0:3,:,:] * 2.0) - 1.0
        #renderer_outputs_alpha = renderer_outputs_raw[:,3,:,:].unsqueeze(1)
        return torch.cat( (  (renderer_outputs_raw[:,0:3,:,:] * 2.0) - 1.0, 
                             renderer_outputs_raw[:,3,:,:].unsqueeze(1)
                          ), dim=1)

    def correct_renders_old(self, renderer_outputs_raw, whiten_bg=None, single_light=True): # needed because of reasons
        """
        Based on the lighting environment (scene parameters), we have to correct the images from the renderer.
        """
        # Correction based on one or two lights
        if single_light: # [0,1] -> [-1,1]
            renderer_outputs_rgb = (renderer_outputs_raw[:,0:3,:,:] * 2.0) - 1.0
        else: # Two lights
            renderer_outputs_rgb = renderer_outputs_raw[:,0:3,:,:] - 1.0 # [0,2] -> [-1,1]
        # Extract alpha channel mask (in [0,1])
        renderer_outputs_alpha   = renderer_outputs_raw[:,3,:,:].unsqueeze(1)
        # Whiten the background or not
        if whiten_bg == True or (not self.use_alpha):
            renderer_outputs     = torch.clamp(
                                            renderer_outputs_rgb
                                            + (1.0 - renderer_outputs_alpha)*2,
                                        min = -1, max = 1)
        else:
            renderer_outputs = torch.cat( (renderer_outputs_rgb, renderer_outputs_alpha), dim=1)
        # Return corrected outputs
        return renderer_outputs

    def shape_to_deformed_template(self, shapes, normals):
        """
        Map a PN shape (PC+normals) to its canonical deformed representation.

        Returns:
            latent shape v, nodal perturbation delta, canonical shape M_preeuc, [mu_v, logvar_v]
        """
        assert len(shapes.shape) == 3 and shapes.shape == normals.shape
        v, mu_v, logvar_v = self.mesh_template_pert_inferrer( torch.cat( (shapes, normals), dim = 2 ) )
        delta    = self.deformation_decoder(v)
        M_preeuc = self.pre_euc_perturbed_template(delta, split=False)[:, -1, :, :]
        return (v, delta, M_preeuc, mu_v, logvar_v)
    
    def render(self, template_node_positions, textures, renderer):
        """
        Run the renderer on an input shape with texture
        """
        assert len(template_node_positions.shape) == 3, template_node_positions.shape
        assert len(textures.shape)                == 3, textures.shape
        assert template_node_positions.shape      == textures.shape
        B, nV, _ = template_node_positions.shape
        renders  = self.correct_renders( 
                      renderer( template_node_positions, 
                                self.template_F.unsqueeze(0).expand(B,-1,-1), 
                                textures ) )
        return renders # B x C=4 x H x W

    ####### Deep graph library methods (for GCNN-based approaches) #######

    def get_dgl_template_graph(self):
        """ Generates a *new* DGL graph of the template shape (no features inserted) """
        nV = self.nV
        g = dgl.DGLGraph()
        g.add_nodes(nV)
        edges = self.template_E # single direction
        src, dst = tuple(zip(*edges))
        # Insert edges into graph
        g.add_edges(src, dst)
        # Reverse edges (DGL edges are directional)
        g.add_edges(dst, src)
        # Self-edges (i.e. self-loops per node)
        g.add_edges(g.nodes(), g.nodes())
        return g

    def generate_featureless_dgl_template_graph_batch(self, B, as_dglb=False):
        g_template = self._dgl_template_graph # 
        # The constructor passing a DGL graph only copies the graph index 
        # (not the nodal/edge-wise features)
        GL = [ dgl.DGLGraph(g_template) for _ in range(B) ]
        if as_dglb:
            return dgl.batch(GL)
        return GL


    def construct_dgl_graphs_from_pcs(self, pcs):
        """
        Given a set (batch) of point clouds, constructs a DGL graph batch with
            the vertex positions as features.
        """
        B = pcs.shape[0]
        graphs = self.generate_featureless_dgl_template_graph_batch(B)
        for i, g in enumerate(graphs): g.ndata['features'] = pcs[i]
        G = dgl.batch(graphs) # Construct batch from graphs
        return G

    ###########################
    ### MAIN CYCLE FORWARDS ###
    ###########################

    def infer_on_random_new_view(self, canonical_shape, texture, renderer, 
                                 domain_randomized_pose,
                                 zero_xy_translation,
                                 return_decoded_texture, 
                                 detach_canonical_shape,
                                 include_pose_probs = False):
        """
        Takes (M, C) as input, renders a new image I_new with a random pose, and then runs inference on I_new and returns the resulting
            v_new, xi_T_new, I_new, and optionally C_new and uv_texture_img_new.

        If detach_canonical_shape, we prevent gradients from flowing back through the canonical_shape input.

        Returns:
            v_new, xi_T_new, I_new[, uv_texture_img_new, C_new]
        """
        if detach_canonical_shape:
            canonical_shape = canonical_shape.detach()
        # Generate a set of renders from the input shape and texture
        I_new = self.rerender_from_random_view(
                                deformed_template_pre_euc = canonical_shape, 
                                texture                   = texture, 
                                renderer                  = renderer, 
                                domain_randomized_pose    = domain_randomized_pose, 
                                zero_xy_translation       = zero_xy_translation, 
                                detach_pose               = True
                            )
        # Run image inference on the new renders
        ( v_hat, R_hat, t_hat, r_hat, pose_probs, 
          delta_hat, M_hat, M_hat_preeuc, M_hat_intermeds,
          (xi_T_hat, mu_xi_T_hat, logvar_xi_T_hat) ) = self.image_inference(I_new, detach_pose = True)

        if return_decoded_texture:
            # Decode the texture based on the newly inferred inputs
            decoded_texture_image, decoded_texture = self.decode_texture(xi_T_hat, v_hat if self.v_feed else M_hat_preeuc)
            if include_pose_probs:
                return (v_hat, xi_T_hat, I_new, decoded_texture_image, decoded_texture, pose_probs)
            return (v_hat, xi_T_hat, I_new, decoded_texture_image, decoded_texture)

        # Return the newly inferred outputs
        return (v_hat, xi_T_hat, I_new)

    def image_to_shape(self, images):
        """
        Perform the image to shape mapping
        """
        ( v_hat, R_hat, t_hat, r_hat, pose_probs, 
          delta_hat, M_hat, M_hat_preeuc, M_hat_intermeds,
          (xi_T_hat, mu_xi_T_hat, logvar_xi_T_hat) ) = self.image_inference(images, detach_pose = False)
        decoded_texture_image, decoded_texture = self.decode_texture(xi_T_hat, v_hat if self.v_feed else M_hat_preeuc)
        return (M_hat, None, xi_T_hat, v_hat, R_hat, t_hat, delta_hat, 
                M_hat_preeuc, M_hat_intermeds, pose_probs, r_hat, decoded_texture, 
                mu_xi_T_hat, logvar_xi_T_hat, decoded_texture_image)

    def v_to_image(self, v, renderer, xi_p = None, xi_T = None, detach_shape_in_rendering = False):
        B = v.shape[0]
        device = v.device 
        if xi_p is None:
            xi_p = self.sample_xi_p(B).to(device).unsqueeze(1) # One hypothesis
        if xi_T is None:
            xi_T = self.sample_xi_T(B).to(device)
        delta = self.deformation_decoder(v) # no hypotheses
        R, t, r = self.pose_decode(xi_p, ret_intermed_pose=True)
        # Do NOT detach the pose here, it learns it from the images
        V_new, V_new_preeuc, V_intermeds = self.transformed_template(R, t, delta, detach_pose=True)
        texture_img, texture = self.decode_texture(xi_T, v if self.v_feed else V_new_preeuc) # |V| x 3
        # Render V_new [B x nH x |V| x 3]
        B, nH, nV, _ = V_new.shape
        renders = self.correct_renders( 
                    renderer( 
                        ( V_new.view(B*nH, nV, 3).detach() 
                          if detach_shape_in_rendering else
                          V_new.view(B*nH, nV, 3)  ), 
                        self.template_F.unsqueeze(0).expand(B*nH,-1,-1), 
                        texture.unsqueeze(1).expand(-1,nH,-1,-1).view(B*nH,nV,-1) ) )
        return (renders, texture, V_new, delta, R, t, r, xi_p, xi_T, 
                V_new_preeuc, V_intermeds, texture, texture_img)

    def shape_to_image(self, shapes, normals, renderer, xi_p=None, xi_T=None, 
                             detach_shape_in_rendering=False, duplicated_xi_T_half_batch = False):
        """
        Perform the shape to image mapping

        If detach_shape_in_rendering, note that the RETURNED mesh still has the gradient connections intact.
            It is just that the input to the renderer has the gradients disconnected.
        """
        B = shapes.shape[0]
        device = shapes.device
        if xi_p is None:
            xi_p = self.sample_xi_p(B).to(device).unsqueeze(1) # One hypothesis
        if xi_T is None:
            if duplicated_xi_T_half_batch:
                xi_T = self.sample_xi_T(B // 2).to(device)
                xi_T = torch.cat( (xi_T, xi_T), dim = 0)
            else:
                xi_T = self.sample_xi_T(B).to(device)
        v, mu_v, logvar_v = self.mesh_template_pert_inferrer( torch.cat( (shapes, normals), dim = 2 ) )
        delta = self.deformation_decoder(v) # no hypotheses
        R, t, r = self.pose_decode(xi_p, ret_intermed_pose=True)
        # Do NOT detach the pose here, it learns it from the images
        V_new, V_new_preeuc, V_intermeds = self.transformed_template(R, t, delta, detach_pose=True)
        texture_img, texture = self.decode_texture(xi_T, v if self.v_feed else V_new_preeuc) # |V| x 3
        # Render V_new [B x nH x |V| x 3]
        B, nH, nV, _ = V_new.shape
        renders = self.correct_renders( 
                        renderer( 
                            ( V_new.view(B*nH, nV, 3).detach() 
                              if detach_shape_in_rendering else
                              V_new.view(B*nH, nV, 3)  ), 
                            self.template_F.unsqueeze(0).expand(B*nH,-1,-1), 
                            texture.unsqueeze(1).expand(-1,nH,-1,-1).view(B*nH,nV,-1) )
                  )
        return (renders, texture, V_new, delta, R, t, r, v, xi_p, xi_T, 
                V_new_preeuc, V_intermeds, mu_v, logvar_v, texture_img)

    def transformations_to_image(self, R, t, v, delta, texture=None, xi_T=None, renderer=None, detach_pose=True):
        """
        Map from the Euclidean and non-rigid transforms to a rendered image.

        Texture can be either not passed (will be sampled and decoded), passed as a latent (will be decoded), 
            or directly passed.
        """
        assert len(v.shape) == 2
        V_new, V_new_preeuc, V_intermeds = self.transformed_template(R, t, delta, detach_pose=detach_pose)
        if not xi_T is None:
            assert texture is None
            texture_img, texture = self.decode_texture(xi_T, v if self.v_feed else V_new_preeuc)
        B, nH, nV, _ = V_new.shape
        texture = texture.expand(-1,nH,-1,-1)
        assert texture.shape == V_new.shape
        renders = self.correct_renders( 
                        renderer( V_new.view(B*nH, nV, 3), 
                                  self.template_F.unsqueeze(0).expand(B*nH,-1,-1), 
                                  texture.reshape(B*nH,nV,-1) ) ) 
        return renders     

    def rerender_from_random_view(self, deformed_template_pre_euc, texture, renderer, 
                                  domain_randomized_pose, zero_xy_translation, detach_pose):
        """
        Generates a new set of renders, based on the textures and shapes of the inputs, but with random poses.

        The pose can be generated either via the domain randomized methods OR using the standard cycle 1 approach.
        """
        B, nV, _ = deformed_template_pre_euc.shape
        assert texture.shape == (B, nV, 3)
        # Domain randomized rotation and translation
        if domain_randomized_pose:
            R = self.generate_random_upper_hemi_pose(B)
            t = self.domain_randomized_translation(B)
        # Cycle-1-style sampled Euclidean pose
        else:
            R, t = self.random_euclidean_pose(B, ret_intermed_pose=False)
            if detach_pose:
                R = R.detach()
                t = t.detach()
        # Zero out the translation if needed
        if zero_xy_translation:
            assert t.shape == (B, 1, 3, 1)
            t[:,0,0,0] = 0.0
            t[:,0,1,0] = 0.0
        # GPU transfer
        device = deformed_template_pre_euc.device
        R = R.to(device)
        t = t.to(device)
        # Deform the template with a Euclidean transform
        # We detach the pose, since we are just trying to fix the occluded parts of the textures
        deformed_template_post_euc = self.transformed_template(R, t, 
                                                delta = None, 
                                                detach_pose = detach_pose, 
                                                deformed_template_pre_euc = deformed_template_pre_euc)
        # Compute the rendering
        return self.render(template_node_positions = deformed_template_post_euc.squeeze(1), 
                           textures = texture, 
                           renderer = renderer )


    def pretrain_iteration(self, shapes, normals):
        """
        Train the M -> delta -> D(Template, delta) = M_tilde sub-system.

        Runs an autoencoding cycle of the mesh autoencoder.
        """
        v, mu_v, logvar_v = self.mesh_template_pert_inferrer( torch.cat( (shapes, normals), dim = 2 ) )
        delta = self.deformation_decoder(v)
        pert_template, intermeds = self.pre_euc_perturbed_template(delta) # no euc transform
        return pert_template, intermeds, delta, v, mu_v, logvar_v

    def run_cycle_1(self, shapes, normals, renderer, duplicated_xi_T_half_batch = False):
        """
        S -> I -> S_hat
        """
        ( renders, texture, V_new, delta, 
          R, t, r, v, xi_p, xi_T, V_new_pe, 
          V_new_ints, mu_v, logvar_v, inward_tex_img ) = self.shape_to_image(shapes, normals, renderer, 
                                                             duplicated_xi_T_half_batch = duplicated_xi_T_half_batch,
                                                             xi_p = None, xi_T = None,
                                                             detach_shape_in_rendering = True)
        ( M_hat, _, xi_T_hat, v_hat, 
          R_hat, t_hat, delta_hat, M_hat_pe,
          M_hat_ints, pose_probs, r_hat, decoded_texture, 
          mu_xi_T_hat, logvar_xi_T_hat, tex_img_hat ) = self.image_to_shape(renders)

        return ( renders,         # Generated image renders
                 texture,         # Random texture (from xi_T)
                 V_new,           # Deformed nodal positions (incl. Euclidean transform)
                 V_new_pe,        # Deformed template (before Euclidean transform)
                 V_new_ints,      # Intermediate deformations (S -> I)
                 delta,           # Nodal perturbation vector
                 R, t,            # Random Euclidean transform (pose) [input]
                 r,               # Random rotation in 6D format
                 v,               # Random latent deformation [input]
                 xi_p,            # Random latent pose [input]
                 xi_T,            # Random latent texture [input]
                 M_hat,           # Reconstructed mesh [output]
                 M_hat_pe,        # Reconstructed mesh before Euclidean transform [output]
                 M_hat_ints,      # Intermediate deformations reconstruction (I -> S)
                 None,            # Inferred latent pose
                 xi_T_hat,        # Inferred latent texture
                 decoded_texture, # Decoded reconstructed texture
                 v_hat,           # Inferred latent deformation
                 R_hat,           # Inferred rotation
                 t_hat,           # Inferred translation
                 r_hat,           # Inferred 6D rotation
                 delta_hat,       # Inferred perturbation
                 pose_probs,      # Pose probabilities from image inference
                 mu_v, logvar_v,  # VAE output parameters from encoder
                 inward_tex_img,  # First cy1 texture image (randomly sampled)
                 tex_img_hat,     # Reconstructed texture image
               )

    def run_cycle_2(self, images, renderer, run_secondary_inference=False, renders_and_probs_only=False):
        """
        I -> S -> I_hat
        """
        # Map the input images to a mesh shape. The pose and texture are inferred here.
        ( M, _, xi_T, v, R, t, delta, M_pe, M_ints, pose_probs, r, decoded_texture, 
          mu_xi_T_hat, logvar_xi_T_hat, tex_img_hat ) = self.image_to_shape(images)
        # Map the input shapes to images -> The pose and texture latents are simply copied through
        # This means that we don't need to compute/sample/use v, R, t, or delta since they'll be the same.
        # Just render M with the inferred xi_T and pose directly.
        B, nH, nV, _ = M.shape 
        T = decoded_texture 
        assert T.shape == M.shape
        renders = self.correct_renders(
                        renderer( M.view(B*nH, nV, -1), 
                                  self.template_F.unsqueeze(0).expand(B*nH, -1, -1), 
                                  T.reshape(B*nH, nV, -1) )
                  )
        if renders_and_probs_only:
            return renders, pose_probs
        return [ M,               # Inferred mesh shape (post-euc)
                 M_pe,            # Inferred mesh shape before Euclidean transform
                 M_ints,          # Inferred intermediate mesh shapes over impulse time steps
                 None,            # Inferred latent pose
                 xi_T,            # Inferred latent texture
                 v,               # Inferred latent deformation
                 R,               # Inferred rotation
                 r,               # Inferred rotation intermediate
                 t,               # Inferred translation
                 delta,           # Inferred perturbation
                 T,               # Reconstructed texture (from inferred latent)
                 renders,         # Reconstructed image render
                 pose_probs,      # Pose hypothesis probabilities
                 mu_xi_T_hat,     # Mean VAE parameter for inference
                 logvar_xi_T_hat, # Log-variance VAE parameter for inference
                 tex_img_hat
               ] # + _fo

    ###############################################
    #### Domain-randomized pretraining Methods ####
    ###############################################

    def generate_random_texture(self, B, sigma=0.01, n_planes=1): 
        """
        Generates a random DR texture
        """
        # Based on the nodal template positions, color half the nodes differently based 
        # on a randomly chosen plane.
        # The plane has position centered at the chosen node and a random unit normal.
        V      = self.template_V
        device = V.device
        nV     = self.nV
        # Whether to use FDR textures from image samples
        using_sampled_texture = not (self.random_pixels is None)
        # Generate the random single colours from U[0,1] per batch member (B x nV x 3)
        if using_sampled_texture:
            _indssi = np.random.choice(self.NRP, size = (B,), replace = False)
            mu0 = self.random_pixels[_indssi, :].unsqueeze(1).expand(-1, self.nV, -1).to(device).reshape(nV*B, 3) 
        else:
            mu0 = torch.rand(B, 3).unsqueeze(1).expand(-1, self.nV, -1).to(device).reshape(nV*B, 3)
        def random_plane_defined_texture():
            chosen_nodes_inds = np.random.choice(nV, size=(B)) # Allow replacement
            chosen_nodes      = V[chosen_nodes_inds, :] # B x 3
            chosen_normals    = utils.random_unit_3vectors(B).to(device) # Plane normal per shape (B x 3)
            # Plane equation: 
            #       dot( plane_normal, (plane_point - point_xyz) ) = 0
            #       dot(plane_normal, plane_point) - dot(plane_normal, point_xyz) = 0
            # Compute the plane offsets
            offsets = torch.bmm( chosen_normals.unsqueeze(1), chosen_nodes.unsqueeze(-1) ).view(B)
            # Compute the plane_normal-to-vertex dot products
            dots = torch.bmm(  
                        # B x 3 -> B x nV x 1 x 3 -> B*nV x 1 x 3
                        chosen_normals.unsqueeze(1).expand(-1,nV,-1).reshape(nV*B,1,3),
                        # |V| x 3 -> B x |V|x 3 x 1 -> nV*B x 3 x 1
                        V.unsqueeze(0).expand(B,-1,-1).reshape(B*nV,3,1),
            ).view(B, nV) # dots_ij = v_i dot n_j
            # Compute the offset dot prods, used to get the signs/masks
            offset_dot_prods = dots - offsets.unsqueeze(-1).expand(-1, nV)
            # Compute the mask of what to recolour
            recolouring_mask_u = (offset_dot_prods > 0).reshape(B*nV) # B x nV -> nV*B unfolded
            # Generate the random colours from U[0,1] per batch member
            if using_sampled_texture:
                _indss = np.random.choice(self.NRP, size = (B,), replace = False)
                mu_r = self.random_pixels[_indss, :].unsqueeze(1).expand(-1, self.nV, -1).to(device) # B x nV x 3
            else:
                mu_r = torch.rand(B, 3).unsqueeze(1).expand(-1, self.nV, -1).to(device) # B x nV x 3
            return mu_r.clone().reshape(nV*B, 3), recolouring_mask_u
        # Generate textures
        for i in range(n_planes):
            mu_r, recolouring_mask_u = random_plane_defined_texture()
            # Colour over the masked parts of mu with mu_r
            mu0[recolouring_mask_u, :] = mu_r[recolouring_mask_u, :]
        # Finished recolouring
        mu = mu0.reshape(B,nV,3)
        # Small magnitude Gaussian noise in the texture 
        gaussian_vars = sigma * torch.randn(B, self.nV, 3).to(device) # from N(0, sigma^2 I)
        single_colours = (gaussian_vars + mu).clamp(min=0.001, max=0.999)
        return single_colours

    def generate_random_upper_hemi_pose(self, B):
        """
        Generates a random orientation (equivalent to placing the camera in the upper hemisphere randomly).
        Done by sampling a random rotation matrix.

        Output: B x 1 x 3 x 3
        """
        return graphicsutils.random_upper_hemi_rotm_manual(B).unsqueeze(1) # one hypothesis

    def domain_randomized_translation(self, B):
        """
        Sample a random 3D translation (in a fixed 3D rectangular prism)
        """
        hv_lims = 0.25  # TODO make a (random) function of z/depth
        min_tz  = -0.35 # Allow moving farther back than closer 
        max_tz  = 0.25  # 
        tx      = (2*torch.rand(B) - 1) * hv_lims 
        ty      = (2*torch.rand(B) - 1) * hv_lims 
        tz_len  = max_tz - min_tz
        tz      = (torch.rand(B) * tz_len) + min_tz # [min_tz, max_tz]
        return torch.stack( (tx, ty, tz), dim = -1 ).unsqueeze(1).unsqueeze(-1) # one hypothesis, & col vec

    def domain_randomized_shape_to_image(self, shapes, normals, renderer, learned_tex=False):
        """
        Perform the domain randomized shape-to-image cycle.

        This means that the pose and texture are randomly generated from hand-crafted distributions, rather
            than the learned mappings we wish to bootstrap into later.
        """
        B = shapes.shape[0]
        device = shapes.device
        v, mu_v, logvar_v = self.mesh_template_pert_inferrer( torch.cat( (shapes, normals), dim = 2 ) )
        delta = self.deformation_decoder(v)
        # We are not attempting to learn the pose and texture generating mechanisms
        # So we hand-craft simple methods of obtaining both
        # The goal is to pretrain the inference mechanism, which will
        #   later help the generation mechanism
        with torch.no_grad():
            R = self.generate_random_upper_hemi_pose(B).to(device)
            t = self.domain_randomized_translation(B).to(device)
            if not learned_tex:
                texture = self.generate_random_texture(B).to(device)
                _xi_T = None
        # Euclidean transform the template
        V_new, V_new_preeuc, V_intermeds = self.transformed_template(R, t, delta, detach_pose=True)
        if learned_tex:
            assert False
            _xi_T = self.sample_xi_T(B).to(device)
            texture = self.decode_texture(_xi_T, v if self.v_feed else V_new_preeuc)
        # Render V_new [B x nH x |V| x 3]
        B, nH, nV, _ = V_new.shape
        renders = self.correct_renders( 
                        renderer( V_new.view(B*nH, nV, 3), 
                                  self.template_F.unsqueeze(0).expand(B*nH,-1,-1), 
                                  texture.unsqueeze(1).expand(-1,nH,-1,-1).view(B*nH,nV,-1) )
                  )
        return renders, texture, V_new, delta, R, t, v, V_new_preeuc, V_intermeds, _xi_T, mu_v, logvar_v

    def run_domain_randomized_cycle_1(self, shapes, normals, renderer, learned_tex = False):
        """ 
        Domain randomized: S -> I -> S_hat 
        """
        ( renders, texture, V_new, delta, R, t, v, 
          V_new_pe, V_new_ints, _xi_T, mu_v, logvar_v 
          ) = self.domain_randomized_shape_to_image(shapes, normals, renderer, learned_tex)

        ( M_hat, _, xi_T_hat, v_hat, 
          R_hat, t_hat, delta_hat, M_hat_pe,
          M_hat_ints, pose_probs, r_hat, decoded_texture, mu_xi_T_hat, logvar_xi_T_hat, texture_img_hat
          ) = self.image_to_shape(renders)

        return ( renders,         # Generated image renders
                 texture,         # Random texture (from xi_T)
                 V_new,           # Deformed nodal positions (incl. Euclidean transform)
                 V_new_pe,        # Deformed template (before Euclidean transform)
                 V_new_ints,      # Intermediate deformations (S -> I)
                 delta,           # Nodal perturbation vector
                 R, t,            # Random Euclidean transform (pose) [input]
                 v,               # Random latent deformation [input]
                 M_hat,           # Reconstructed mesh [output]
                 M_hat_pe,        # Reconstructed mesh before Euclidean transform [output]
                 M_hat_ints,      # Intermediate deformations reconstruction (I -> S)
                 xi_T_hat,        # Inferred latent texture
                 decoded_texture, # Decoded reconstructed texture
                 v_hat,           # Inferred latent deformation
                 R_hat,           # Inferred rotation
                 t_hat,           # Inferred translation
                 r_hat,           # Inferred intermediate rotation representation
                 delta_hat,       # Inferred perturbation
                 pose_probs,      # Pose probabilities from image inference
                 _xi_T,           # Initial DR latent xi_T [None for FDR, given a value with LTDR]
                 mu_v, logvar_v, 
                 texture_img_hat)

#-------------------------------------------------------------------------------------------------#

##### Utilities #####

def _pq(x, s, nt=1, nd=True):
    ts = '\t' * nt
    if nd:
        logging.info('%sDim(%s): %d' % (ts,s,x))
    else:
        logging.info('%s%s: %d' % (ts,s,x))
    return x



#
