import torch, torch.nn as nn, torch.nn.functional as F, math, logging
from networks.networks import *
import graphicsutils

class ImageShapePoseConditionedInferrer(nn.Module):
    """
    Map I, q, v, and {M_E}_i to {xi_T}_i
    """
    def __init__(self, q_dim, v_dim, nV, n_hyps, 
                 dim_xi_T, renderer_obj, template_faces, options):
        super(ImageShapePoseConditionedInferrer, self).__init__()
        logging.info('Building image, shape, and pose conditioned xi_T inferrer')
        logging.info('\tq_dim = %d, v_dim = %d, nV = %d, n_hyps = %d' 
            % (q_dim, v_dim, nV, n_hyps))

        self.q_dim           = q_dim
        self.v_dim           = v_dim
        self.nV              = nV
        self.n_hyps          = n_hyps
        self.dim_xi_T        = dim_xi_T
        self.ren_obj         = renderer_obj
        self.detach_occ_info = options['detach_occ_info']

        #self.template_faces = template_faces
        self.register_buffer('template_faces', template_faces.clone())

        ### SHORTCUT I -> S
        #self.use_shortcut = False
        #logging.info('\tUsing shortcut? ' + str(self.use_shortcut))
        # Mapping from I to shortcut signature
        #if self.use_shortcut:
        #    self.shortcut_dim = 256
        #    self.texturer_shortcut = ResShortcutNetwork(4, self.shortcut_dim)
        #else: 
        self.shortcut_dim = 0

        self.proc_shape_dim = 32 # dim(processed_v)
        self.v_preproc = LBA_stack_to_reshape(
                            [ self.v_dim, 128, self.proc_shape_dim ], 
                            [ self.proc_shape_dim ],
                            norm_type = 'bn',
                            end_with_lin = False)

        self.use_M_E = False
        logging.info('\tUsing M_E directly? ' + str(self.use_M_E))
        if self.use_M_E:
            self.M_E_i_init_dim = 32
            self.M_E_i_init_processor = nn.Sequential(
                                            Unfolder(),
                                            LBA_stack_to_reshape(
                                                [ 3 * self.nV, 512, 256, self.M_E_i_init_dim],
                                                [ self.M_E_i_init_dim ],
                                                norm_type = 'bn',
                                                end_with_lin = False) )# Next network starts with linear
        else:
            self.M_E_i_init_dim = 0

        self.use_occlusion = options['use_occlusion']
        self.backprop_through_unprojection = options['backprop_through_unprojection']
        self.intermed_pixunproj_dim = 512
        self.occ_dim = 1 if self.use_occlusion else 0
        self.n_input_channels = 4 + self.occ_dim  # 
        self.unproj_outdim = self.n_input_channels # 
        self.pixunproj_processor = LBA_stack_to_reshape(
                                    [ self.n_input_channels * self.nV, self.intermed_pixunproj_dim ], 
                                    [ self.intermed_pixunproj_dim ],
                                    norm_type = 'bn',
                                    end_with_lin = False)

        _tot_D = (self.M_E_i_init_dim + self.shortcut_dim + self.proc_shape_dim + 
                  self.q_dim + self.intermed_pixunproj_dim)
        logging.info('\tTotal initial input dimension: %d' % _tot_D)
        self.hyp_generator = LBA_stack_to_reshape(
                                [ _tot_D, self.dim_xi_T*3, self.dim_xi_T*2, self.dim_xi_T ], 
                                [ self.dim_xi_T ],
                                norm_type = 'bn',
                                end_with_lin = True)

        self.detach_M_E = options['detach_M_E_in_tex_inference']
        logging.info('\tDetaching M_E? ' + str(self.detach_M_E))

        # Same for v
        self.detach_v = True 
        logging.info('\tDetaching v? ' + str(self.detach_v))        

        logging.info('\tEnd of shape+pose dependent texture inference network constructor')

    def pixunproj(self, V, I):
        """
        Input: V (B x |V| x 3), I (B x C x H x W)
        Output: B x |V|*C [Unfolded, per vertex pixel value]
        
        We assume C == 4 (i.e., the input image has an alpha channel)
        """
        B, C, H, W = I.shape
        if self.use_occlusion:
            ( pixel_value_per_node,             
              rendered_depth_alpha_mask,       
              depth_image,                      
              depth_alpha_mask_value_per_node,  
              unproj_zbuffer_value_per_node,    
              depth_per_v,                       
              depth_difference,                 
              occlusion_signal                  
            ) = graphicsutils.get_pixel_unprojection_with_vertex_occlusion(
                    V     = V, 
                    ren   = self.ren_obj, 
                    I     = I, 
                    faces = self.template_faces.unsqueeze(0).expand(B,-1,-1),
                    detach_occ_info = self.detach_occ_info )
            pixunproj = torch.cat( ( pixel_value_per_node, 
                                     depth_difference.unsqueeze(-1) ), dim = 2)
        else:
            pixunproj = graphicsutils.pixel_unproject(V, self.ren_obj, I) 
        pixunproj = pixunproj.reshape(-1, self.nV * self.unproj_outdim)
        return pixunproj

    def processed_pixunproj(self, M_E, I):
        assert len(M_E.shape) == 3 # B(*nH) x |V| x 3
        if self.backprop_through_unprojection:
            return self.pixunproj_processor( self.pixunproj(M_E, I) )
        else:
            with torch.no_grad():
                unprojs = self.pixunproj(M_E, I)
            return self.pixunproj_processor(unprojs.detach())

    def forward(self, I, q, v, M_E):
        """
        Infer the texture hypotheses for I

        output: B x nH x dim(xi_T)
        """
        assert len(M_E.shape) == 4        
        B, nH, nTV, _ = M_E.shape
        BI, C, H, W   = I.shape
        assert len(I.shape) == 4 # B x C x H x W
        assert q.shape == (B, self.q_dim) # B x dim(q)
        assert v.shape == (B, self.v_dim) # B x dim(v)
        assert nTV     == self.nV and self.n_hyps == nH
        assert B       == BI
        assert C       == self.n_input_channels - (1 if self.use_occlusion else 0) 

        # Separate M_E from the computation graph, if needed
        if self.detach_M_E: M_E = M_E.detach()
        if self.detach_v:   v   = v.detach()

        # Signatures (image, shape, [shortcut])
        #com = [ q, self.v_preproc(v) ]
        #if self.use_shortcut:
        #    com += self.texturer_shortcut(I)
        S = torch.cat( (q, self.v_preproc(v)), dim = 1) # B x dim_full_sig

        # Unfold the Euclidean-transformed deformed templates
        M_E = M_E.reshape(B*nH, nTV, 3)

        # Preprocess the transformed M_Es
        if self.use_M_E:
            # (B*nH) x dim(pME_i)
            pME = self.M_E_i_init_processor( M_E ) 

        # Perform the pixel unprojection
        # Output: (B*nH) x dim(pixel_unproj_sig)
        pixel_unprojs = self.processed_pixunproj(
                          M_E, #.reshape(-1, nTV, 3), 
                          I.unsqueeze(1).expand(-1, nH, -1, -1, -1).reshape(B*nH,C,H,W) )

        # Append the processed transformed meshes to the processed unprojected pixels
        if self.use_M_E:
            Y = torch.cat( (pME, pixel_unprojs), dim = 1)
        else:
            Y = pixel_unprojs

        # Append the signatures (image and shape info) to the processed hypotheses
        B, dimS = S.shape
        Y = torch.cat( 
                ( Y, S.unsqueeze(1).expand(-1, nH, -1).reshape(B*nH, dimS) ), 
                dim = 1 ) # (B*nH) x (dim(pME_i + dimS))

        # Perform the final processing
        final = self.hyp_generator(Y)

        # Reshape from independently processed batches to hypotheses
        return final.view(B, nH, self.dim_xi_T)


#######################################################################################################################






#
