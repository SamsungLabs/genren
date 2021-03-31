import torch, ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as chamfer_distance
import torch.nn as nn, meshutils, os, sys, logging
from graphicsutils import RotationNegEntropyLoss, WeightedMultiHypMinAngComposedRotLoss
from utils import covariance
from vector_adversaries import VaeKL, StdNormalSlicedWassersteinMatcher
from perceptual_helpers import PerceptualMethodsHandler

class Cycle_1_loss_calculator(nn.Module):

    def __init__(self, V, E, F, options, mrl=None, for_mae=False):
        super(Cycle_1_loss_calculator, self).__init__()
        n_template_point_samples = options['n_template_point_samples']
        logging.info('Setting up Cy1 loss')
        self.cham_loss = chamfer_loss_object()
        self.V = V
        self.E = E
        self.F = F
        self.NTPS = n_template_point_samples
        # Rotation Matrix Distance loss
        self.R_dist_loss = WeightedMultiHypMinAngComposedRotLoss() 
        # Multi-hypothesis Pose regularizers
        self.rot_m_negent = RotationNegEntropyLoss()
        # Mesh reg loss
        if mrl is None:
            self.mrl = MeshRegularizationLoss(V, E, F)
        else:
            self.mrl = mrl
        self.intermeds_loss = IndepSeqMeshRegLoss( self.mrl )
        self.vae_kl = VaeKL()
        # Whether or not the mesh AE is frozen (hence whether to compute the losses)
        self.frozen_mae = False 
        if for_mae:
            self.mae_w_cham      = options['mae_w_cham']
            self.mae_w_emd       = options['mae_w_emd']
            self.mae_w_normals   = options['mae_w_normals']
            self.mae_w_mesh_reg  = options['mae_w_mesh_reg']
            self.mae_w_delta_reg = options['mae_w_delta_reg']
        else:
            ##### Loss function parameter weights #####
            self.w_cham_orig     = options['w_cham_orig']
            self.w_normals_orig  = options['w_normals_orig']
            self.w_cham_recon    = options['w_cham_recon']
            self.w_normals_recon = options['w_normals_recon']
            self.w_l2_con        = options['w_l2_con']
            self.w_texture_recon = options['w_texture_recon'] # Loss on texture recon for cy1
            self.w_texture_realism = options['w_texture_realism']
            # Fully domain randomized texture recon (Pretraining stage 2)
            self.w_texture_recon_PTS2_FDR = options['w_texture_recon_PTS2_FDR'] 

            self.w_adv_img       = options['w_adv_img']
            self.w_xi_T          = options['w_xi_T']
            #self.w_xi_p          = options['w_xi_p']
            self.w_mesh_reg      = options['w_mesh_reg']
            self.w_delta_reg     = options['w_delta_reg']
            self.w_pose_prob_ent = options['w_pose_prob_ent']
            self.w_R_negent      = options['w_R_negent']
            self.w_rot_recon     = options['w_rot_recon']
            self.w_t_recon       = options['w_t_recon']
            self.w_pose_adv      = options['w_pose_adv']
            self.w_v_reg         = options['w_v_reg']
            self.w_v_recon       = options['w_v_recon']

            self.w_tex_global_smoothness = options['w_tex_stddev_global']
            self.w_tex_varia_pen = options['w_tex_varia_pen']

            # Options for regularizing v
            self.v_reg_type = options['v_reg_type'].lower()
            assert self.v_reg_type in ['l2', 'swd']
            if self.v_reg_type == 'swd':
                self.v_regularizer = StdNormalSlicedWassersteinMatcher(num_projections = options['v_swd_nprojs'],
                                                                       expected_dim    = options['dim_lat_pert'])    

            # Cross-pose texture consistency (CPTC)
            self.using_cy1_cptc = options['use_CY1_cptc']
            if self.using_cy1_cptc:
                self.w_cy1_cptc_v_consis      = options['w_cy1_cptc_v_consis']
                self.w_cy1_cptc_xi_T_consis   = options['w_cy1_cptc_xi_T_consis']
                self.w_cy1_cptc_teximg_consis = options['w_cy1_cptc_teximg_consis']
                self.w_mixed_cy1_reren        = options['w_mixed_cy1_reren']
            # Used for adversarial CPTC
            self.w_adversarial_cptc = options['w_adversarial_cptc_cy1']

            # Histogram calculator (used if histogram critic not present)
            from vector_adversaries import UvHistogram
            self.UvHistogram = UvHistogram(nV = V.shape[0],
                                           nS = 25,
                                           #nS = options['num_histo_samples_per_dim'] ** 2,
                                           B  = options['B_shapes'] )

            if self.w_cham_recon < 1e-6 and self.w_normals_recon < 1e-6:
                logging.info('Inactive Chamfer and normals loss for reconstruction')

        if for_mae and self.mae_w_emd > 1e-6:
            import geomloss
            self.emd = geomloss.SamplesLoss(loss     = 'sinkhorn', 
                                            p        = 1, 
                                            blur     = 0.01, # recommended for the unit cube
                                            diameter = 1.2) # ~max distance

        logging.info({ k : v for k, v in vars(self).items() if type(v) is float })

        # Another coefficient on the adversarial pose loss
        # It should only be applied when cy2 has been operational for awhile
        self.apply_pose_loss = 0.0
        logging.info('Cy1 adversarial pose loss initially deactivated')

    def allow_adversarial_decoupled_pose_loss(self):
        self.apply_pose_loss = 1.0
        logging.info('Cy1 adversarial pose loss activated')

    def set_mae_as_frozen(self): 
        logging.info('Setting Cy1 loss for MAE as frozen')
        self.frozen_mae = True

    def set_mae_as_unfrozen(self):
        logging.info('Setting Cy1 loss for MAE as unfrozen')
        self.frozen_mae = False

    def get_mrl(self): return self.mrl

    def mae_preoptimization_loss(self, S_pc_real, N_pc_real, S_hat, delta):
        # S_pc_real, N_pc_real : 1 x Ns x 3
        # S_hat, delta : 1 x |V_template| x 3
        # Reconstruction (Chamfer + normals)
        if self.mae_w_cham < 1e-6 and self.mae_w_normals < 1e-6:
            chamfer_loss = torch.zeros(1).to(delta.device)
            normals_loss = torch.zeros(1).to(delta.device)
        else:
            chamfer_loss, normals_loss = self.compute_cham_and_normals_loss(
                                                S_pc_real, N_pc_real, S_hat,
                                                self.mae_w_cham, self.mae_w_normals)
        if self.mae_w_emd < 1e-6:
            emd_loss = torch.zeros(1).to(S_hat.device)
        else:
            pc_fake, normals_fake = meshutils.sample_triangle_mesh_with_normals(
                                          S_hat, self.F, self.NTPS)
            # TODO can append normals and emd match in 6D but need custom loss function
            # Else the normals won't match
            #emd_loss = self.mae_w_emd * self.emd(
            #                torch.cat( (pc_fake, normals_fake), dim=2), 
            #                torch.cat( (S_pc_real, N_pc_real),  dim=2)
            #            )
            emd_loss = self.mae_w_emd * self.emd(pc_fake, S_pc_real) 
        # Regularization on the deformed template mesh
        mesh_reg_loss, loss_d = self.mrl(S_hat, return_dict = True)
        mesh_reg_loss = self.mae_w_mesh_reg * mesh_reg_loss
        loss_d = { k : v*self.mae_w_mesh_reg for k,v in loss_d.items() }
        # Regularization on the additive perturbation
        delta_reg_loss, delta_dict = perturbation_seq_regularizer(delta, self.F, ret_dict = True)
        delta_reg_loss = self.mae_w_delta_reg * delta_reg_loss
        delta_dict = { k : _c(self.mae_w_delta_reg*v) for k, v in delta_dict.items() }
        # Store loss values
        loss_d['emd']     = _c(emd_loss)
        loss_d['chamfer'] = _c(chamfer_loss)
        loss_d['normals'] = _c(normals_loss)
        loss_d['total_mesh_reg'] = _c(mesh_reg_loss)
        loss_d['delta_pert_reg'] = _c(delta_reg_loss)
        loss_d.update(delta_dict) 
        # Return total_loss, loss_dictionary
        total_loss = chamfer_loss + normals_loss + emd_loss + mesh_reg_loss + delta_reg_loss
        return total_loss, loss_d

    def pretraining_loss(self, S_pc_real, N_pc_real, S_hat, S_intermeds, delta, v, mu_v, logvar_v):
        """
        S_pc_real: samples from the real PC
        N_pc_real: normals per point sample from the true mesh
        S_hat: Deformed template vertices
        """
        assert len(S_hat.shape) == 3 # B x |V| x 3
        # Compute chamfer and normals loss 
        chamfer_loss, normals_loss = self.compute_cham_and_normals_loss(
                                                S_pc_real, N_pc_real, S_hat,
                                                self.w_cham_orig, self.w_normals_orig)
        # Regularization
        final_out_reg  = self.w_mesh_reg  * self.mrl(S_hat) #.unsqueeze(1))
        intermeds_reg  = self.w_mesh_reg  * self.intermeds_loss(S_intermeds).to(v.device)
        delta_reg_loss = self.w_delta_reg * perturbation_seq_regularizer(delta, self.F)

        if mu_v is None or logvar_v is None:
            assert mu_v is None and logvar_v is None    
            #v_reg = self.w_v_reg * 0.5 * v.pow(2).mean()
            v_reg = self.w_v_reg * self.v_regularizer(v).mean()
        else: # VAE case
            v_reg = self.w_v_reg * self.vae_kl(mu_v, logvar_v)
        #v_reg          = self.w_v_reg     * self.vae_kl(mu_v, logvar_v) # 0.5 * v.pow(2).mean()
        
        return chamfer_loss, normals_loss, final_out_reg + intermeds_reg + delta_reg_loss, v_reg

    def compute_cham_and_normals_loss(self, pc_true, normals_true, deformed_template_V, w_cham, w_normals):
        """
        Returns chamfer loss and normals loss (as a tuple). Includes loss weighting.

        Chamfer dist output:
           d1 - dist of closest point on b of points from a
           d2 - dist of closest point on a of points from b
           i1 - idx of closest point on b of points from a
           i2 - idx of closest point on a of points from b
        Note that i1 indexes into shape 2 (pc_true)
        Similarly i2 indexes into shape 1 (pc_fake)
        """
        ### Compute Chamfer distance from deformed template verts to GT point cloud (direct loss)
        dc1, dc2, _, _ = self.cham_loss(deformed_template_V, pc_true)
        direct_chamfer = w_cham * (dc1.mean() + dc2.mean())
        #direct_chamfer = w_cham * (dc1.clamp(min=1e-7).sqrt().mean() + dc2.clamp(min=1e-7).sqrt().mean())
        ### Compute Chamfer distance from sampled PC (of template) to GT point cloud (sampled loss)
        # Obtain resampling of deformed template(s) with normals
        pc_fake, normals_fake = meshutils.sample_triangle_mesh_with_normals(
                                          deformed_template_V, self.F, self.NTPS)
        dist_1, dist_2, idx_1, idx_2 = self.cham_loss(pc_fake, pc_true)
        # Combined chamfer distances in both directions
        sampled_chamfer = w_cham * (dist_1.mean() + dist_2.mean())
        #sampled_chamfer = w_cham * (dist_1.clamp(min=1e-7).sqrt().mean() + dist_2.clamp(min=1e-7).sqrt().mean())
        ### Combine the direct and sampled chamfers
        total_chamfer = (direct_chamfer + sampled_chamfer) # Similar to RMSE
        ### Compute normals loss using the sampled face normals
        # Normals loss: absolute value of the dot product of the normals
        # True -> fake normals (B x N_S x 3)
        fake_normals_closest_to_true = torch.gather(normals_fake, 1, idx_2.long().unsqueeze(-1).expand(-1,-1,3)) 
        dist_true_to_fake = - (fake_normals_closest_to_true * normals_true).sum(-1).abs() # Note negative
        # Fake -> true normals
        true_normals_closest_to_fake = torch.gather(normals_true, 1, idx_1.long().unsqueeze(-1).expand(-1,-1,3)) 
        dist_fake_to_true = - (true_normals_closest_to_fake * normals_fake).sum(-1).abs()
        # Total normals loss
        total_normals_loss = w_normals * (dist_fake_to_true.mean() + dist_true_to_fake.mean())
        return total_chamfer, total_normals_loss

    def adv_I_partial_cycle1(self, S_pc_real, S_hat, S_intermeds, delta, renders, img_critic):
        recon_loss, reg_loss = self.pretraining_loss(S_pc_real, S_hat, S_intermeds, delta)
        adv_loss = self.w_adv_img * img_critic(for_gen=True, I_fake=renders)
        return recon_loss, reg_loss, adv_loss

    def forward(self, generator_iteration, S, S_hat, orig_M, orig_normals, img_critic, xi_p, xi_p_hat, xi_T, xi_T_hat, 
                v, v_hat, renders, delta, S_ints, S_hat_ints, R, t, R_hat, t_hat, 
                pose_probs, pose_critic, rsd, rsd_hat, input_texture, output_texture,
                mu_v, logvar_v, texture_critic, sampled_texture_image,
                reconstructed_texture_image, mixed_cy1_rerens_comparators = None,
                adv_cptc_loss = None,
                ):
        """
        Loss from M -> I -> M_hat cycle.

        In cycle 1, the shape pose is random (i.e., from the sampled latent pose),
            so we need to compute the distance between the pre-Euclidean transformed
                shapes and the original input, NOT the final output.
            This pre-transform shape is also what is assumed to belong to the dataset
                distribution (i.e., subjected to adversarial loss).
            The Euclidean transformed mesh is only used for generating the render.
        
        Notationally
            orig_M -> delta -> S [template] -> renders -> delta_hat -> S_hat [template]

        Args:
            S: the deformed template vertices inferred from the original mesh
            S_hat: the reconstructed shape (deformed template vertices inferred from the image)
            orig_M: a PC sampled from the original shape
        """
        assert type(generator_iteration) is int
        rv_dim = rsd.shape[-1]
        assert len(v.shape) == 2
        Bv, dim_v = v.shape
        B, nV, _ = S_hat.shape
        assert B == Bv
        assert len(xi_T.shape) == 2 and len(xi_T_hat.shape) == 3
        B, nH, dim_xi_T = xi_T_hat.shape

        ZERO = torch.tensor([0.0]).to(S.device)

        # Handle the adversarial CPTC loss
        if adv_cptc_loss is None:
            adv_cptc_loss = ZERO
        else:
            adv_cptc_loss = self.w_adversarial_cptc * adv_cptc_loss

        # Matching the original mesh to the initial deformed template
        if (not self.frozen_mae) and (self.w_cham_orig > 1e-6 or self.w_normals_orig > 1e-6):
            cham_orig_to_estimate, Lnormals_orig = self.compute_cham_and_normals_loss(
                                                      orig_M, orig_normals, S, #.squeeze(1), 
                                                      self.w_cham_orig, self.w_normals_orig)
        else:
            cham_orig_to_estimate = ZERO
            Lnormals_orig = ZERO
        # L2 loss between input & output deformed templates, as the nodes are in correspondence 
        # Matching the inferred template mesh from the image to the inferred template mesh from the original mesh
        # WARNING: S is detached here, so that it is not pushed towards the (likely poorer) S_hat
        # 
        # L2_template_loss_orig_to_reconstruction = self.w_l2_con * ( 
        #                                             (S.expand(-1,nH,-1,-1).detach() - S_hat)**2 
        #                                           ).mean()
        if self.w_l2_con > 1e-6:
            L2_template_loss_orig_to_reconstruction = self.w_l2_con * ( (S.detach() - S_hat)**2 ).mean()
        else:
            L2_template_loss_orig_to_reconstruction = ZERO

        # Also match the latent deformation v value
        # v_hat is from an image so it is B x nH x dim(v)
        # latent_deformation_v_recon_loss = self.w_v_recon * ( 
        #                                     (v.unsqueeze(1).expand(-1,nH,-1).detach() - v_hat).abs() 
        #                                 ).mean()
        latent_deformation_v_recon_loss = self.w_v_recon * ( (v.detach() - v_hat).abs() ).mean()

        # Chamfer loss from output deformed template to original input
        if self.w_cham_recon < 1e-6 and self.w_normals_recon < 1e-6:
            cham_orig_to_reconstruction = ZERO
            Lnormals_recon = ZERO
        else:
            cham_orig_to_reconstruction, Lnormals_recon = self.compute_cham_and_normals_loss(
                                                            orig_M, orig_normals, S_hat,
                                                            self.w_cham_recon, self.w_normals_recon)
        # Adversarial loss on the images
        img_adv_loss = self.w_adv_img * img_critic(for_gen = True, I_fake = renders).mean() 
        
        # Cyclic reconstruction loss on the latent pose variable
        #xi_p_loss = self.w_xi_p * ( ((xi_p - xi_p_hat)**2).mean(dim=-1) * pose_probs ).sum(dim=1).mean(dim=0) 
        
        # Cyclic reconstruction loss on the inferred pose variables
        rot_recon_loss = self.w_rot_recon * self.R_dist_loss(R.detach(), R_hat, pose_probs)
        t_recon_loss = self.w_t_recon * ( (t.detach() - t_hat) ** 2 ).mean() # TODO pose prob weight?

        # Encourage rotational diversity
        rot_div_loss = self.w_R_negent * self.rot_m_negent(R_hat, pose_probs)

        # Cyclic reconstruction loss on the latent texture variable
        xi_T_loss = self.w_xi_T * ( (xi_T.unsqueeze(1).expand(-1,nH,-1) - xi_T_hat)**2 ).mean()
        # Shape regularization
        shape_reg_loss = self.w_mesh_reg * self.mrl(S) / 2.0
        # Regularization on shape intermediates (for num_impulses > 1)
        shape_intermeds_loss = self.w_mesh_reg * ( self.intermeds_loss(S_ints) #+ 
                                                   # self.intermeds_loss(S_hat_ints)
                                                 ).to(S.device) / 2.0
        #
        #delta_reg_loss = self.w_delta_reg * single_additive_perturbation_variance_regularizer(delta, self.F) 
        delta_reg_loss = self.w_delta_reg * perturbation_seq_regularizer(delta, self.F) 
        
        # Decoupled adversarial pose loss -> penalize unrealistic (r,t)
        if self.apply_pose_loss < 1e-6:
            pose_adv_loss = ZERO
        else:
            npap = len(list(pose_critic.parameters()))
            if npap > 0:
                assert False
                random_r_and_t = torch.cat( (rsd.view(-1,rv_dim), t.view(-1,3)), dim=-1 ).view(-1, rv_dim + 3)
                pose_adv_loss = self.apply_pose_loss * self.w_pose_adv * pose_critic( 
                                                                            for_gen = True, 
                                                                            v_fake  = random_r_and_t 
                                                                         ).mean()
            else:
                pose_adv_loss = self.apply_pose_loss * self.w_pose_adv * pose_critic(
                                                                            for_gen = True,
                                                                            R_fake  = R.view(-1,3,3), # Cy1 LHS, not recon
                                                                            t_fake  = t.view(-1,3)
                                                                         ).mean() # for dataparallel
        # L2 penalty on v (regularized autoencoder loss)
        if mu_v is None or logvar_v is None:
            assert mu_v is None and logvar_v is None    
            latent_def_reg_v = self.w_v_reg * self.v_regularizer(v).mean()
            #latent_def_reg_v = self.w_v_reg * 0.5 * (v.pow(2).mean() + v_hat.pow(2).mean())
            #xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic( xi_T_best ).mean()
        else: # VAE case
            latent_def_reg_v = self.w_v_reg * self.vae_kl(mu_v, logvar_v)

        #
        if self.w_texture_recon < 1e-6:
            texture_loss = ZERO
        else:
            texture_loss = self.w_texture_recon * (
                                input_texture.detach().unsqueeze(1).expand(-1,nH,-1,-1) # Sampled texture
                                - output_texture # Reconstruction of sampled texture
                            ).abs().mean()

        # Enforce texture realism (from cy2)
        assert input_texture.shape == (B, nV, 3)
        if self.w_texture_realism < 1e-6:
            adv_texture_loss = ZERO
        else:
            if texture_critic.module.use_histo:
                adv_texture_loss, thistogram = texture_critic(
                                                   for_gen = True, 
                                                   v_fake = ( input_texture, # Texture itself
                                                              sampled_texture_image, # UV image of texture
                                                              #S.detach() 
                                                              ),
                                                   return_histogram = True
                                                ) 
            else:
                adv_texture_loss = texture_critic(
                                               for_gen = True, 
                                               v_fake = ( None, 
                                                          sampled_texture_image,
                                                          #S.detach() 
                                                          ),
                                            )
                thistogram = None
            adv_texture_loss = adv_texture_loss.mean() * self.w_texture_realism
        
        ### Texture smoothness ###
        if self.w_tex_varia_pen > 1e-8:
            if thistogram is None: 
                thistogram = self.UvHistogram(input_texture)
            tex_varia_pen = self.w_tex_varia_pen * histogram_entropy(thistogram).mean()
            #tex_varia_pen = self.w_tex_varia_pen * texture_smoothness(input_texture, self.F)
        else:
            tex_varia_pen = ZERO

        ### Mixed cy1 re-rendering consistency loss ###
        if (not mixed_cy1_rerens_comparators is None) and self.w_mixed_cy1_reren > 1e-7:
            assert self.using_cy1_cptc
            mixed_cy1_reren_loss = self.w_mixed_cy1_reren * (
                                        mixed_cy1_rerens_comparators[0] - mixed_cy1_rerens_comparators[1]
                                    ).pow(2).mean()
        else:
            mixed_cy1_reren_loss = ZERO

        ### Compute CPTC losses ###
        if self.using_cy1_cptc:

            assert len(v_hat.shape) == 2 # B x dim(v)
            assert len(xi_T_hat.shape) == 3 # B x nH x dim(xi_T)
            assert len(reconstructed_texture_image.shape) == 5 # B x nH x C x H x W

            b2 = B // 2
            v_hat_H1 = v_hat[0 : b2]
            v_hat_H2 = v_hat[b2 : ]
            xi_T_hat_H1 = v_hat[0 : b2]
            xi_T_hat_H2 = v_hat[b2 : ]
            reconstructed_texture_image_H1 = reconstructed_texture_image[0 : b2]
            reconstructed_texture_image_H2 = reconstructed_texture_image[b2 : ]

            # Compute CPTC losses
            cptc_v_consis      = self.w_cy1_cptc_v_consis      * ( (v_hat_H1 - v_hat_H2).abs() ).mean()
            cptc_xi_T_consis   = self.w_cy1_cptc_xi_T_consis   * ( (xi_T_hat_H1 - xi_T_hat_H2).pow(2) ).mean()
            cptc_teximg_consis = self.w_cy1_cptc_teximg_consis * ( 
                                        (reconstructed_texture_image_H1 - reconstructed_texture_image_H2).pow(2) 
                                    ).mean()
        else:
            cptc_v_consis      = ZERO
            cptc_xi_T_consis   = ZERO
            cptc_teximg_consis = ZERO
            # self.w_cy1_cptc_v_consis      = options['w_cy1_cptc_v_consis']
            # self.w_cy1_cptc_xi_T_consis   = options['w_cy1_cptc_xi_T_consis']
            # self.w_cy1_cptc_teximg_consis
            # cptc_v_consis      = self.w_cptc_v_consis * ( (v.detach() - cptc_v_hat).abs() ).mean()
            # cptc_xi_T_consis   = self.w_cptc_xi_T_consis * ( 
            #                             (xi_T_best.unsqueeze(1).expand(-1,NH,-1).detach() - cptc_xi_T_hat)**2 ).mean()
            # if self.w_cptc_teximg_consis > 1e-8:
            #     assert len(inferred_texture_images.shape) == 5 # B x NH x C x H x W
            #     assert inferred_texture_images.shape[0:3] == (B, NH, 3)
            #     assert inferred_texture_images.shape[2:] == cptc_decoded_texture_image.shape[1:]
            #     B, NH, Cti, Hti, Wti = inferred_texture_images.shape
            #     best_texture_images = inferred_texture_images.gather(dim = 1,
            #                                 index = inds.view(B, 1, 1, 1, 1).expand(-1, -1, Cti, Hti, Wti)
            #                             ).squeeze(1) # B x Cti=3 x Hti x Wti
            #     cptc_teximg_consis = self.w_cptc_teximg_consis * ( (best_texture_images - cptc_decoded_texture_image)**2 ).mean()

        # Meant to provide a signal to unobserved parts of the mesh (e.g. car or chair bottoms)
        tgs_loss = self.w_tex_global_smoothness * global_textural_stddev(input_texture)
        #tgs_loss = self.w_tex_global_smoothness * global_textural_stddev_img(sampled_texture_image)

        # Sum all the loss terms together
        LOSS = ( cham_orig_to_estimate +
                 Lnormals_orig + 
                 cham_orig_to_reconstruction +
                 Lnormals_recon +
                 shape_intermeds_loss +
                 L2_template_loss_orig_to_reconstruction +
                 latent_deformation_v_recon_loss +
                 shape_reg_loss +
                 img_adv_loss +
                 delta_reg_loss +
                 #xi_p_loss +
                 xi_T_loss + 
                 rot_recon_loss + 
                 t_recon_loss + 
                 rot_div_loss + 
                 latent_def_reg_v +
                 pose_adv_loss +
                 texture_loss +
                 tex_varia_pen +
                 tgs_loss +
                 cptc_v_consis +
                 cptc_xi_T_consis +
                 cptc_teximg_consis +
                 mixed_cy1_reren_loss +
                 adv_cptc_loss +
                 adv_texture_loss ) #+
                 #pose_prob_ent_loss ) 

        return LOSS, { 'cham_o'     : _c(cham_orig_to_estimate), 
                       'normals_o'  : _c(Lnormals_orig),
                       'cham_r'     : _c(cham_orig_to_reconstruction),
                       'normals_r'  : _c(Lnormals_recon),
                       's_ints'     : _c(shape_intermeds_loss),
                       'L2_tem_rec' : _c(L2_template_loss_orig_to_reconstruction),
                       'v_recon'    : _c(latent_deformation_v_recon_loss),
                       'adv_I'      : _c(img_adv_loss),
                       #'recon_xi_p' : _c(xi_p_loss),
                       'recon_xi_T' : _c(xi_T_loss),
                       'M_reg_c1'   : _c(shape_reg_loss),
                       'rot_recon'  : _c(rot_recon_loss),
                       't_recon'    : _c(t_recon_loss),
                       'delta_reg'  : _c(delta_reg_loss), 
                       'rot_div'    : _c(rot_div_loss),
                       'pose_adv'   : _c(pose_adv_loss),
                       'cy1_reg_v'  : _c(latent_def_reg_v),
                       'tex_loss'   : _c(texture_loss),
                       'tex_varia'  : _c(tex_varia_pen),
                       'tex_glob_stddev' : _c(tgs_loss),
                       # 'pprob_ent'  : _c(pose_prob_ent_loss),
                       'adv_tex'       : _c(adv_texture_loss), 
                       # CPTC losses
                       'cptc_v'        : _c(cptc_v_consis),
                       'cptc_xi_T'     : _c(cptc_xi_T_consis),
                       'cptc_teximg'   : _c(cptc_teximg_consis),
                       'cptc_mixed'    : _c(mixed_cy1_reren_loss),
                       'adv_cptc_loss' : _c(adv_cptc_loss),
                     }

    def domain_randomized_loss(self, S, S_hat, orig_M, orig_normals, 
                               v, v_hat, 
                               renders, 
                               delta, S_ints, S_hat_ints, 
                               R, t, R_hat, t_hat, 
                               pose_probs, 
                               input_texture, output_texture,
                               _xi_T = None, xi_T_hat = None, 
                               img_critic = None, 
                               mu_v=None, logvar_v=None,
                               adv_cptc_loss = None):
        """
        Domain randomized cycle 1 loss function

        S: the deformed template vertices inferred from the original mesh
        S_hat: the reconstructed shape (deformed template vertices inferred from the image)
        orig_M: a PC sampled from the original shape
        """
        assert len(v_hat.shape) == 2 # B x dimv
        assert len(S_hat.shape) == 3 # B x |V| x 3
        if not _xi_T is None: assert len(_xi_T.shape) == 2 
        if not xi_T_hat is None: assert len(xi_T_hat.shape) == 3
        assert len(output_texture.shape) == 4
        B, nH, nV, _ = output_texture.shape

        ZERO = torch.tensor([0.0]).to(S.device)

        if adv_cptc_loss is None:
            adv_cptc_loss = ZERO
        else:
            adv_cptc_loss = self.w_adversarial_cptc * adv_cptc_loss

        # Matching the original mesh to the initial deformed template
        if not self.frozen_mae:
            cham_orig_to_estimate, Lnormals_orig = self.compute_cham_and_normals_loss(
                                                      orig_M, orig_normals, 
                                                      S, #.squeeze(1), # Single hypothesis 
                                                      self.w_cham_orig, self.w_normals_orig)
        else:
            cham_orig_to_estimate = ZERO
            Lnormals_orig = ZERO

        # L2 loss between input & output deformed templates, as the nodes are in correspondence 
        # Matching the inferred template mesh from the image to the inferred template mesh from the original mesh
        # WARNING: S is detached here, so that it is not pushed towards the (likely poorer) S_hat
        if self.w_l2_con < 1e-6:
            L2_template_loss_orig_to_reconstruction = ZERO
        else:
            L2_template_loss_orig_to_reconstruction = self.w_l2_con * ( (S.detach() - S_hat)**2 ).mean()

        # Also match the latent deformation v value
        latent_deformation_v_recon_loss = self.w_v_recon * ( (v.detach() - v_hat).abs() ).mean()

        # Chamfer loss from output deformed template to original input
        if self.w_cham_recon < 1e-6 and self.w_normals_recon < 1e-6:
            cham_orig_to_reconstruction = ZERO
            Lnormals_recon = ZERO
        else:
            cham_orig_to_reconstruction, Lnormals_recon = self.compute_cham_and_normals_loss(
                                                            orig_M, orig_normals, 
                                                            S_hat, #.reshape(B*nH, nV, 3),
                                                            self.w_cham_recon, self.w_normals_recon)
        # Cyclic reconstruction loss on the inferred Euclidean pose variables
        rot_recon_loss = self.w_rot_recon * self.R_dist_loss(R.detach(), R_hat, pose_probs)
        t_recon_loss = self.w_t_recon * ( (t.detach() - t_hat)**2 ).mean() 
        # Encourage rotational diversity
        rot_div_loss = self.w_R_negent * self.rot_m_negent(R_hat, pose_probs)
        #pose_prob_ent_loss = self.w_pose_prob_ent * (
        #                            torch.log(1e-5 + pose_probs) * pose_probs 
        #                     ).sum(dim=-1).mean()

        # Cyclic reconstruction loss on the inferred texture variable
        texture_loss = self.w_texture_recon_PTS2_FDR * (
                                input_texture.detach().unsqueeze(1).expand(-1,nH,-1,-1) 
                                - output_texture).abs().mean()
        # Shape, latent shape, and perturbation regularizers
        shape_reg_loss = self.w_mesh_reg * ( self.mrl(S) #+ #.squeeze(1)) + 
                                             #self.mrl(S_hat) #.reshape(B*nH,nV,3)) 
                                           ) / 2.0
        shape_intermeds_loss = self.w_mesh_reg * ( self.intermeds_loss(S_ints) #+ 
                                                   #self.intermeds_loss(S_hat_ints)
                                                 ).to(S.device) / 2.0
        delta_reg_loss = self.w_delta_reg * perturbation_seq_regularizer(delta, self.F) 
        #
        #latent_def_reg_v = self.w_v_reg * 0.5 * (v.pow(2).mean() + v_hat.pow(2).mean())
        if mu_v is None or logvar_v is None:
            assert mu_v is None and logvar_v is None    
            #latent_def_reg_v = self.w_v_reg * 0.5 * (v.pow(2).mean() + v_hat.pow(2).mean())
            latent_def_reg_v = self.w_v_reg * self.v_regularizer(v).mean()
        else: # VAE case
            latent_def_reg_v = self.w_v_reg * self.vae_kl(mu_v, logvar_v)

        # Total loss
        LOSS = ( cham_orig_to_estimate +
                 Lnormals_orig + 
                 cham_orig_to_reconstruction +
                 Lnormals_recon +
                 shape_intermeds_loss +
                 L2_template_loss_orig_to_reconstruction +
                 latent_deformation_v_recon_loss +
                 shape_reg_loss +
                 delta_reg_loss +
                 rot_recon_loss + 
                 t_recon_loss + 
                 rot_div_loss + 
                 latent_def_reg_v +
                 adv_cptc_loss + 
                 texture_loss ) #+
                 #pose_prob_ent_loss ) 

        loss_dict = {  'cham_o'        : _c(cham_orig_to_estimate), 
                       'normals_o'     : _c(Lnormals_orig),
                       'cham_r'        : _c(cham_orig_to_reconstruction),
                       'normals_r'     : _c(Lnormals_recon),
                       's_ints'        : _c(shape_intermeds_loss),
                       'tem_recon'     : _c(L2_template_loss_orig_to_reconstruction),
                       'v_recon'       : _c(latent_deformation_v_recon_loss),
                       'M_reg_c1'      : _c(shape_reg_loss),
                       'rot_recon'     : _c(rot_recon_loss),
                       't_recon'       : _c(t_recon_loss),
                       'delta_reg'     : _c(delta_reg_loss), 
                       'rot_div'       : _c(rot_div_loss),
                       'cy1_reg_v'     : _c(latent_def_reg_v),
                       'adv_cptc_loss' : _c(adv_cptc_loss),
                       'tex_loss'      : _c(texture_loss), }
                       #'pprob_ent'  : _c(pose_prob_ent_loss), }

        # If the texture was from the learned generator, the _xi_T will not be None
        # In that case, we need to add terms for (1) reconstructing the latent texture
        # and (2) enforcing that the output images are realistic (via the image critic)
        if not _xi_T is None: # or not xi_T_hat is None or not img_critic is None:
            xi_T_loss    = self.w_xi_T * ( (_xi_T - xi_T_hat)**2 ).mean()
            img_adv_loss = self.w_adv_img * img_critic(for_gen = True, I_fake = renders).mean() # DP
            LOSS         = LOSS + xi_T_loss + img_adv_loss
            loss_dict['adv_I']      = _c(img_adv_loss)
            loss_dict['recon_xi_T'] = _c(xi_T_loss)

        # Final loss and loss dictionary
        return LOSS, loss_dict


#---------------------------------------------------------------------------------------------------------------#


class Cycle_2_loss_calculator(nn.Module):

    def __init__(self, V, E, F, options, mrl=None):    
        super(Cycle_2_loss_calculator, self).__init__()
        logging.info('Setting up Cy2 loss')
        self.V = V
        self.E = E
        self.F = F
        # Mesh reg loss
        if mrl is None:
            self.mesh_reg_loss = MeshRegularizationLoss(V, E, F)
        else:
            self.mesh_reg_loss = mrl
        self.intermeds_loss = IndepSeqMeshRegLoss(self.mesh_reg_loss)
        # Function for computing loss between rotation matrices
        self.rot_m_negent = RotationNegEntropyLoss()
        ##### Loss function parameter weights #####
        self.w_img_recon_l1     = options['w_img_recon_l1']
        self.w_img_recon_pd     = options['w_img_recon_pd']
        #self.w_img_grad_edge    = options['w_img_grad_edge']
        # Keep adv losses at 1 (so gen loss scale ~=~ critic loss scale)
        self.w_shape_adv_loss   = options['w_shape_adv_loss']
        #self.w_xi_p_adv_loss    = 0.0 
        self.w_xi_T_adv_loss    = options['w_xi_T_adv_loss'] 
        self.w_mesh_reg         = options['w_mesh_reg']
        self.w_delta_reg        = options['w_delta_reg']
        #self.w_sec              = 0.0 # <<<<<<<<<<<< inactive #
        self.w_pose_prob_ent    = options['w_pose_prob_ent']
        self.w_R_negent         = options['w_R_negent']
        self.w_v_reg            = options['w_v_reg_cy2']
        # Weight on applying texture critic to all pose hypothese
        #self.w_hyp_texture_reg  = options['w_hyp_texture_reg'] 
        # Weight on applying image adversary to randomly viewed rerendering of cy2 recons
        self.w_reren_adv_loss   = options['w_reren_adv_loss']
        # MS-SSIM
        self.ms_ssim_loss       = MS_SSIM_Loss(inchannels=4)
        self.w_img_recon_ms     = options['w_img_recon_ms']
        # VGG-based perceptual loss
        self.use_vgg_per_loss = options['VGG_per_loss']
        if self.use_vgg_per_loss:
            self.PLH = PerceptualMethodsHandler(vgg_path=options['VGG_path'])

        if options['vae_for_xi_T']:
            self.vae_xi_T = True
            self.vae_kl = VaeKL()
        else:
            self.vae_xi_T = False

        self.ganbased_xi_T_critic = not options['use_SWD_loss_for_xi_T']

        #self.w_hyp_tex_consistency = options['w_hyp_tex_match']
        self.w_hyp_xi_T_match = options['w_hyp_xi_T_match']
        self.w_hyp_t_match    = options['w_hyp_t_match']

        # CPTC (cy2) loss
        self.w_cptc_v_consis      = options['w_cptc_v_consis']
        self.w_cptc_xi_T_consis   = options['w_cptc_xi_T_consis']
        self.w_cptc_teximg_consis = options['w_cptc_teximg_consis']
        self.w_adversarial_cptc   = options['w_adversarial_cptc_cy2']

        # Type of adv loss on hyps
        self.adv_loss_on_hyps_type = options['adv_loss_on_hyps_type']
        assert self.adv_loss_on_hyps_type in ['weighted', 'best', 'all']

        # Options for regularizing v
        self.v_reg_type = options['v_reg_type'].lower()
        assert self.v_reg_type in ['l2', 'swd']
        if self.v_reg_type == 'swd':
            self.v_regularizer = StdNormalSlicedWassersteinMatcher(num_projections = options['v_swd_nprojs'],
                                                                   expected_dim    = options['dim_lat_pert']) 

        # Edge-based image loss
        #self.use_grad_edge_loss = False # options['use_grad_edge_loss']
        #if self.use_grad_edge_loss:
        #  from networks.img_proc_utils import WeightedGradientImageDistance
        #  self.GEL = WeightedGradientImageDistance()

        #self.xi_p_wll           = 0.0
        #self.w_L2_xi_p          = 1.0
        #self.w_sigma_match_xi_p = 0.0
        logging.info({ k : v for k, v in vars(self).items() if type(v) is float })
        options['w_xi_T_adv_loss'] 
        if 'w_xi_p_adv_loss' in options.keys():
            logging.info('Warning: xi_p adversarial loss is not being used, but loss weight was defined')


    def forward(self, I, I_hat, M_pe, xi_p, xi_T, 
                shape_critic, xi_p_critic, xi_T_critic, img_critic,
                v, delta, M_pe_dgl, M_pe_ints, renderer,
                pose_probs, R_inferred, t_inferred, model,
                inferred_texture_images = None,
                inferred_texture = None,
                texture_critic = None,
                # Rerenders of the cy2 outputs from a different viewpoint
                rerendered_imgs  = None,
                # These are only needed for secondary inference consistency
                v_sec = None, 
                xip_sec = None, xit_sec = None,
                # VAE inputs for KL loss, if used
                mu_xi_T = None, logvar_xi_T = None,
                # The cross-pose texture consistency loss
                cptc_data = None,
                adv_cptc_loss = None,
            ):
        """ 
        Loss from I -> M -> I_hat cycle 
        
        pose_probs: B x NH 
        """
        assert len(M_pe.shape) == 3 # B x |V| x 3
        assert len(inferred_texture.shape) == 4 # B x nH x nV x 3
        # Dimensionalities
        B, dim_v = v.shape
        BNH, NC, H, W = I_hat.shape
        #B, nH, nV, _ = M_pe.shape
        NH = BNH // B
        B, nV, _ = M_pe.shape
        #dxp = xi_p.shape[-1]

        ZERO = torch.tensor([0.0]).to(I.device)

        if adv_cptc_loss is None:
            adv_cptc_loss = ZERO
        else:
            adv_cptc_loss = self.w_adversarial_cptc * adv_cptc_loss

        # Pose-probability weighted image distance
        I = I.unsqueeze(1).expand(-1,NH,-1,-1,-1)
        I_hat = I_hat.view(B, NH, NC, H, W)
        
        img_loss = self.w_img_recon_l1 * ( 
                      pose_probs *
                        ( I - I_hat ).abs().mean(dim=-1).mean(dim=-1).sum(dim=-1)
                   ).sum(dim=-1).mean()

        # MS-SSIM loss
        if self.w_img_recon_ms > 1e-7:
            _fold = lambda x: x.reshape(B*NH, NC, H, W)
            ms_ssim_loss = self.w_img_recon_ms * ( 
                              pose_probs * self.ms_ssim_loss(_fold(I), _fold(I_hat)).reshape(B, NH)
                           ).sum(dim=-1).mean() # mean for dataparallel
        else:
            ms_ssim_loss = ZERO

        # Critic-based perceptual image distance
        if self.w_img_recon_pd > 1e-6: 
            if self.use_vgg_per_loss:
                p_img_loss = self.w_img_recon_pd * self.PLH.weighted_perceptual_loss(
                                                      I1 = I, I2 = I_hat, weights = pose_probs)
            else:
                p_img_loss = self.w_img_recon_pd * img_critic(for_gen=None, I_fake=None, 
                                                              compute_hyp_loss = True,
                                                              I1=I, I2=I_hat, p=pose_probs).mean()
        else:
            p_img_loss = ZERO

        # Pose probability entropy
        #pose_prob_ent_loss = self.w_pose_prob_ent * ( 
        #                        torch.log(1e-5 + pose_probs) * pose_probs 
        #                     ).sum(dim=-1).mean()
        
        # Rotation diversity loss (maximize pw distance = minimize negative pw distance)
        rot_div_loss = self.w_R_negent * self.rot_m_negent(R_inferred, pose_probs)

        # Shape adversary loss
        # No need for additional latent regularization on v, as the critic should do it
        #shape_adv_loss = self.w_shape_adv_loss * shape_critic.compute_loss_for_generator(M_pe)
        #shape_adv_loss = self.w_shape_adv_loss * shape_critic.compute_loss_for_generator(M_pe_dgl)
        shape_adv_loss = self.w_shape_adv_loss * shape_critic(for_gen=True, 
                                                    #fakes = (v, delta, M_pe) ).mean()
                                                    fakes = (v, M_pe)
                                                    #fakes = ( v.view(BNH,-1), M_pe.view(BNH,nV,3) ) 
                                                ).mean()
        # 
        # Note: max_probs_only controls whether to use a weighted adversarial loss vs
        # applying it ONLY to the max probability pose hypotheses
        #xi_p_adv_loss = self.w_xi_p_adv_loss * xi_p_critic(for_gen=True, 
        #                                                   v_fake=xi_p,
        #                                                   prob_weights=pose_probs, 
        #                                                   max_probs_only=True).mean()
        #
        #xi_p_weighted_sq_radius = self.w_L2_xi_p * (xi_p.pow(2).mean(dim=-1) * pose_probs).sum(-1).mean()

        # Note we only match xi_p_tilde to have covar 1
        #xi_p_sigma_match_loss = self.w_sigma_match_xi_p * ( 
        #                            covariance(xi_p.view(B*NH, -1)) - torch.eye(dxp).to(xi_p.device) # TODO speed?
        #                        ).pow(2).sum() / dxp 
        #xi_p_weighted_likelihood = self.xi_p_wll * ( model.xi_p_neglogprob( xi_p ) * pose_probs**2 ).mean()

        #
        #xi_T_adv_loss = self.w_xi_T_adv_loss * xi_T_critic(for_gen=True, v_fake=xi_T).mean()
        #xi_T_adv_loss = self.w_xi_T_adv_loss * xi_T_critic(xi_T).mean()
        if self.vae_xi_T:
            xi_T_reg_loss = self.w_xi_T_adv_loss * self.vae_kl(mu_xi_T, logvar_xi_T)
        else:
            assert xi_T.shape[0:2]  == (B,NH)
            assert pose_probs.shape == (B,NH)

            #--- Apply the adv loss ONLY on the best hypotheses ---#
            inds = pose_probs.detach().argmax(dim=1) # Shape: B
            if self.adv_loss_on_hyps_type == 'best':
                # Only apply the xi_T loss to the best hypothesis
                xi_T_best = xi_T.view(B, NH, -1) # NC = 3 or 4
                mls       = xi_T_best.shape # B x NH x dim(xi_T)
                xi_T_best = xi_T_best.gather(dim   = 1, 
                                             index = inds.view(B, 1, 1).expand(-1, -1, mls[2])
                                            ).squeeze(1) # B x dim(xi_T)
                #if len(list(xi_T_critic.parameters())) > 0:
                if self.ganbased_xi_T_critic:
                    xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic(
                                                              for_gen = True, 
                                                              v_fake  = xi_T_best ).mean()
                                                              #v_fake  = xi_T_best.reshape(B*NH, -1) ).mean()
                else:
                    xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic( xi_T_best ).mean()
                    #xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic( xi_T_best ).mean()

            #--- Apply the adv loss to every hypothesis ---#
            elif self.adv_loss_on_hyps_type == 'all':
                if self.ganbased_xi_T_critic:
                    xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic(
                                                              for_gen = True, 
                                                              v_fake  = xi_T.reshape(B*NH,-1) ).mean()
                                                              #v_fake  = xi_T_best.reshape(B*NH, -1) ).mean()
                else:
                    xi_T_reg_loss = self.w_xi_T_adv_loss * xi_T_critic( xi_T.reshape(B*NH,-1) ).mean()

            #--- Apply the adv loss with weights ---#
            elif self.adv_loss_on_hyps_type == 'weighted':
                if self.ganbased_xi_T_critic:
                    raise ValueError('Not done')
                else:
                    # Can't do with SWD
                    raise ValueError('Not done')

        # Mesh regularizations
        shape_reg_loss = self.w_mesh_reg * self.mesh_reg_loss( M_pe )
        #
        intermeds_loss = self.w_mesh_reg * self.intermeds_loss( M_pe_ints ).to(xi_T.device)
        #
        delta_reg_loss = self.w_delta_reg * perturbation_seq_regularizer(delta, self.F) 
        #
        
        # NOTE Turning off -> the shape critic should be doing this
        #latent_def_reg_v = self.w_v_reg * 0.5 * v.pow(2).mean()
        latent_def_reg_v = self.w_v_reg * self.v_regularizer(v).mean()
        #latent_def_reg_v = torch.zeros(1).to(v.device)
        #
        # if (not v_sec is None) and (not xip_sec is None) and (not xit_sec is None):
        #     # TODO pose_probs <- how to handle? Only take max prob one?
        #     sec_inference_loss = self.w_sec * ( (v - v_sec).pow(2).mean() +
        #                                         (xi_p - xip_sec).pow(2).mean() +
        #                                         (xi_T - xit_sec).pow(2).mean() )
        # else:
        #     sec_inference_loss = torch.zeros(1).to(delta_reg_loss.device)
        #

        # Adversarial rerendering loss
        if rerendered_imgs is None or self.w_reren_adv_loss < 1e-6:
            reren_loss = ZERO
        else:
            reren_loss = self.w_reren_adv_loss * img_critic(for_gen = True, 
                                                            I_fake  = rerendered_imgs).mean() 

        # Consistencies of the textures across hypotheses
        #bbinds = pose_probs.detach().argmax(dim=1) # Shape: B
        if self.w_hyp_xi_T_match > 1e-6:

            # Careful: this will break if 'best' is not chosen for the xi_T critic above
            # xi_T_best : B x dim_xi_T
            assert xi_T_best.shape == (B, xi_T.shape[-1])
            tex_consis_loss = self.w_hyp_xi_T_match * (
                                    xi_T_best.unsqueeze(1).detach() - xi_T
                                ).pow(2).mean()

            # bestxi_T_best = xi_T.view(B, NH, -1) # NC = 3 or 4
            #     inds      = pose_probs.detach().argmax(dim=1) # Shape: B
            #     mls       = xi_T_best.shape # B x NH x dim(xi_T)
            #     xi_T_best = xi_T_best.gather(dim   = 1, 
            #                                  index = inds.view(B, 1, 1).expand(-1, -1, mls[2])
            #                                 ).squeeze(1) # B x dim(xi_T)         
            #inferred_texture # B x nH x nV x 3
            
            # best_texture    = inferred_texture.gather(
            #                         dim   = 1,
            #                         index = bbinds.view(B, 1, 1, 1).expand(-1, -1, nV, 3)
            #                   ) # .squeeze(1) # B x 1 x nV x 3
            # tex_consis_loss = self.w_hyp_tex_consistency * (
            #                         best_texture.detach() - inferred_texture
            #                     ).pow(2).mean()

        # Consistency of the translations across hypotheses (prevents chair nans)
        if self.w_hyp_t_match > 1e-6:
            t_inferred = t_inferred.squeeze(-1)
            assert (t_inferred.shape == (B, NH, 3) ) # B x nH x 3
            best_translations = t_inferred.gather(
                                    dim   = 1,
                                    index = inds.view(B, 1, 1).expand(-1, -1, 3)
                              )
            translation_consistency_loss = self.w_hyp_t_match * (
                                                best_translations.detach() - t_inferred
                                            ).pow(2).mean()

        # Cross-pose texture consistency loss
        if not (cptc_data is None):
            (cptc_v_hat, cptc_xi_T_hat, cptc_I_new, cptc_decoded_texture_image, cptc_decoded_texture) = cptc_data
            cptc_v_consis      = self.w_cptc_v_consis * ( (v.detach() - cptc_v_hat).abs() ).mean()
            cptc_xi_T_consis   = self.w_cptc_xi_T_consis * ( 
                                        (xi_T_best.unsqueeze(1).expand(-1,NH,-1).detach() - cptc_xi_T_hat)**2 ).mean()
            if self.w_cptc_teximg_consis > 1e-8:
                assert len(inferred_texture_images.shape) == 5 # B x NH x C x H x W
                assert inferred_texture_images.shape[0:3] == (B, NH, 3)
                #print('wat', inferred_texture_images.shape, cptc_decoded_texture_image.shape)
                assert inferred_texture_images.shape == cptc_decoded_texture_image.shape
                B, NH, Cti, Hti, Wti = inferred_texture_images.shape
                #best_texture_images = inferred_texture_images.gather(dim = 1,
                #                            index = inds.view(B, 1, 1, 1, 1).expand(-1, -1, Cti, Hti, Wti)
                #                        ).squeeze(1) # B x Cti=3 x Hti x Wti
                cptc_teximg_consis = self.w_cptc_teximg_consis * ( 
                                        (inferred_texture_images - cptc_decoded_texture_image)**2 ).mean()
            else:
                cptc_teximg_consis = ZERO
        else:
            cptc_v_consis      = ZERO
            cptc_xi_T_consis   = ZERO
            cptc_teximg_consis = ZERO


        # Total up the loss
        LOSS = ( img_loss +
                 p_img_loss +
                 ms_ssim_loss +
                 #grad_edge_loss +
                 delta_reg_loss +
                 shape_adv_loss +
                 #xi_p_adv_loss +
                 intermeds_loss +
                 shape_reg_loss +
                 #sec_inference_loss +
                 xi_T_reg_loss +
                 #hyp_texture_loss +
                 reren_loss +
                 tex_consis_loss +
                 translation_consistency_loss +
                 rot_div_loss +
                 cptc_v_consis +
                 cptc_xi_T_consis +
                 cptc_teximg_consis +
                 latent_def_reg_v + 
                 adv_cptc_loss
                ) # +
                 #latent_def_reg_v +
                 #xi_p_weighted_likelihood +
                 #xi_p_weighted_sq_radius +
                 #xi_p_sigma_match_loss + 
                 #pose_prob_ent_loss )

        return LOSS, { 'img_recon'    : _c(img_loss),
                       'p_img_dist'   : _c(p_img_loss),
                       'ms_ssim_loss' : _c(ms_ssim_loss),
                       #'grad_img_d' : _c(grad_edge_loss),
                       'S_adv'        : _c(shape_adv_loss),
                       #'xi_p_adv'   : _c(xi_p_adv_loss),
                       'xi_T_reg'     : _c(xi_T_reg_loss),
                       'delta_reg'    : _c(delta_reg_loss),
                       'M_ints_reg'   : _c(intermeds_loss),
                       #'sec_inf'    : _c(sec_inference_loss),
                       'M_reg_c2'     : _c(shape_reg_loss),
                       'tex_consis'   : _c(tex_consis_loss),
                       'trans_consis' : _c(translation_consistency_loss),
                       #'hyp_tex'    : _c(hyp_texture_loss), 
                       'reren_loss'   : _c(reren_loss),
                       'rot_negent'   : _c(rot_div_loss), 
                       'cptc_v_consis'      : _c(cptc_v_consis),
                       'cptc_xi_T_consis'   : _c(cptc_xi_T_consis),
                       'cptc_teximg_consis' : _c(cptc_teximg_consis),
                       'v_reg_cy2'          : _c(latent_def_reg_v),
                       'adv_cptc_loss'      : _c(adv_cptc_loss),
                    }
                       #'v_reg_cy2'  : _c(latent_def_reg_v),
                       #'xi_p_lik'   : _c(xi_p_weighted_likelihood),
                       #'xi_p_sq_r'  : _c(xi_p_weighted_sq_radius),
                       #'xi_p_sig_m' : _c(xi_p_sigma_match_loss),

class IndepSeqMeshRegLoss(nn.Module):
    def __init__(self, mrl):
        super(IndepSeqMeshRegLoss, self).__init__()
        self.mrl = mrl #MeshRegularizationLoss(V, E, F)
    def forward(self, new_V):
        if new_V is None: return torch.zeros(1)
        B, nI, nV, _ = new_V.shape
        out = new_V.reshape(B*nI, nV, 3)
        return self.mrl(out)

class PoseRegularizationLoss(nn.Module):
    def __init__(self):
        super(PoseRegularizationLoss, self).__init__()
        
    def forward(self, R, t):
        # Prefer rotations that are purely azimuthal 
        # Compute the axis of rotation
        B, nH, _, _ = R.shape
        R = R.view(B*nH, 3, 3)
        trR = R[:,0,0] + R[:,1,1] + R[:,2,2]
        Q = R + R.permute(0,2,1) + (1 - trR)*torch.eye(3,3).to(R.device).unsqueeze(0).expand(B*nH,3,3)
        v = Q[:, 0, :].unsqueeze(-1) # B*nH x 3 x 1
        j_hat = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).expand(B*nH,-1).unsqueeze(1).to(v.device) # B*nH x 1 x 3
        # TODO check
        angle = torch.matmul(j_hat, v) # dot product
        return angle.mean()

class MeshRegularizationLoss(nn.Module):
    def __init__(self, V, E, F):
        """ Computes mesh regularization losses. """
        super(MeshRegularizationLoss, self).__init__()
        # Template mesh values
        self.V = V
        self.E = E
        self.F = F
        # Parameters
        self.w_edge = 1500 # 25.0
        self.w_lap  = 10.0 #0.05 #0.05 #10.0 # As in the Softras paper: 10x flat loss
        self.w_flat = 1.0 #0.005 #0.005 #1.0
        # Edge length loss (as in Pixel2Mesh)
        logging.info('Setting up mesh regularization loss')
        #self.edge_loss = lambda new_v: meshutils.mean_sq_edge_len(new_v, self.E)
        # Laplacian loss (as in SoftRas)
        logging.info('\tSetting up Laplacian loss')
        self.lap_loss = meshutils.LaplacianLoss(self.V, self.F, average=True)
        # Flatness loss (as in SoftRas)
        logging.info('\tSetting up flatness loss')
        self.flat_loss = meshutils.FlattenLoss(self.F, average=True)

    def forward(self, pertV, return_dict = False):
        assert len(pertV.shape) == 3 # B x |V| x 3
        #B, nH, nV, _ = pertV.shape
        #pertV = pertV.reshape(B*nH, nV, 3)
        LE = self.w_edge * meshutils.mean_sq_edge_len(pertV, self.E)
        #LE = self.w_edge * meshutils.mean_edge_len_variance(pertV, self.E)
        #LE = LE1 + LE2
        LL = self.w_lap  * self.lap_loss(pertV)
        LF = self.w_flat * self.flat_loss(pertV)
        total = LE + LL + LF
        if return_dict:
            return total, { 'Edge Reg' : _c(LE), 'Laplacian Reg' : _c(LL), 'Flatness Reg' : _c(LF) }
        else:
            return total

def texture_smoothness(texture, F):
    """
    texture: B x |V| x 3
    F: faces, |F| x 3

    texture[:, F] is B x |F| x 3 x 3 = batch x faces x node ID x xyz-coord
    """
    assert len(texture.shape) == 3
    var = texture[:, F].var(dim=2).sum(dim=-1).mean()
    return var

def perturbation_seq_regularizer(delta, F, w_var=1.0, w_mag=0.0, ret_dict = False): #
    """
    delta: B x |I| x |V| x 3
    faces: |F| x 3
    """
    assert len(delta.shape) == 4
    #assert len(delta.shape) == 5
    #B, nH, nI, nV, _ = delta.shape
    #delta = delta.reshape(B*nH, nI, nV, 3)
    # TODO similar magnitude through time, i.e. total magnitude variance over time
    # Compute variance over the faces (delta[:,:,F] is B x |I| x |F| x 3 x 3)
    # @-1=4: coordinate index; @-2=3: node-in-each-face index
    # Compute variance over the delta values per face (3 nodes, 3 coords), summed over coords
    var = w_var * delta[:, :, F].var(dim=3).sum(dim=-1).mean() 
    mag = w_mag * delta.abs().mean()
    #if ret_dict: return var + mag, { 'delta_var_pen' : var, 'delta_mag_pen' : mag }
    if ret_dict: return var + mag,  { 'delta_var_pen' : var, 'delta_mag_pen' : mag }
    return var + mag

def chamfer_loss_object():
    """
    Return an object that computes the chamfer loss.
    Inputs: two B x |V_i| x 3 point sets
    Outputs: (dist1, dist2, idx1, idx2)
    disti is B x |V_i|, same for idxi.

    Chamfer dists:
    - dist of closest point on b of points from a
    - dist of closest point on a of points from b
    - idx of closest point on b of points from a
    - idx of closest point on a of points from b

    From: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
    """
    return chamfer_distance.chamfer_3DDist()


class MS_SSIM_Loss(nn.Module):
    def __init__(self, inchannels, win_size=4):
        super(MS_SSIM_Loss, self).__init__()
        from pytorch_msssim import MS_SSIM
        logging.info('Building MS-SSIM loss object')
        self.Cin = inchannels
        self.ms_ssim = MS_SSIM(data_range       = 1.0, 
                               size_average     = False, 
                               channel          = inchannels, 
                               win_size         = win_size,
                               nonnegative_ssim = True )

    def forward(self, I1, I2):
        """
        Assume I1, I2 are in [-1,1] and B x C x H x W.
        Returns B losses, for each corresponding batch member.
        """
        I1 = (I1 + 1.0) / 2.0
        I2 = (I2 + 1.0) / 2.0
        return 1.0 - self.ms_ssim(I1, I2)


def _c(val):
    return float( val.cpu().detach().numpy() )

#----------------------------------------------------------------------------------#

class Cycle_2_loss_calculator_coupled(nn.Module):

    def __init__(self, V, E, F, options):    
        super(Cycle_2_loss_calculator_coupled, self).__init__()
        logging.info('Setting up Cy2 loss (coupled)')
        self.V = V
        self.E = E
        self.F = F
        # Mesh reg loss
        self.mesh_reg_loss = MeshRegularizationLoss(V, E, F)
        self.intermeds_loss = IndepSeqMeshRegLoss(self.mesh_reg_loss)
        # Function for computing loss between rotation matrices
        self.rot_m_negent = RotationNegEntropyLoss()
        ##### Loss function parameter weights #####
        self.w_img_recon_l1     = options['w_img_recon_l1']
        self.w_img_recon_pd     = options['w_img_recon_pd']
        # Keep adv losses at 1 (so gen loss scale ~=~ critic loss scale)
        self.w_shape_adv_loss   = options['w_shape_adv_loss']
        self.w_xi_p_adv_loss    = options['w_xi_p_adv_loss']
        self.w_xi_T_adv_loss    = options['w_xi_T_adv_loss'] 
        self.w_mesh_reg         = options['w_mesh_reg']
        self.w_delta_reg        = options['w_delta_reg']
        #self.w_sec              = 0.0 # <<<<<<<<<<<< inactive #TODO
        self.w_pose_prob_ent    = options['w_pose_prob_ent']
        self.w_R_negent         = options['w_R_negent']
        #
        # VGG-based perceptual loss
        self.use_vgg_per_loss = options['VGG_per_loss']
        if self.use_vgg_per_loss:
            self.PLH = PerceptualMethodsHandler()
        #self.xi_p_wll           = 0.0
        #self.w_L2_xi_p          = 1.0
        #self.w_sigma_match_xi_p = 0.0
        logging.info({ k : v for k, v in vars(self).items() if type(v) is float })

    def forward(self, I, I_hat, M_pe, xi_p, xi_T, 
                shape_critic, xi_p_critic, xi_T_critic, img_critic,
                v, delta, M_pe_dgl, M_pe_ints, renderer,
                pose_probs, R_inferred, t_inferred, model,
                # These are only needed for secondary inference consistency
                v_sec=None, 
                xip_sec=None, xit_sec=None):
        """ 
        Loss from I -> M -> I_hat cycle 
        
        pose_probs: B x NH 
        """
        # Dimensionalities
        BNH, NC, H, W = I_hat.shape
        B, nV, _ = M_pe.shape
        NH = BNH // B
        #dxp = xi_p.shape[-1]

        # Pose-probability weighted image distance
        I = I.unsqueeze(1).expand(-1,NH,-1,-1,-1)
        I_hat = I_hat.view(B, NH, NC, H, W)
        img_loss = self.w_img_recon_l1 * ( 
                      pose_probs *
                        ( I - I_hat ).abs().mean(dim=-1).mean(dim=-1).sum(dim=-1)
                   ).sum(dim=-1).mean()

        # Critic-based perceptual image distance
        # Note it is rerouted horribly through forward for easy parallelism
        if self.w_img_recon_pd > 1e-7:
            if self.use_vgg_per_loss:
              p_img_loss = self.w_img_recon_pd * self.PLH.weighted_perceptual_loss(
                                                      I1 = I, I2 = I_hat, weights = pose_probs)
            else:
              p_img_loss = self.w_img_recon_pd * img_critic(for_gen=None, I_fake=None, 
                                                            compute_hyp_loss = True,
                                                            I1=I, I2=I_hat, p=pose_probs).mean()
        else:
            p_img_loss = torch.zeros(1).to(I.device)

        # Pose probability entropy
        pose_prob_ent_loss = self.w_pose_prob_ent * ( 
                                torch.log(1e-5 + pose_probs) * pose_probs ).sum(dim=-1).mean()
        
        # Rotation diversity loss (maximize pw distance = minimize negative pw distance)
        rot_div_loss = self.w_R_negent * self.rot_m_negent(R_inferred)

        # Shape adversary loss
        #shape_adv_loss = self.w_shape_adv_loss * shape_critic.compute_loss_for_generator(M_pe)
        #shape_adv_loss = self.w_shape_adv_loss * shape_critic.compute_loss_for_generator(M_pe_dgl)
        shape_adv_loss = self.w_shape_adv_loss * shape_critic(for_gen=True, 
                                                    #fakes = (v, delta, M_pe) ).mean()
                                                    fakes = (v, M_pe) ).mean()
        # 
        # Note: max_probs_only controls whether to use a weighted adversarial loss vs
        # applying it ONLY to the max probability pose hypotheses
        xi_p_adv_loss = self.w_xi_p_adv_loss * xi_p_critic(for_gen=True, 
                                                           v_fake=xi_p,
                                                           prob_weights=pose_probs, 
                                                           max_probs_only=True).mean()
        #
        #xi_p_weighted_sq_radius = self.w_L2_xi_p * (xi_p.pow(2).mean(dim=-1) * pose_probs).sum(-1).mean()

        # Note we only match xi_p_tilde to have covar 1
        #xi_p_sigma_match_loss = self.w_sigma_match_xi_p * ( 
        #                            covariance(xi_p.view(B*NH, -1)) - torch.eye(dxp).to(xi_p.device) # TODO speed?
        #                        ).pow(2).sum() / dxp 
        #xi_p_weighted_likelihood = self.xi_p_wll * ( model.xi_p_neglogprob( xi_p ) * pose_probs**2 ).mean()

        #
        xi_T_adv_loss = self.w_xi_T_adv_loss * xi_T_critic(for_gen=True, v_fake=xi_T).mean()
        #
        shape_reg_loss = self.w_mesh_reg * self.mesh_reg_loss( M_pe )
        #
        intermeds_loss = self.w_mesh_reg * self.intermeds_loss( M_pe_ints ).to(xi_T.device)
        #
        #delta_reg_loss = self.w_delta_reg * single_additive_perturbation_variance_regularizer(delta, self.F) 
        delta_reg_loss = self.w_delta_reg * perturbation_seq_regularizer(delta, self.F) 
        #
        if (not v_sec is None) and (not xip_sec is None) and (not xit_sec is None):
            sec_inference_loss = self.w_sec * ( (v - v_sec).pow(2).mean() +
                                                (xi_p - xip_sec).pow(2).mean() +
                                                (xi_T - xit_sec).pow(2).mean() )
        else:
            sec_inference_loss = torch.zeros(1).to(delta_reg_loss.device)
        #
        LOSS = ( img_loss +
                 p_img_loss +
                 delta_reg_loss +
                 shape_adv_loss +
                 xi_p_adv_loss +
                 intermeds_loss +
                 shape_reg_loss +
                 sec_inference_loss +
                 xi_T_adv_loss +
                 rot_div_loss +
                 #xi_p_weighted_likelihood +
                 #xi_p_weighted_sq_radius +
                 #xi_p_sigma_match_loss + 
                 pose_prob_ent_loss )

        return LOSS, { 'img_recon'  : _c(img_loss),
                       'p_img_dist' : _c(p_img_loss),
                       'S_adv'      : _c(shape_adv_loss),
                       'xi_p_adv'   : _c(xi_p_adv_loss),
                       'xi_T_adv'   : _c(xi_T_adv_loss),
                       'delta_reg'  : _c(delta_reg_loss),
                       'M_ints_reg' : _c(intermeds_loss),
                       'sec_inf'    : _c(sec_inference_loss),
                       'M_reg_c2'   : _c(shape_reg_loss),
                       'rot_negent' : _c(rot_div_loss),
                       #'xi_p_lik'   : _c(xi_p_weighted_likelihood),
                       #'xi_p_sq_r'  : _c(xi_p_weighted_sq_radius),
                       #'xi_p_sig_m' : _c(xi_p_sigma_match_loss),
                       'posep_ent'  : _c(pose_prob_ent_loss) } # TODO also loss weights



#------#

def histogram_entropy(h):
    assert len(h.shape) == 2 # B x nbins
    # Normalize
    h = h / h.sum(dim = 1, keepdim = True).clamp(1e-5)
    # Log-probabilities
    return -( h * torch.log(h.clamp(1e-5)) ).sum(dim = 1) # shape: B

def global_textural_stddev(texture):
    """
    Designed to give a small signal to the UNobserved portions of the car
    """
    assert len(texture.shape) == 3 and texture.shape[-1] == 3 # B x |V| x 3
    return texture.std(dim = 1).mean()

def global_textural_stddev_img(texture_img):
    """
    Designed to give a small signal to the UNobserved portions of the car
    """
    assert len(texture_img.shape) == 4 and texture_img.shape[1] == 3 # B x C x H x W
    B, C, H, W = texture_img.shape
    return texture_img.view(B, C, H*W).std(dim = -1).mean()










#
