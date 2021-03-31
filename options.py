import os, sys, torch, numpy as np, logging

OPTIONS = {
    'BASE' : { # DEFAULT basic hyper-parameters (used in CORE runs)
        'allow_overwrite'                   : True,
        # Critic iterations
        'img_critic_iters_per_gen_iter'        : 1,   # 
        'img_critic_use_cy2_renders_chance_fp' : 0.5, # Chance of replacing a real img with its recon for the img critic
        'cy2_critics_update_per_gen_iter'      : 1,   # Texture critic, [xi_T critic if learned], [[xi_P critic if present]]
        'shape_critic_updates'                 : 1,   # Number of update for the shape critic
        # Model parameters and settings
        'fixed_eye'           : [0.0, 0.0, 1.2],
        'nts'                 : 1, # num impulses = num perturbation time steps
        'num_pose_hypotheses' : 4, # 4,
        'rrm'                 : '3a', # rotation representation method: 'uq', '6d', '3a'
        'pose_prob_method'    : 'softmax', # One of 'softmax', 'simplex', 'allan'
        'VGG_per_loss'        : False, # Use vgg perceptual loss
        'VGG_path'            : '/vol/genren-models/vgg16-397923af.pth',
        # Takes only a subset of the SHAPE data
        'subset_num'          : None,
        'shape_data_dup'      : 1,
        'shape_data_type'     : 'PN', # points+normals data
        # Takes only a subset of the IMAGE data
        'img_data_subset'     : None, # Set to none to use all data
        # Template controls
        'template_path'       : None,
        'manual_template_UV'  : None,
        'subdivide_template'  : False, # quadruples n_faces
        'learn_template'      : False,
        'icosasphere_subdivs' : 10, # 12, # 8, # 10, # Only matters when using a sphere template (template_path is None)
        'use_vae_for_v'       : False, # Use a PointNet VAE (beta = w_v_reg)
        'img_size'            : 64, #          
        ### Texture inferrer settings
        # Whether to detach (no_grad) the computation of the depth/occlusion info
        'use_occlusion'                 : False, #True, # Whether to use occlusion info at all  
        'detach_occ_info'               : True, # Whether to *not* back-prop through occlusion info
        'backprop_through_unprojection' : True,
        'detach_M_E_in_tex_inference'   : False,  
        'vae_for_xi_T'                  : False,
        ### Texture decoder settings
        'detach_v_for_texture_decoder'  : True, # 
        'large_tex_dec'                 : True,
        'tex_dec_norm_type'             : 'bn', # 'sn+bn'
        'tex_img_gen_type'              : '64', # or '32'
        ### Adversarial settings (images, latents, & texture especially) ###
        'pose_buffer_size'             : 128, # Number of stored poses
        'num_swd_projections_xi_T'     : 150, # Only applies if using SWD
        'texture_critic_type'          : 'uvimage', # {uvimage, histo, uvimage+histo}        
        'img_critic_type'              : 'wgangp', # 'lsgan',
        'img_critic_arch'              : '3scale', # ONLY FOR NON-WGAN ARCHITECTURES/TYPES # 'patch' '3scale' 'spec' 
        'wgan_img_mini_critic'         : 'full',   # ONLY IF USING WGAN-GP img critic (in {full, mini})
        'texture_critic_arch'          : 'wgan',   # lsgan or wgan
        'use_SWD_loss_for_xi_T'        : True, # Use VAE for xi_T vs SWD vs GAN critic -> False to use vector critic; set vae_for_xi_T to use VAE instead
        'adv_loss_on_hyps_type'        : 'best', # One of {'weighted', 'all', 'best'} [For loss on xi_T]
        'tex_critic_gp_pen_weight_fp'  : 10.0,
        'img_critic_gp_pen_weight_fp'  : 10.0,
        'histo_type'                   : 'rgb',
        'shape_critic_type'            : 'com', # 'single_stage', 
        # The type of normalization to use in generator subnetworks
        # Purely inference networks are not affected
        'generator_normalization' : 'bn', # 'sn+bn', 'sn', 'bn'
        # Whether to use the uniformly random unit quat sampling for xi_p
        # and the 4D rotation transformation method
        'use_quat_sampling' : False,
        # Preloading path for mesh AE
        'mesh_ae_load_path' : None,
        # Controls alpha channel usage in both renders and dataloader
        'use_alpha'         : True, # False # 
        # Number of sample points from meshes 
        'n_pc_points'              : 1000, # Sampling from true meshes
        'n_template_point_samples' : 1000, # sampling from templates
        # Minimum and maximum translation limits for generation and inference
        # This does NOT affect the domain randomized pose generation
        'translation_mins' : [-0.50, -0.50, -0.50], # x,y,z
        'translation_maxs' : [ 0.50,  0.50,  0.60],
        # Dimensionalities
        'dim_xi_p'      : 16, # mapped to dim(R) + dim(t) = 6 + 3 iff sample_quats is true
        'dim_xi_T'      : 128, # 
        'dim_lat_pert'  : 64, # dim(v), where v -> delta
        'dim_backbone'  : 512, # Used to get xi_T, v, and (r,t)
        # Critic learning rates
        'imc_lr'        : 0.0001, # Img critic LR
        'shc_lr'        : 0.0001, # Shape critic LR
        'xip_lr'        : 0.0009, # Only used if xi_p adv loss is learned
        'xit_lr'        : 0.0001, # Only used if xi_T adv loss is learned
        'C_lr'          : 0.0001, # Learning rate for texture critic (cy1 look like cy2)
        # Model learning rate
        'gen_lr_main'   : 0.0004,        
        # Optimizer settings
        'beta1_fp'      : 0.5,
        'beta2_fp'      : 0.99,
        # Ground truth usage
        'USE_GT'        : False,
        'init_gt_iters' : None, # Happens AFTER the chamfer-only iters
        #### Cycle 1 loss weights ####
        # Mesh AE
        'w_cham_orig'              : 2000.0, # 2000 # 750.0,
        'w_normals_orig'           : 5.0,
        # Shape reconstruction
        'w_cham_recon'             : 0.0, #300.0,
        'w_normals_recon'          : 0.0, #5.0,
        'w_l2_con'                 : 1000.0, # 
        'w_v_recon'                : 5.0, # 
        # Adversaries 
        'w_pose_adv'               : 10.0, # Distribution matching loss on the pose
        'w_adv_img'                : 0.2, 
        'w_rot_recon'              : 5.0,
        'w_t_recon'                : 10.0,
        'w_xi_T'                   : 5.0, # Reconstruction 
        'w_texture_recon'          : 0.0, # Don't learn anything about what those textures should be
        'w_texture_recon_PTS2_FDR' : 10.0, # Texture reconstruction during pretraining [pretraining only]
        'w_texture_realism'        : 0.2, # Adv loss on cy1 textures to match cy2 outputs
        'w_tex_varia_pen'          : 0.0, # Penalty to promote texture smoothness 
        'w_tex_stddev_global'      : 0.0, # Penalty on global stddev for unobserved parts
        #### Cycle 2 loss weights ####
        'w_img_recon_l1'           : 40.0, #50.0, 
        'w_img_recon_pd'           : 200.0, # 10K for vgg, remove for critic percept loss
        'w_img_recon_ms'           : 0.0,  # MS-SSIM loss between images
        'w_shape_adv_loss'         : 1.0, # Realistic v
        'w_xi_T_adv_loss'          : 5.0,  # adv loss or KL div VAE loss OR SWD loss weight [see use_SWD opt below!]
        'w_xi_p_adv_loss'          : 0.0,  # Only used in the COUPLED case
        'w_hyp_xi_T_match'         : 1.0,  # Push hyps to look like the best one (in latent texture)
        'w_hyp_t_match'            : 10.0,  # Hypothesis matching (translation) 
        'w_reren_adv_loss'         : 0.0,  # Adversarial re-rendering loss (random view) [can be destabilizing!]
        'run_rerenderings'         : False, # Must be true if trying to use reren adv loss
        #### Shared reg losses ####
        'w_R_negent'               : 0.1,
        'w_pose_prob_ent'          : 0.0,
        'w_mesh_reg'               : 0.05, 
        'w_delta_reg'              : 0.05, #0.005,
        'w_v_reg'                  : 5.0, # VAE KL div (if v is computed with a VAE) OR other (L2 norm, swd)
        'w_v_reg_cy2'              : 0.1,
        'v_reg_type'               : 'swd', # or 'L2'
        'v_swd_nprojs'             : 150, # Only applies when using SWD for v-reg
        ### Cross-pose texture consistency
        'enforce_cross_pose_tex_consis' : False, # Cy2 cyclic consistency
        'w_cptc_v_consis'               : 0.25,
        'w_cptc_xi_T_consis'            : 0.25, 
        'w_cptc_teximg_consis'          : 0.0,
        'use_CY1_cptc'                  : False, # Cy1 cyclic consistency
        'w_cy1_cptc_v_consis'           : 0.25,
        'w_cy1_cptc_xi_T_consis'        : 0.25, 
        'w_cy1_cptc_teximg_consis'      : 0.0,
        'use_mixed_reren_loss'          : False, # Additional cy1 consistency
        'w_mixed_cy1_reren'             : 1.0,
        'use_adversarial_cptc'          : True, # Adversarial CPTC consistency
        'w_adversarial_cptc_cy1'        : 0.0, # Dangerous when non-zero
        'w_adversarial_cptc_cy2'        : 0.35,
        ### Training regime options
        # Freezing iters happen right after pretraining [s1]. DR will use the frozen mesh AE.
        'freeze_mesh_ae_iters'     : 0, #1000000, # < Permanent
        'cy2_loss_only_iters'      : 0, # Also happens right after pretraining, meaning freezing may be concurrent
        'save_pretrained_mesh_ae'  : False,
        'save_dr_pretrained_model' : False,
        'save_main_model'          : True,
        'save_latest_model_every'  : 2000, # Overwrites
        'save_model_every'         : 20000, # for main model [model+statedict]
        'save_only_latest_model'   : True,
        # Batch sizes
        'B_imgs'               : 64, # 50, 
        'B_shapes'             : 64, # 50, 
        # Pretraining stage 2 (domain-randomized) settings
        'pts2_lt_start_prob'   : 0.01, # Weight on learned texture loss (start)
        'pts2_lt_final_prob'   : 0.99, # Weight on learned texture loss (end)
        # Total number of iterations
        'n_gen_iters'          : 99999,
        # Pretraining stage 1 learning rates [chamfer-only]
        'gen_lr_pt1'           : 0.0008,
        'gen_lr_pt1_gamma'     : 0.5,
        'gen_lr_pt1_ms'        : [ 4000 ], 
        # Pretraining iteration counts
        'chamfer_only_iters'     : 5000,
        'stage_2_pretrain_iters' : 3000,
        # Main training phase settings
        'initial_cy2_weight'   : 0.01,  # Initial weight on cy2
        'cy2_annealing_period' : 1000, #1000, # Num iters to anneal cy2 weight up to 1 [during mode 1]
        'mode_1_iters'         : 3000, #3000, # num iters in mode 1 (cy2 [annealed] + LT-DR for cy1)
        'mode_2_iters'         : 3000, #1500, # num iters in mode 2 (annealing full cy1 up + LTDR down)
        #------------------------#
        # MAE options #
        'mae_eta_lr'          : 0.001,
        'mae_n_iters'         : 6000,
        'mae_print_every'     : 500,
        'mae_n_sample_points' : 1000, # Controls both template and target sampling
        'mae_gamma'           : 0.1,
        'mae_lr_schedule'     : [ 5000 ],
        'mae_w_cham'          : 1000.0, # 750.0,
        'mae_w_emd'           : 0.0, #1000.0, # 750.0,
        'mae_w_normals'       : 5.0,
        'mae_w_mesh_reg'      : 0.01, #0.005, 
        'mae_w_delta_reg'     : 0.01, #0.005,
    },

    'test' : { # Example
        'COPY_FROM'              : 'shapenet',
        'TEST_img_data_dir'      : '/path/to/TEST/images',
        'TEST_shapes_data_dir'   : '/path/to/TEST/shapes',
        'img_data_dir'           : './cabinet-mini/images-test-cabinet-02933112-mini',
        'shape_data_dir'         : './cabinet-mini/shapes-test-cabinet-02933112-mini',
        'chamfer_only_iters'     : 100,
        'template_path'    : None,
        'template_scale'   : None, # 1.0,
        # Set these to use a template
        #'template_path'      : './Models/plane-template/plane-UVs-16cones.obj',  
        #'manual_template_UV' : './Models/plane-template/plane-UVs-16cones.obj',  
        #'template_scale'     : 0.97,
        #---#
        'stage_2_pretrain_iters' : 100,
        'mode_1_iters'           : 20,
        'mode_2_iters'           : 20,
        'cy2_annealing_period'   : 10, 
        'out_dir_prepen'   : '',
        'data_scale'       : None, # 1.0,
        'B_imgs'           : 8,
        'B_shapes'         : 8,
        'print_every'      : 5,
        'save_imgs_every'  : 9,  
    },

    ################################################################################

    'shapenet' : {
        'COPY_FROM'        : 'BASE',
        'template_R_angle' : torch.FloatTensor([0.0]).unsqueeze(0),
        'template_R_axis'  : torch.FloatTensor(np.array([1,0,0])).unsqueeze(0),
        'data_R_angle'     : torch.FloatTensor([0.0]).unsqueeze(0),
        'data_R_axis'      : torch.FloatTensor(np.array([1,0,0])).unsqueeze(0),  
        'shape_data_dup'   : 1,    
        'print_every'      : 200,
        'save_imgs_every'  : 1000,  
        'preload_img_data' : True, # False,
    },
    #----#
    'chairs' : {
        'COPY_FROM'        : 'shapenet',
        'TEST_img_data_dir'    : '/path/to/TEST/images',
        'TEST_shapes_data_dir' : '/path/to/TEST/shapes',
        'img_data_dir'         : '/path/to/TRAIN/images',
        'shape_data_dir'       : '/path/to/TRAIN/shapes',
        'out_dir_prepen'   : '/path/to/genren-output',
        'template_path'    : None,
        'template_scale'   : None, # 1.0,
        'data_scale'       : None, # 1.0,
    },
    #----#
    'sofa' : {
        'COPY_FROM'        : 'shapenet',
        'TEST_img_data_dir'    : '/path/to/TEST/images',
        'TEST_shapes_data_dir' : '/path/to/TEST/shapes',
        'img_data_dir'         : '/path/to/TRAIN/images',
        'shape_data_dir'       : '/path/to/TRAIN/shapes',
        'out_dir_prepen'   : '/path/to/genren-output',
        'template_path'    : None,
        'template_scale'   : None, # 1.0,
        'data_scale'       : None, # 1.0,
    },
    #----#
    'cabinet' : {
        'COPY_FROM'        : 'shapenet',
        'TEST_img_data_dir'    : '/path/to/TEST/images',
        'TEST_shapes_data_dir' : '/path/to/TEST/shapes',
        'img_data_dir'         : '/path/to/TRAIN/images',
        'shape_data_dir'       : '/path/to/TRAIN/shapes',
        'out_dir_prepen'   : '/path/to/genren-output',
        'template_path'    : None,
        'template_scale'   : None, # 1.0,
        'data_scale'       : None, # 1.0,
    },
    #----#
    'planes' : { # Templated 
        'COPY_FROM' : 'planes-occ7',
        'template_path'      : './Models/plane-UVs-16cones.obj',  
        'manual_template_UV' : './Models/plane-UVs-16cones.obj',  
        'template_scale'     : 0.97,
        'w_img_recon_l1' : 100,
        'w_img_recon_pd' : 100,
        'w_v_reg'        : 0.5,
        'img_critic_use_cy2_renders_chance_fp' : 0.5,
        'w_adversarial_cptc_cy2'               : 0.0,
    },
    'planes-base' : {
        'COPY_FROM'        : 'shapenet',
        'TEST_img_data_dir'    : '/path/to/TEST/images',
        'TEST_shapes_data_dir' : '/path/to/TEST/shapes',
        'img_data_dir'         : '/path/to/TRAIN/images',
        'shape_data_dir'       : '/path/to/TRAIN/shapes',
        'out_dir_prepen'   : '/path/to/genren-output',
        'template_path'      : None,
        'template_scale'     : None,
        'data_scale'         : None, #0.97,  
        'manual_template_UV' : None,      
    },
    'planes-occ7' : {
        'COPY_FROM' : 'planes-base',
        'img_critic_use_cy2_renders_chance_fp' : 0.95, #
        'learn_template'    : True,
        'icosasphere_subdivs' : 10, # <<< New: use a small sphere to emulate the older template
        #'use_occlusion'     : False, # False for performance reasons o_o'
        'dim_lat_pert'      : 128,
        'w_cham_orig'       : 2000.0, # Same
        'w_normals_orig'    : 5.0,    # Same
        'w_cham_recon'      : 0.0,    # Same
        'w_normals_recon'   : 0.0,    # Same
        'w_l2_con'          : 1000.0, # Same
        'w_v_recon'         : 1.0,
        'w_pose_adv'        : 10.0, # Same
        'w_adv_img'         : 0.5,
        'w_rot_recon'       : 1.0,
        'w_t_recon'         : 10.0, # Same
        'w_xi_T'            : 1.0,
        'w_texture_recon'   : 10.0,
        'w_texture_realism' : 1.0,
        'w_img_recon_l1'    : 50.0,
        'w_img_recon_pd'    : 1.0,
        'w_img_recon_ms'    : 0.0, # Same
        'w_shape_adv_loss'  : 1.0, # Same
        'w_xi_T_adv_loss'   : 1.0,
        'w_xi_p_adv_loss'   : 0.0, # Same
        'w_R_negent'        : 0.1, # Same
        'w_pose_prob_ent'   : 0.0, # Same
        'w_mesh_reg'        : 0.005,
        'w_delta_reg'       : 0.1,
        'w_v_reg'           : 0.1,
        'w_v_reg_cy2'       : 0.1, # 
    },
    #----#
    'cars' : {
	'COPY_FROM' 	   : 'shapenet',
        'TEST_img_data_dir'    : '/path/to/TEST/images',
        'TEST_shapes_data_dir' : '/path/to/TEST/shapes',
        'img_data_dir'         : '/path/to/TRAIN/images',
        'shape_data_dir'       : '/path/to/TRAIN/shapes',
        'out_dir_prepen'   : '/path/to/genren-output',
        'template_path'    : None, # icosa sphere
        'template_scale'   : None, # 1.0,
        'data_scale'       : None, # 1.0,
    },
}

#---------------------------------------------------------------------------------------------#

def str_keys():
    return [ 'template_path', 'img_data_dir', 'shape_data_dir', 'out_dir_prepen', 'rrm',
             'mesh_ae_load_path', 'generator_normalization', 'tex_dec_norm_type',
             'img_critic_type', 'wgan_img_mini_critic', 'img_critic_arch',
             'texture_critic_arch', 'pose_prob_method', 'adv_loss_on_hyps_type',
             'texture_critic_type', 'histo_type', 'shape_critic_type', 'tex_img_gen_type',
           ]

def int_keys():
    return [ 'dim_xi_p', 'dim_xi_T', 'dim_lat_pert', 'dim_backbone', 'B_shapes', 'B_imgs', 
             'img_size', 'num_pose_hypotheses', 'n_gen_iters', 'chamfer_only_iters', 
             'freeze_mesh_ae_iters', 'shape_data_dup', 'nts', 'stage_2_pretrain_iters', 
             'print_every', 'img_critic_iters_per_gen_iter', 'mode_1_iters', 'mode_2_iters',
             'img_data_subset', 'icosasphere_subdivs',
           ]

def bool_keys():
    return [ 'preload_img_data', 'VGG_per_loss', 'use_alpha', 'use_quat_sampling', 
             'use_occlusion', 'use_SWD_loss_for_xi_T', 'large_tex_dec', 'use_vae_for_v',
             'detach_v_for_texture_decoder', 'enforce_cross_pose_tex_consis', 
             'detach_M_E_in_tex_inference', 'use_CY1_cptc', 'use_mixed_reren_loss',
             'use_adversarial_cptc', 'save_only_latest_model', 'learn_template'
           ]

#---------------------------------------------------------------------------------------------#

def all_keys():
    keys = OPTIONS.keys()
    A = []
    for k in keys:
        curr_d = OPTIONS[k]
        A += list( curr_d.keys() )
    return list( set(A) )

def get_options(c, curr_dict=None):
    """
    Recursively gathers options. At each stage, places all the current keys into the dict
    being constructed, then either returns or runs the function again on the next level of
    COPY_FROM (i.e., next recursion). No key is ever overwritten in the process, thus we 
    progressively fill in from the top of the hierarchy.
    """
    logging.info('Retrieving options from', c)
    # Solid dict - never overwritten, starts empty
    solid_dict = {} if curr_dict is None else curr_dict
    solid_keys = solid_dict.keys()
    # Current dict
    curr_dict = OPTIONS[c]
    # Go through the current dict and add anything corresponding to keys NOT solidly present
    for key, value in curr_dict.items():
        # Skip present keys <- never overwriting
        if key in solid_keys: continue
        # Skip the COPY_FROM <- it's handled afterwards, to choose whether to pass down or not
        if 'COPY_FROM' == key: continue
        # Otherwise, copy over the entry
        solid_dict[key] = value
    # We have now copied over any previously unset keys into the current options set
    # So we are done with the options set described by c
    # Now, if this current set was inheriting from another one, we need to get all those 
    # lower entries. Otherwise, we can return with what we have
    # See if the current dict includes COPY_FROM
    if 'COPY_FROM' in curr_dict.keys():
        return get_options(curr_dict['COPY_FROM'], solid_dict)
    else:
        return solid_dict

def get_options_old(c):
    opts = OPTIONS[c]
    keys = opts.keys()
    if 'COPY_FROM' in keys:
        # Get the options list to copy from as a starting point
        opts_new = OPTIONS[opts['COPY_FROM']]
        # For each key in the target set, override the value from the starting point
        for key in keys:
            # Skip the 'copy_from' flag
            if key == 'COPY_FROM': continue
            # Overwrite the value 
            opts_new[key] = opts[key]
        return opts_new
    else:
        return opts

def optformat(d, tab=2):
    s = ['{\n']
    for k, v in d.items():
        if isinstance(v, dict):
            v = format(v, tab+1)
        else:
            v = str(v) # repr(v)
        s.append('%s%r: %s,\n' % ('  '*tab, k, v))
    s.append('%s}' % ('  '*tab))
    return ''.join(s)

#
