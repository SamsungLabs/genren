import torch, torch.nn as nn, torch.nn.functional as F, shutil, itertools, imageio
import numpy as np, numpy.random as npr, trimesh, math, argparse, os, sys, shutil

from model import CyclicGenRen
from renderer import SrRenderer
import imgutils, shapedata, meshutils, graphicsutils, utils
from options import *
from utils import bool_string_type
import torchvision.utils as TU

def main():

    parser = argparse.ArgumentParser(description = 'GenRen testing script')
    parser.add_argument('model_path', 
            type = str, 
            help = 'Path to model file')
    parser.add_argument('outpath', 
            type = str, 
            help = 'Path to output dir')
    parser.add_argument('mode',
            type = str,
            choices = [ 'img_recon', 'shape_recon' ],
            help    = 'Task to perform')
    parser.add_argument('--imgs_dir',
            type = str,
            help = 'Specify the directory with images on which operate')
    parser.add_argument('--shapes_dir',
            type = str,
            help = 'Specify the directory with PN-shapes on which operate')
    parser.add_argument('--random_subset_size',
            type = int,
            default = 16,
            help    = "Choose to operate on a random subset of given size from the data folder")
    parser.add_argument('--options_choice', 
            type = str, 
            help = 'Name of options set (needed to load a state_dict, etc...)')
    parser.add_argument('--allow_overwrite',
            type = bool_string_type,
            help = 'Whether to allow overwriting the output folder')
    parser.add_argument('--subset_initial',
            type = bool_string_type,
            default = False,
            help = 'Whether to use the first n initial ordered data points (rather than random) for reproducible visualizations')
    parser.add_argument('--device', 
            type = str, 
            help = 'GPU device number', 
            default = '0')
    args = parser.parse_args()
    
    #-------------------------------------#
    model_path     = args.model_path
    outpath        = args.outpath
    eval_mode      = args.mode == 'shape_recon'
    IMSIZE         = 64
    subset_initial = args.subset_initial
    #-------------------------------------#

    construct_renderer = True
    device = torch.device('cuda:%s' % args.device)

    ##### State Dict Loading #####
    if 'state' in model_path:
        assert not args.options_choice is None, "options_choice must be specified"
        # Read inputs arguments (if needed)
        options = get_options( args.options_choice )
        # Initialize renderer
        if construct_renderer:
            print('Building renderer')
            renderer = SrRenderer(IMSIZE).to(device)
            renderer.set_eye( torch.FloatTensor(options['fixed_eye']) )
            renderer.move_at_up(device)
        else:
            renderer = None
        # Initialize template
        use_predefined_template = not ( options['template_path'] is None )
        template_path = options['template_path']
        if use_predefined_template:
            template, t_faces = meshutils.read_surface_mesh(template_path, 
                                    to_torch=True, subdivide=options['subdivide_template'])
            template_scale    = options['template_scale']
            template_R_angle  = options['template_R_angle']
            template_R_axis   = options['template_R_axis']
            template       = meshutils.norm_mesh(template, scale = template_scale)
            template       = meshutils.rotate(template_R_angle, template_R_axis, template)
            template_mesh  = (template, t_faces)
        else:
            template_mesh  = None
            template_scale = None
        learn_template = options['learn_template']
        # Construct generative model
        print('Constructing model')
        cyclic_genren = CyclicGenRen(
                                dim_xi_T                     = options['dim_xi_T'],
                                dim_xi_p                     = options['dim_xi_p'],
                                dim_lat_pert                 = options['dim_lat_pert'],
                                dim_backbone                 = options['dim_backbone'],
                                num_impulses                 = options['nts'],
                                num_hypotheses               = options['num_pose_hypotheses'],
                                use_alpha                    = options['use_alpha'],
                                options                      = options,
                                template_mesh                = template_mesh,
                                learn_template               = learn_template,
                                parallelize                  = [device],
                                rotation_representation_mode = options['rrm'],
                                FDR_pixel_distribution       = None, 
                                renderer                     = renderer )
        print('\tLoading state dict from', model_path)
        cyclic_genren.load_state(model_path, eval_mode = eval_mode)

    ##### Full Model Loading #####
    else:
        print('Loading model (full) from', model_path)
        cyclic_genren = CyclicGenRen.load_model(model_path, eval_mode = eval_mode)
        print('Retrieving renderer')
        renderer = cyclic_genren.lat_texture_inferrer.ren_obj
        renderer.move_at_up(device)

    # Model device handling
    cyclic_genren = cyclic_genren.to(device)
    print('\tModel construction complete')

    # Output check
    out_exists = os.path.isdir(outpath)
    if not args.allow_overwrite:
        assert not out_exists, "Output path already exists (%s)" % outpath
    assert os.path.isfile(model_path), "Model path does not exist (%s)"  % model_path
    if not out_exists:
        os.makedirs(outpath)

    ##### Perform the designated task #####
    #-------------------------------------------------------------------------------------------------------------#
    if   args.mode == 'img_recon': # Vision cycle
        assert (not args.imgs_dir is None) and os.path.isdir(args.imgs_dir), 'imgs are required for ' + args.mode
        write_reconstructions(cyclic_genren, renderer, args.imgs_dir, outpath, device, 
                              save_combined_best = None,
                              resize             = None,
                              subset_initial     = subset_initial,
                              random_subset_size = args.random_subset_size)
    elif args.mode == 'shape_recon': # Graphics cycle
        assert (not args.shapes_dir is None) and os.path.isdir(args.shapes_dir), 'shapes are required for ' + args.mode  
        generate_random_images_from_shapes(cyclic_genren, renderer, args.shapes_dir, 
                                           num_images_per_shape = 3,
                                           output_dir           = outpath, 
                                           device               = device, 
                                           copy_original_input  = True,
                                           combine_all_imgs     = False,
                                           resize               = None, 
                                           full_cycle_1         = True,
                                           random_subset_size   = args.random_subset_size)

#####################################################################################################################

def write_reconstructions(model, renderer, images_dir, output_dir, device, 
                          fixed_elev                   = math.pi / 5.0, 
                          mark_correspondences         = False,
                          resize                       = None,
                          copy_original_input          = True,
                          corresponding_points_targets = None,
                          num_corres_points            = None,
                          save_combined_best           = None,  # 
                          subset_initial               = False, # 
                          random_subset_size           = None,
                          save_ply                     = True, 
                          num_views                    = None,
                          recon_set_mode               = False, # 
                          corres_set_mode              = False, # 
                          corres_colours               = 'Okabe-Ito',
                          animated                     = False, 
                        ):
    from PIL import Image, ImageDraw
    ES = 64 # Expected img size
    if save_combined_best == False:
        save_combined_best = None
    assert corres_colours in ['Okabe-Ito', 'random']
    if type(corresponding_points_targets) is str:
        assert not num_corres_points is None
    else:
        assert num_corres_points is None

    if mark_correspondences:
        n_targets = num_corres_points if (not num_corres_points is None) else len(corresponding_points_targets)
        if corres_colours == 'random':
            random_colours = graphicsutils.generateDifferentRandomColors(n_targets, as_ints=True) 
        elif corres_colours == 'Okabe-Ito':
            random_colours = graphicsutils.okabe_ito_colours()[0 : n_targets]
    
    NUM_AZIS = 6 if num_views is None else (num_views + 1)
    AZIS     = np.linspace(0, 2*np.pi, NUM_AZIS)[:-1] + np.pi / 12.0
    NUM_AZIS = len(AZIS)
    img_data = load_images_data(images_dir, random_subset_size = random_subset_size, 
                                subset_initial = subset_initial)
   
    if not num_views is None:
        assert NUM_AZIS == num_views

    best_com = {}
    if (not save_combined_best is None):
        assert type(save_combined_best) is int
        num_combined_to_save = save_combined_best
        save_combined_best = True
        best_path_out = os.path.join(output_dir, 'COMB.png')
        best_path_out_corres = best_path_out.replace('.png', '-corres.png')

    imgs_com = []
    from torch.utils.data import DataLoader
    BBB = 64
    dataloader = DataLoader(img_data, batch_size=BBB, shuffle=False, sampler=None,
                            batch_sampler=None, num_workers=0, collate_fn=None,
                            pin_memory=False, drop_last=False, timeout=0,
                            worker_init_fn=None)
    nH = model.num_pose_hypotheses
    for datab in dataloader:
        _Q = model.run_cycle_2(datab.to(device), renderer, run_secondary_inference=False)
        currB = _Q[0].shape[0]
        print('CurrB', currB)
        _Q[11] = _Q[11].reshape(currB, nH, 4, ES, ES)
        for i,s in enumerate(_Q): 
            if not s is None:
                assert s.shape[0] == currB, str(i) + ' ' + str(s.shape)
        imgs_com.append(_Q)
    _nn = len(imgs_com[0])  
    newcom = [None for i in range(_nn)]  
    for _u in range(_nn):
        if imgs_com[0][_u] is None:
            newcom[_u] = None
        else:
            newcom[_u] = torch.cat( [ bb[_u] for bb in imgs_com ], dim = 0 )
    n_tot_sams = len(newcom[0])
    jm = [ [] for i in range(n_tot_sams) ] 
    for i in range(n_tot_sams):
        for j in range(_nn):
            if newcom[j] is None:
                jm[i].append( None )
            else:
                jm[i].append( newcom[j][i].unsqueeze(0) )
    assert len(jm) == len(img_data)
    
    mastergif = []

    for i, (img, f) in enumerate(zip(img_data, img_data.files)):
        f_base = os.path.basename(f)
        print('On recon', i, "(%s [%s])" % (f,f_base))
        img = img.unsqueeze(0).to(device)
        ( cy2_M,          # Inferred mesh shape 
          cy2_M_pe,       # Inferred mesh shape before Euclidean transform
          cy2_M_ints,     # Inferred mesh deformation intermediates
          cy2_xi_p,       # Inferred latent pose
          cy2_xi_T,       # Inferred latent texture
          cy2_v,          # Inferred latent deformation
          cy2_R,          # Inferred rotation
          cy2_r,          # Inferred rotation (6D)
          cy2_t,          # Inferred translation
          cy2_delta,      # Inferred perturbation
          cy2_T,          # Reconstructed texture (from inferred latent)
          cy2_renders,    # Reconstructed image render
          cy2_pose_probs, # Pose hypothesis probabilities
          mu_xi_T_hat,    # xi_T VAE mean param (if used)
          logvar_xi_T_hat, # xi_T VAE log-var param (if used)
          texture_img_hat,
          ) = jm[i]
        kj, NH, C, H, W = cy2_renders.shape
        assert kj == 1
        nV = cy2_M.shape[-2]
        ML_images     = cy2_renders.view(1, NH, 4, H, W) 
        inds          = cy2_pose_probs.detach().argmax(dim = 1)
        mls           = ML_images.shape
        ML_cy2_M      = cy2_M.gather(dim = 1, # B x nH x |V| x 3
                            index = inds.view(1,1,1,1).expand(-1,-1,nV,3)
                        ).squeeze(1) # B=1 x |V| x 3
        ML_images     = ML_images.gather(dim = 1, 
                            index = inds.view(1,1,1,1,1).expand(-1,-1,mls[2],mls[3],mls[4]) 
                         ).squeeze(1) # B=1 x C x H x W
        ML_teximages  = texture_img_hat.gather(dim = 1, 
                            index = inds.view(1,1,1,1,1).expand(-1,-1,3,mls[3],mls[4]) 
                         ).squeeze(1) # B=1 x C x H x W
        ML_textures   = cy2_T.gather(dim = 1, # B x NH x |V| x 3
                            index = inds.view(1,1,1,1).expand(-1,-1,model.nV,3)
                            ).squeeze(1) # B x |V| x 3
        ML_translates = cy2_t.gather(dim = 1, # B x NH x |V| x 3
                            index = inds.view(1,1,1,1).expand(-1,-1,3,1)
                            ).squeeze(1) # 1 x 3 x 1
        recon_error = (ML_images - img).abs().mean().cpu().detach().numpy()
        nonext_fbase = os.path.splitext(f_base)[0]
        ply_path     = os.path.join(output_dir, nonext_fbase + "-inf-" + str(i) + '.ply')
        if save_ply:
            print('Saving ply to', ply_path)
            meshutils.write_textured_ply(ply_path, 
                                         cy2_M_pe.cpu().detach().squeeze(0).numpy(), 
                                         model.template_F.cpu().detach().numpy(), 
                                         ML_textures.cpu().detach().squeeze(0).numpy() )
        teximg_path = os.path.join(output_dir, nonext_fbase + "-inf-teximg-" + str(i) + ".png")
        imgutils.imswrite_t(ML_teximages, teximg_path, denorm_m1_1 = False)
        path = os.path.join(output_dir, os.path.splitext(f_base)[0] + "-recon-" + str(i) + '.png')
        if not recon_set_mode and not corres_set_mode:
            imgutils.imswrite_t(ML_images, path, ncols = 1, white_mask = True, resize = resize)
        print('\tSaving', path)
        if copy_original_input or animated:
            if animated or (not recon_set_mode and not corres_set_mode):
                print('\tCopying', f, 'to', output_dir)
                shutil.copy(f, output_dir)
        storage = []
        ellipse_storage = []
        if corres_set_mode:
            if corresponding_points_targets == 'random_euc_farthest':
                corresponding_points_targets = utils.euclidean_random_far_point_indices(
                                                    P = cy2_M_pe.squeeze(0).cpu().detach().numpy(), 
                                                    nT = num_corres_points )
            vert_vals = graphicsutils.target_vertex_projected_locations(
                            ML_cy2_M, renderer, 
                            corresponding_points_targets,  
                            in_unnorm_pixel_coords = True, 
                            img_sl = H, # or W 
                            as_pil_type = "ellipse") 
            ellipse_storage.append(vert_vals) 
            storage.append(img)

        if animated:
            nr_anim = 100
            aziset  = np.linspace(0.0, 2 * np.pi, nr_anim)
            FPS     = 30
            storage2 = []
            recon_set_path = path.replace('.png', '-recon-%d.gif' % i)
            writer = imageio.get_writer( recon_set_path, mode='I', fps = FPS)
            for j, azi in enumerate(aziset):
                _fixed_rot    = graphicsutils.random_upper_hemi_rotm_manual(1, 
                                                fixed_azi = azi, fixed_elev = fixed_elev)
                curr_cy2_M_pe = torch.bmm(cy2_M_pe, 
                                          _fixed_rot.permute(0,2,1).to(device)) # B=1 x |V| x 3
                curr_cy2_M_pe = curr_cy2_M_pe + ML_translates.view(1,1,3)
                new_render = model.render(curr_cy2_M_pe, ML_textures, renderer)
                new_render = new_render.detach() #.permute(1,2,0).cpu() #.numpy()
                #
                rgb        = new_render[:,0:3,:,:].clamp(min = -1.0, max = 1.0)
                rgb        = (rgb + 1) / 2
                mask       = new_render[:,3,:,:].unsqueeze(1)
                rgb        = rgb * mask
                invmask    = (mask * -1.0) + 1.0
                rgb        = rgb + invmask
                new_render = rgb.clamp(min=0.0, max=1.0)[0].permute(1,2,0).cpu()
                #
                imgrgb     = img[:,0:3,:,:].cpu().clamp(min = -1.0, max = 1.0)
                imgrgb     = (imgrgb + 1) / 2
                imgmask    = imgrgb[:,3,:,:].unsqueeze(1)
                imgrgb     = imgrgb * imgmask
                invmask    = (imgmask * -1.0) + 1.0
                imgrgb     = imgrgb + invmask
                imgrgb     = imgrgb.clamp(min=0.0, max=1.0)[0].permute(1,2,0).cpu()
                #
                img_data   = (255 * new_render).numpy().astype(np.uint8)
                storage2.append( torch.cat( imgrgb,
                                            new_render,
                                    dim = 0, # x-axis
                                 ) )
                writer.append_data(img_data)
            writer.close()
            mastergif.append(storage2)
            continue 

        for j, azi in enumerate(AZIS):
            _fixed_rot    = graphicsutils.random_upper_hemi_rotm_manual(1, 
                                            fixed_azi = azi, fixed_elev = fixed_elev)
            curr_cy2_M_pe = torch.bmm(cy2_M_pe, 
                                      _fixed_rot.permute(0,2,1).to(device)) # B=1 x |V| x 3
            curr_cy2_M_pe = curr_cy2_M_pe + ML_translates.view(1,1,3)

            if mark_correspondences:
                if corresponding_points_targets == 'random_euc_farthest':
                    corresponding_points_targets = utils.euclidean_random_far_point_indices(
                                                        P = cy2_M_pe.squeeze(0).cpu().detach().numpy(), 
                                                        nT = num_corres_points )
                vert_vals = graphicsutils.target_vertex_projected_locations(
                        curr_cy2_M_pe, renderer, 
                        corresponding_points_targets, # 
                        in_unnorm_pixel_coords = True, # 
                        img_sl = H, # 
                        as_pil_type = "ellipse") # 
                ellipse_storage.append(vert_vals) 

            new_render = model.render(curr_cy2_M_pe, ML_textures, renderer)
            storage.append(new_render)
        rerenders = torch.cat(storage, dim = 0) # B=NUM_AZIS x C=4 x H x W
        
        if recon_set_mode:
            assert ( img.shape == (1, 4, 64, 64) and ML_images.shape == img.shape and 
                     rerenders.shape == (NUM_AZIS, 4, 64, 64) )
            all_rens = torch.cat( (img, ML_images, rerenders), dim = 0 )
            print('Total', NUM_AZIS + 2, all_rens.shape)
            recon_set_path = path.replace('.png', '-set-%d.png' % i)
            imgutils.imswrite_t(all_rens, recon_set_path, ncols = NUM_AZIS + 2, 
                                white_mask = True, no_grid = True)
            print('\tSaving', recon_set_path)

        else:       
            rr_path = path.replace('.png', '-reren_set.png') 
            if mark_correspondences:
                if corres_set_mode:
                    _box_first_img = True
                    assert len(ellipse_storage) == NUM_AZIS + 1 and len(rerenders) == NUM_AZIS + 1
                    ellipse_positions = torch.cat(ellipse_storage, dim = 0) # B=NUM_AZIS+1 x n_targs x 4
                    ellipse_colours = [ random_colours for _ in range(NUM_AZIS + 1) ] # B=NUM_AZIS+1 x n_targs x 3
                    if _box_first_img:
                        grid = imgutils.imswrite_t(rerenders, None, 
                                            ncols               = NUM_AZIS + 1, 
                                            white_mask          = True, 
                                            resize              = resize,
                                            dot_positions       = ellipse_positions,
                                            no_grid             = True,
                                            return_grid_no_save = True,
                                            dot_colours         = ellipse_colours)
                        assert grid.shape[0:2] == (3, ES)
                        line_width = int(1)
                        _numpy_img = np.uint8( grid.permute(1,2,0).numpy() * 255 ) # H x W x C
                        _pil_version = Image.fromarray( _numpy_img ) 
                        _draw_obj    = ImageDraw.Draw( _pil_version )  
                        _draw_obj.rectangle([ (0,0), (ES-1,ES-1) ], fill=None, 
                                            outline=(0, 0, 0), width=line_width)# outline = colour
                        _pil_version.save(rr_path)
                    else:
                        imgutils.imswrite_t(rerenders, rr_path, 
                                            ncols         = NUM_AZIS + 1, 
                                            white_mask    = True, 
                                            resize        = resize,
                                            dot_positions = ellipse_positions,
                                            no_grid       = True,
                                            dot_colours   = ellipse_colours)
                else:
                    ellipse_positions = torch.cat(ellipse_storage, dim = 0) # B=NUM_AZIS x n_targs x 4
                    ellipse_colours = [ random_colours for _ in range(NUM_AZIS) ] # B=NUM_AZIS x n_targs x 3
                    imgutils.imswrite_t(rerenders, rr_path, 
                                        ncols         = NUM_AZIS, 
                                        white_mask    = True, 
                                        resize        = resize,
                                        dot_positions = ellipse_positions, 
                                        dot_colours   = ellipse_colours)
            else:
                imgutils.imswrite_t(rerenders, rr_path, ncols = NUM_AZIS, white_mask = True)
            print('\tSaving', rr_path)
        
        if save_combined_best:
            best_com[i] = { 'error' : recon_error, 'ML_images' : ML_images, 
                            'input' : img, 'rerens' : rerenders }
            if mark_correspondences:
                best_com[i]['ell_pos'] = ellipse_storage 
                best_com[i]['ell_col'] = ellipse_colours

    if animated:
        convert = lambda t: (255 * t).numpy().astype(np.uint8) # tensor to np image
        print('Found %d subimgs' % len(mastergif))
        nrows = 2
        ncols = 7
        n = nrows * ncols
        chunked = [ mastergif[i : i + n] for i in range(0, len(mastergif), n)]
        chunked = chunked[0 : -1]
        for qu, chunk in enumerate(chunked): 
            namae = os.path.join(output_dir, 'SET-%d.png' % qu)
            nframes = len(chunk[0][0])
            FPS = 30
            writer = imageio.get_writer(namae, mode = 'I', fps = FPS)
            for frame in range(nframes):
                all_comframes = torch.stack([ listset[frame] for listset in chunk ], dim = 0)
                combined = imgutils.imswrite_t(all_comframes.permute(0,3,1,2), None, ncols = ncols, white_mask = False, 
                                               denorm_m1_1 = False, corner_strings = None, 
                                               dot_positions = None, dot_colours = None, resize = None, 
                                               no_grid = True, return_grid_no_save = True,
                                            ).permute(1,2,0).cpu().numpy()
                writer.append_data((255*combined).astype(np.uint8))
            writer.close()

    if save_combined_best:
        n_targets     = num_combined_to_save
        n_to_check    = len(best_com)
        num_best_cols = NUM_AZIS + 1 + 1 
        errors = np.array([ best_com[i]['error'] for i in range(n_to_check) ])
        inds   = np.argpartition(errors, n_targets)[0 : n_targets] 
        targs  = [ torch.cat( 
                    (best_com[i]['input'], best_com[i]['ML_images'], best_com[i]['rerens']),
                    dim = 0
                   ) for i in inds ]
        targsc = torch.cat(targs, dim = 0) 
        if mark_correspondences:
            ellipse_positions = [ ([None, None] + best_com[i]['ell_pos']) for i in inds ]
            ellipse_positions = list(itertools.chain(*ellipse_positions))
            ellipse_positions = [ None if qq is None else qq.squeeze(0) for qq in ellipse_positions ]
            ellipse_colours = [ ([None, None] + best_com[i]['ell_col']) for i in inds ]
            ellipse_colours = list(itertools.chain(*ellipse_colours))
            imgutils.imswrite_t(targsc, best_path_out, ncols = num_best_cols, 
                                white_mask = True, resize = resize)
            imgutils.imswrite_t(targsc, best_path_out_corres, 
                                ncols         = num_best_cols, 
                                white_mask    = True,
                                resize        = resize,
                                dot_positions = ellipse_positions, 
                                dot_colours   = ellipse_colours)
        else:
            imgutils.imswrite_t(targsc, best_path_out, ncols = num_best_cols, 
                                white_mask = True, resize = resize)

def generate_random_images_from_shapes(model, renderer, shapes_dir, 
                                       output_dir, device, 
                                       num_images_per_shape, 
                                       combine_all_imgs,
                                       resize = None,
                                       copy_original_input = True,
                                       full_cycle_1 = False,
                                       random_subset_size = None):
    shape_data    = load_shapes_data(shapes_dir, random_subset_size = random_subset_size)
    pc_files      = shape_data.pc_files
    normals_files = shape_data.normals_files
    if combine_all_imgs:
        all_imgs = []
        com_set_path_0 = os.path.join(output_dir, 'com_set')
        com_set_path = com_set_path_0 + ".png"
        ii = 0
        while os.path.isfile(com_set_path):
            com_set_path = com_set_path_0 + "-%d.png" % ii
            ii += 1

    for i, ((shape, normal), pc_file, normal_file) in enumerate(zip(shape_data, pc_files, normals_files)):
        print('Performing conditional generation', i)
        shape  = shape.unsqueeze(0).to(device)
        normal = normal.unsqueeze(0).to(device)
        f_base       = os.path.basename(pc_file)
        nonext_fbase = os.path.splitext(f_base)[0] # extensionless base filename
        outpath_base = os.path.join(output_dir, nonext_fbase)
        print('\tShape File:', pc_file, '[%s]' % nonext_fbase)
        subset_ps = []
        for j in range(num_images_per_shape):
            ( renders, texture, V_new, delta, 
              R, t, r, v, xi_p, xi_T, 
              V_new_preeuc, V_intermeds, 
              mu_v, logvar_v, s2i_teximg ) = model.shape_to_image(shape, normal, renderer, xi_p=None, xi_T=None)
            if full_cycle_1:
                ( M_hat, _, xi_T_hat, v_hat,
                  R_hat, t_hat, delta_hat, M_hat_pe,
                  M_hat_ints, pose_probs, r_hat, decoded_texture,
                  mu_xi_T_hat, logvar_xi_T_hat, i2s_teximg ) = model.image_to_shape(renders)
                inds        = pose_probs.detach().argmax(dim = 1)
                decoded_texture = decoded_texture.gather(dim = 1, # B x NH x |V| x 3
                                        index = inds.view(1,1,1,1).expand(-1,-1,model.nV,3)
                                    ).squeeze(1) # B x |V| x 3
                i2s_teximg = i2s_teximg.gather(dim = 1, # B x NH x C x H x W
                                        index = inds.view(1,1,1,1,1).expand(-1,-1,3,64,64)
                                    ).squeeze(1) # B x C x H x W
                
                pathreconply = os.path.join(output_dir, nonext_fbase + "-reconshape-" + str(i) + "-" + str(j) + '.ply')
                print("\tSaving reconstructed ply", pathreconply)       
                meshutils.write_textured_ply(pathreconply, 
                                         M_hat_pe.cpu().detach().squeeze(0).numpy(), 
                                         model.template_F.cpu().detach().numpy(), 
                                         decoded_texture.cpu().detach().squeeze(0).numpy() )
                path_tex_img_s2i = os.path.join(output_dir, nonext_fbase + '-s2i_teximg-' + str(i) + "-" + str(j) + ".png")
                imgutils.imswrite_t(s2i_teximg, path_tex_img_s2i, denorm_m1_1 = False)
                path_tex_img_i2s = os.path.join(output_dir, nonext_fbase + '-i2s_teximg-' + str(i) + "-" + str(j) + ".png")
                imgutils.imswrite_t(i2s_teximg, path_tex_img_i2s, denorm_m1_1 = False)
            pathply = os.path.join(output_dir, nonext_fbase + "-genshape-" + str(i) + "-" + str(j) + '.ply')
            print('\tSaving ply', pathply)
            meshutils.write_textured_ply(pathply, 
                                         V_new_preeuc.cpu().detach().squeeze(0).numpy(), 
                                         model.template_F.cpu().detach().numpy(), 
                                         texture.cpu().detach().squeeze(0).numpy() )
            pathpng = os.path.join(output_dir, nonext_fbase + "-genren-" + str(i) + "-" + str(j) + '.png')
            pathsubcom = pathpng.replace('-genren-', '-genren-subcom-')
            subset_ps.append(renders)
            if not combine_all_imgs: 
                print('\tSaving png', pathpng)
                imgutils.imswrite_t(renders, pathpng, ncols = 1, white_mask = True)
            else:
                all_imgs.append(renders)
        if copy_original_input:
            print('\tCopying input PC file', pc_file , 'to', output_dir)
            shutil.copy(pc_file, output_dir)
            print('\tCopying input N file', normal_file , 'to', output_dir)
            shutil.copy(normal_file, output_dir)
        if combine_all_imgs:
            imgs = torch.cat(all_imgs, dim = 0)
            print("\tSaving combined images to", com_set_path)
            imgutils.imswrite_t(imgs, com_set_path, ncols = num_images_per_shape, # len(all_imgs), 
                                white_mask = True, resize = resize, no_grid = True)

#----------------------------------------------------------------------------------------------#

def load_shapes_data(shapes_dir, num_pc_points = 1000, random_subset_size = None):
    # Shapenet settings
    data_scale = None 
    R_angle = torch.FloatTensor([0.0]).unsqueeze(0)
    R_axis = torch.FloatTensor(np.array([1,0,0])).unsqueeze(0)

    shapedata_type = shapedata.DirectPointsAndNormals
    print('Loading shapes', shapes_dir)
    shape_dataset  = shapedata_type(shapes_dir,     
                                   num_pc_points,
                                   duplicate      = None,
                                   pre_transform  = True,
                                   subset_num     = random_subset_size,
                                   scale          = data_scale,
                                   rot_angle      = R_angle,
                                   rot_axis       = R_axis)
    print('\tObtained', len(shape_dataset), 'shapes')
    return shape_dataset

def load_images_data(images_dir, imsize = 64, random_subset_size = None, subset_initial = False):
    img_data_class = imgutils.SinglePreloadedDirImageDataset
    print('Loading images from', images_dir)
    img_dataset    = img_data_class(images_dir, 
                                    use_alpha          = True, 
                                    resize             = imsize, 
                                    load_gt_file       = False,  
                                    num_fixed_to_store = 1, # 
                                    subset_initial     = subset_initial,
                                    take_subset        = random_subset_size ) 
    print('\tObtained', len(img_dataset), 'images')
    return img_dataset


#-------------------------#
if __name__ == "__main__":
    main()
#-------------------------#









#
