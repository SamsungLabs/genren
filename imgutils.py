import torch, imageio, numpy as np, torchvision.utils as TU, os, sys, utils, json, logging
from skimage import io
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import InfiniteDataLoader
import random, torch.nn.functional as F

### IO ###

def imswrite_t(imgs, path, ncols = 8, white_mask = True, denorm_m1_1 = True, corner_strings = None, 
               dot_positions = None, dot_colours = None, resize = None, no_grid = False, 
               return_grid_no_save = False,
               ):
    """ 
    Writes an image batch to path (imgs: B x C x H x W).
    Assumes input in [-1,1].
    Pass None for path if return_grid_no_save is true
    """
    #preimg = (imgs.detach() * 255).astype(np.uint8)
    B, C, H, W = imgs.shape
    preimg = imgs.cpu().detach()[:,0:3,:,:].clamp(min = -1.0, max = 1.0) # <----
    if denorm_m1_1: # [-1, 1] --> [0, 1]
        preimg = (preimg + 1.0) / 2.0

    # Perform conversion of mask to white background
    if white_mask and C == 4:
        assert imgs.shape[1] > 3, "White mask requested but no alpha channel present"
        mask = imgs.cpu().detach()[:,3,:,:].unsqueeze(1) # 1 = shown, 0 = masked
        preimg = preimg * mask # Ensure cleanliness
        invmask = (mask * -1.0) + 1.0
        preimg = preimg + invmask
        preimg = torch.clamp(preimg, min=0.0, max=1.0)# clamp to ensure viability

    # Add positional dots
    # Uses: List of list of dots + colours, for each image
    if (not dot_positions is None) or (not dot_colours is None):
        assert len(dot_positions) == len(dot_colours)
        for d, c in zip(dot_positions, dot_colours): 
            if d is None:
                assert c is None
            if c is None:
                assert d is None
            if (not c is None) and (not d is None):
                assert len(d) == len(c)
        numpy_imgs = np.uint8( preimg.permute(0,2,3,1).numpy() * 255 ) # B x H x W x C
        all_images = []
        for ii, numpy_img in enumerate(numpy_imgs):
            pil_version = Image.fromarray( numpy_img ) 
            draw_obj    = ImageDraw.Draw(pil_version)

            cdot_colours = dot_colours[ii]
            if not cdot_colours is None:
                for dot, colour in zip(dot_positions[ii], cdot_colours):
                    dot = tuple(dot.cpu().detach().numpy().tolist())
                    draw_obj.ellipse(dot, fill = tuple(colour))
                    #draw_obj.point(dot, fill = tuple(colour))
            
            numpy_with_text = np.array(pil_version)
            torch_with_text = torch.from_numpy(numpy_with_text).permute(2,0,1) # C x H x W
            all_images.append(torch_with_text)
        preimg = torch.clamp( torch.stack(all_images, dim = 0).float() / 255, min=1e-6, max=0.9999)

    # Add text to the corners if needed
    if not corner_strings is None:
        numpy_imgs = np.uint8( preimg.permute(0,2,3,1).numpy() * 255 ) # B x H x W x C
        all_images = []
        for ii, numpy_img in enumerate(numpy_imgs):
            pil_version = Image.fromarray( numpy_img ) 
            #fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
            d = ImageDraw.Draw(pil_version)
            sii = corner_strings[ii]
            d.text((1,1), sii, fill=(0,0,0,128)) # modifies pil_version in-place
            numpy_with_text = np.array(pil_version)
            torch_with_text = torch.from_numpy(numpy_with_text).permute(2,0,1) # C x H x W
            all_images.append(torch_with_text)
        preimg = torch.clamp( torch.stack(all_images, dim = 0).float() / 255, min=1e-6, max=0.9999)

    # Resize if needed
    if not resize is None:
        if type(resize) is int: resize = (resize, resize)
        preimg = F.interpolate(preimg, size = resize, mode = 'bicubic', align_corners = False)
        preimg = preimg.clamp(0.0, 1.0)

    # Finally save the images
    if no_grid:
        if return_grid_no_save:
            return TU.make_grid(preimg, nrow=ncols, padding=0)
        TU.save_image(preimg, path, nrow=ncols, padding=0) #
    else:
        if return_grid_no_save:
            return TU.make_grid(preimg, nrow=ncols)
        TU.save_image(preimg, path, nrow=ncols) # -_-'

def imwrite_t(img, path, permute=False):
    """ Writes img to path: assumes img is a torch tensor (H x W x 3) """
    if permute: img = img.permute(1,2,0)
    img = (img.detach().numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, img)

### IMAGE VIEWING ###

from PIL import Image
import matplotlib.pyplot as plt

def mpl_show(img):
    plt.imshow( img.permute(1,2,0) )
    plt.show()

### DATA LOADING ###

# NOT preloaded into memory
class SingleDirImageDataset(torch.utils.data.Dataset):
    # TODO use_alpha options 
    # TODO This one is currently overruled
    def __init__(self, folder, use_alpha=True, load_gt_file=True, verbose=True, resize=64,
                num_fixed_to_store=None):

        assert False
        
        self.folder = folder
        ALLOWED_EXTS = [ ".png" ]
        self.files = [ os.path.join(folder, f) for f in os.listdir(folder)
                       if any(f.endswith(a) for a in ALLOWED_EXTS) ]
        normer = transforms.Normalize([0.5]*3 + [0.0], [0.5]*3 + [1.0])
        norm3 = transforms.Normalize([0.5]*3, [0.5]*3)
        if resize is None or resize == False:
            rtrans = []
        else:
            rtrans = [ transforms.Resize(resize) ]

        self.T = transforms.Compose(
                   rtrans +
                   [ transforms.ToTensor(),
                     (normer if use_alpha else norm3)
                 ])

#        self.data = [ ( self.T( Image.open(file) )[0:3,:,:] 
#                          if not use_alpha else
#                          self.T( Image.open(file) ) )
#                        for file in self.files ]
        self.imsize = self[0].shape[-1]
        if verbose:
            logging.info('Created SingleDirImg dataset with %d images from directory %s (%s)', len(self.files), 
                    self.folder, '(Image size: %d)' % self.imsize)
        # Load ground truth rotations
        self.load_gt_file = load_gt_file
        if load_gt_file:
            json_files = [ os.path.join(folder, f) for f in os.listdir(folder)
                           if f.endswith('.json') ]
            assert len(json_files) == 1, "Multiple json files detected: " + str(json_files)
            jf = json_files[0]
            
            # Want a rotations dict + a translations dict
            with open(jf, "r") as json_file:
                data = json.load(json_file)
            
            rot_dict = { key : torch.FloatTensor(value['rotation'])
                         for key, value in data.items() }
            t_dict   = { key : torch.FloatTensor(value['translation'])
                         for key, value in data.items() }
            
            #gt_rot_dict = utils.load_s2torch_json(jf)
            self.filename_stubs = [ f for f in os.listdir(folder) 
                                    if (any(f.endswith(a) for a in ALLOWED_EXTS)) ]
            # TODO change
            self.data = [ (img, rot_dict[fname], t_dict[fname]) 
                          for img, fname in zip(self.data, self.filename_stubs) ]
            if verbose:
                print('\tLoaded GT rotations from', jf)

        # Store some indices to targets that we want to repeatedly compute the output for 
        if type(num_fixed_to_store) is int:
          self.fixed_indices = list(range(num_fixed_to_store))
          # Pre-assemble the fixed batch
          with torch.no_grad():
            self.fixed_batch = torch.stack([self[k] for k in self.fixed_indices], dim = 0)
          if verbose:
            logging.info('\tLoaded fixed batch (size %d)', num_fixed_to_store)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.T( Image.open( self.files[i] ) ) # [0:3, :, :] # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #return self.data[i]

    def get_dataloader(self, B, NW=4, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=B, shuffle=shuffle, 
                                           drop_last=True, pin_memory=True,
                                           num_workers=NW)

    def get_infinite_dataloader(self, B, NW=4, shuffle=True):
        return InfiniteDataLoader(self, batch_size=B, shuffle=shuffle,
                                  drop_last=True, pin_memory=True,
                                  num_workers=NW)
    def get_fixed_batch(self):
        return self.fixed_batch

def rgba_img_to_wbg(img):
  """ Assumes input is [-1,1] """
  preimg = img.cpu().detach()[0:3,:,:].clamp(min=-1.0, max=1.0) 
  mask = img[3,:,:] # 1 = shown, 0 = masked
  preimg = preimg * mask # Ensure cleanliness
  invmask = (mask * -1.0) + 1.0 # 0 = shown, 1 = masked
  preimg = preimg + invmask * 2 # preimg<-1,1> -> shown parts unaffected, masked parts gain +2
  preimg = torch.clamp(preimg, min = -1.0, max = 1.0)
  return preimg

def rgba_img_to_wbg_with_alpha(img):
  """ Assumes input is [-1,1] """
  preimg = img.cpu().detach()[0:3,:,:].clamp(min=-1.0, max=1.0) 
  mask = img[3,:,:] # 1 = shown, 0 = masked
  preimg = preimg * mask # Ensure cleanliness
  invmask = (mask * -1.0) + 1.0 # 0 = shown, 1 = masked
  preimg = preimg + invmask * 2 # preimg<-1,1> -> shown parts unaffected, masked parts gain +2
  preimg = torch.clamp(preimg, min = -1.0, max = 1.0)
  return torch.cat( (preimg, mask), dim = 0)

#########

class SinglePreloadedDirImageDataset(torch.utils.data.Dataset):
    
    def __init__(self, folder, use_alpha=False, load_gt_file=True, verbose=True, resize=64, 
                 num_fixed_to_store=None, take_subset=None, subset_initial=False):
        logging.info('Constructing SinglePreloadedDirImageDataset')
        logging.info('\tUsing Alpha = ' + str(use_alpha))
        self.folder = folder
        ALLOWED_EXTS = [ ".png" ]
        self.files = [ os.path.join(folder, f) for f in os.listdir(folder)
                       if any(f.endswith(a) for a in ALLOWED_EXTS) ]
        print('Obtained', len(self.files), 'files')
        if not take_subset is None:
            assert type(take_subset) is int
            print('Taking subset of size', take_subset, '(Type: %s)' % subset_initial)
            if subset_initial:
                self.files = self.files[0:take_subset]
                print('\tTook initial subset', len(self.files))
            else:
                self.files = random.sample(self.files, take_subset)
                print('\tTook random subset', len(self.files))
    
        normer = transforms.Normalize([0.5]*3 + [0.0], [0.5]*3 + [1.0])
        norm3 = transforms.Normalize([0.5]*3, [0.5]*3)
        if resize is None or resize == False:
            rtrans = []
        else:
            rtrans = [ transforms.Resize(resize) ]

        eg_1 = transforms.ToTensor()( Image.open(self.files[0]) )
        eg_ncs = eg_1.shape[0]
        if not use_alpha and eg_ncs == 4:
          norm3 = normer

        self.T = transforms.Compose(
                   rtrans +
                   [ transforms.ToTensor(),
                     (normer if use_alpha else norm3)
                 ])

        self.data = [ ( self.T( Image.open(file) ) #[0:3,:,:] 
                          if not use_alpha else
                          self.T( Image.open(file) ) )
                        for file in self.files ]
        if (not use_alpha) and eg_ncs == 4:
            logging.info('Removing transparent bg in images')
            self.data = [ rgba_img_to_wbg(datum)
                          for datum in self.data ]
        #
        #self.random_colour_bg = True
        #
        self.imsize = self.data[0].shape[-1]
        if verbose:
            logging.info('\tCreated SinglePreloadedDir dataset with %d images from %s (imsize: %d)' 
                    % (len(self.data), self.folder, self.imsize) )

        # Load ground truth rotations
        self.load_gt_file = load_gt_file
        if load_gt_file:
            json_files = [ os.path.join(folder, f) for f in os.listdir(folder)
                           if f.endswith('.json') ]
            assert len(json_files) == 1, "Multiple json files detected: " + str(json_files)
            jf = json_files[0]
            
            # Want a rotations dict + a translations dict
            with open(jf, "r") as json_file:
                data = json.load(json_file)
            
            rot_dict = { key : torch.FloatTensor(value['rotation'])
                         for key, value in data.items() }
            t_dict   = { key : torch.FloatTensor(value['translation'])
                         for key, value in data.items() }
            
            #gt_rot_dict = utils.load_s2torch_json(jf)
            self.filename_stubs = [ f for f in os.listdir(folder) # TODO subset allowance
                                    if (any(f.endswith(a) for a in ALLOWED_EXTS)) ]
            self.data = [ (img, rot_dict[fname], t_dict[fname]) 
                          for img, fname in zip(self.data, self.filename_stubs) ]
            if verbose:
                logging.info('\tLoaded GT rotations from', jf)

        # Store some indices to targets that we want to repeatedly compute the output for 
        if type(num_fixed_to_store) is int:
          self.fixed_indices = list(range(num_fixed_to_store))
          # Pre-assemble the fixed batch
          with torch.no_grad():
            self.fixed_batch = torch.stack([self.data[k] for k in self.fixed_indices], dim = 0)
          if verbose:
            logging.info('\tLoaded fixed batch (size %d)', num_fixed_to_store)

    def unmasked_pixel_set(self, num_random_images = 10000, num_random_pixels = 5, ZO_norm=True):
        logging.info('Extracting unmasked pixel set for FDR')        
        NI = num_random_images # Number of images to use
        NP = num_random_pixels # Number of pixels per image to extract
        num_actual_imgs = len(self.data)
        if NI > num_actual_imgs:
            logging.info('\tRenumerating to' + str(num_actual_imgs))
            NI = num_actual_imgs
        timgs = random.sample(self.data, k = NI)
        pixel_array = []
        with torch.no_grad():
            for i in range(NI): # Memory safe to vectorize sometimes
                cimg = timgs[i]
                mask = cimg[-1, :, :] > 0.5 # assumes C x H x W, alpha pixel in [0,1], rest in [-1,1]
                rgb  = cimg[0:3, :, :]
                #unmasked_rgb = rgb[ mask.unsqueeze(0).expand(3,-1,-1) ] # Shape: N_valid x 3
                unmasked_rgb = rgb[:, mask] # Shape: 3 x N_valid 
                n_valid = list(unmasked_rgb.shape)[1]
                # Random choice of pixels
                rinds = np.random.choice(n_valid, size = (NP,), replace = False)
                cpixels = unmasked_rgb[:, rinds] # choose the random unmasked pixels we want
                pixel_array.append(cpixels) # 3 x NP
            # Concat all the pixel sets together
            pixel_array = torch.cat(pixel_array, dim = 1).transpose(0,1)
            # WARNING: we now zero-one normalize these pixels! <----------------------------
            # This is because the FDR texturer assumes [0,1] random colours!
            if ZO_norm:
                logging.info('\tZero-One normalizing')
                pixel_array = (pixel_array + 1.0) / 2.0
        #print(pixel_array)
        #sys.exit(0)
        # Return an array of pixel values
        logging.info('\tFinished ' + str(pixel_array.shape))
        return pixel_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # if self.random_colour_bg:
        #     I = self.data[i]
        #     RC = torch.rand()
        return self.data[i]

    def get_dataloader(self, B, NW=4, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=B, shuffle=shuffle, 
                                           drop_last=True, pin_memory=True,
                                           num_workers=NW)

    def get_infinite_dataloader(self, B, NW=4, shuffle=True):
        return InfiniteDataLoader(self, batch_size=B, shuffle=shuffle,
                                  drop_last=True, pin_memory=True,
                                  num_workers=NW)
    def get_fixed_batch(self):
        return self.fixed_batch

#
