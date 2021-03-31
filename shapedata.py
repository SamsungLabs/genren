import trimesh, torch, os, sys, pcutils, meshutils, dgl, numpy as np, logging, random
from torch.utils.data import Dataset, DataLoader
from utils import InfiniteDataLoader

class DatasetCore(Dataset):

    def __len__(self):
        return len(self.files)

    def get_dataloader(self, B, NW=4, shuffle=True, drop_last=True):
        c = self.__class__.collate if self.custom_collate else None
        return torch.utils.data.DataLoader(self, batch_size=B, shuffle=shuffle,
                                           drop_last=drop_last, pin_memory=True,
                                           num_workers=NW, collate_fn=c)

    def get_infinite_dataloader(self, B, NW=4, shuffle=True):
        c = self.__class__.collate if self.custom_collate else None
        return InfiniteDataLoader(self, batch_size=B, shuffle=shuffle,
                                  drop_last=True, pin_memory=True,
                                  num_workers=NW, collate_fn=c)

class ObjToMeshAndPCsDataset(DatasetCore):

    def __init__(self, 
                 folder, 
                 num_pc_points,
                 duplicate, 
                 generate_dgl_objs=False,
                 preload_all=True,
                 transpose_pc=False,
                 pre_transform=True,
                 subset_num=None, 
                 scale=None, 
                 rot_angle=None,
                 rot_axis=None):

        assert False

        self.folder = folder
        self.num_pc_points = num_pc_points
        self.files = [ os.path.join(folder, f) for f in os.listdir(folder)
                       if f.endswith('.obj') ]
        if not subset_num is None:
            self.files = self.files[0:subset_num]
        if not duplicate is None and duplicate > 0:
            self.files = self.files * duplicate
        self.preload = preload_all
        self.transpose_pc = transpose_pc

        self.pre_transform = pre_transform
        if pre_transform:
            self.scale = scale
            self.rot_angle = rot_angle
            self.rot_axis = rot_axis

        self.custom_collate = True

        logging.info('Initialized OMPD shape data with %d files (duping? %d)' % (len(self.files), duplicate))

        if self.preload:
            logging.info("\tPreloading")
            # Preloaded torch meshes (V,F)
            self.preloaded_objs = [ meshutils.read_surface_mesh(f, to_torch=True)
                                    for f in self.files ]
            # Faces
            self.faces = [ m[1] for m in self.preloaded_objs ]
            # Vertices (transformed)
            if self.pre_transform:
                self.verts = [ meshutils.rotate(
                                            self.rot_angle,
                                            self.rot_axis,
                                            meshutils.norm_mesh(
                                                self.preloaded_objs[i][0], 
                                                scale = self.scale
                                            )
                                        )
                                        for i in range(len(self.files)) ]
            else:
                self.verts = [ m[0] for m in self.preloaded_objs ]
            # Trimesh objects from the loaded meshes
            self.pl_trimesh_list = [ trimesh.Trimesh(vertices=v, faces=f)
                                     for v,f in zip(self.verts, self.faces) ]
            # Deep graph library objects
            self.generate_dgl_objs = generate_dgl_objs
            if generate_dgl_objs:
                logging.info('\tGenerating DGL objects')
                self.dgl_graphs = [ self._make_dgl_graph(v,f,t) 
                                    for v,f,t in zip(self.verts, self.faces, self.pl_trimesh_list) ]
            else:
                logging.info('\tNot using DGL objects')
                self.dgl_graphs = [ None ] * len(self.faces)
                    

    def _make_dgl_graph(self, V, F, trimesh_obj):
        # Construct DGL graph instance
        g = dgl.DGLGraph()
        # Add nodes into the graph
        g.add_nodes(V.shape[0])
        # Get edges
        edges = trimesh_obj.edges # |E| x 2
        src, dst = tuple(zip(*edges))
        # Insert edges into graph
        g.add_edges(src, dst)
        # Reverse edges (DGL edges are directional)
        g.add_edges(dst, src)
        # Self-edges (i.e. self-loops per node)
        g.add_edges(g.nodes(), g.nodes())
        # Store vertex coordinates as nodal features
        g.ndata['features'] = V.clone().detach()
        return g

    @staticmethod
    def collate(samples):
        # The input `samples` is a list of pairs (graph, pc_tensor).
        graphs, pcs = map(list, zip(*samples))
        # Construct a batch of graphs within a single sparse tensor
        #if self.generate_dgl_objs:
        if graphs[0] is None:
            batched_graph = None
        else:
            batched_graph = dgl.batch(graphs)
        #else:
        #    batched_graph = None
        return batched_graph, torch.stack(pcs)

    def __getitem__(self, i):
        if self.preload:
            pc = torch.from_numpy( self.pl_trimesh_list[i].sample(self.num_pc_points) ).float()
            g = self.dgl_graphs[i]
            return g, pc
        else:
            raise ValueError("Unimplemented")


#-------------------------------------------------------------------------------------------------#


class DirectPointsAndNormals(DatasetCore):

    def __init__(self, 
                 folder, 
                 num_pc_points,
                 duplicate, 
                 preload_all=True,
                 transpose_pc=False,
                 pre_transform=True,
                 subset_num=None, 
                 scale=None, 
                 rot_angle=None,
                 rot_axis=None):

        self.folder = folder
        self.num_pc_points = num_pc_points
        self.pc_files = [ os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.PC.pt') ]
        if not subset_num is None:
            if subset_num < len(self.pc_files):
                self.pc_files = random.sample(self.pc_files, subset_num)
            else:
                print("WARNING: asked for a larger subset than can be taken (%d/%d)" % 
                        ( (subset_num), len(self.pc_files) ) )
        self.files = self.pc_files # Used in len
        self.normals_files = [ f.replace('.PC.pt', '.normals.pt') for f in self.pc_files ]
        self.preload = preload_all
        self.transpose_pc = transpose_pc
        self.pre_transform = pre_transform
        if pre_transform:
            self.scale = scale
            self.rot_angle = rot_angle
            self.rot_axis = rot_axis
        self.custom_collate = True

        logging.info('Initialized direct P+N shape data from %s with %d files (duping? %d)' % 
                (folder, len(self.pc_files), 1 if duplicate is None else duplicate))

        if self.preload:
            logging.info("\tPreloading/transforming")
            self.points  = [ torch.load(f).float() for f in self.pc_files ]
            self.normals = [ torch.load(f).float() for f in self.normals_files ]
            if self.pre_transform:
                self.points = [ meshutils.rotate( self.rot_angle, self.rot_axis,
                                                  meshutils.norm_mesh(
                                                        self.points[i], 
                                                        scale = self.scale
                                                  ) )
                                for i in range(len(self.pc_files)) ]
                # Note that rotations apply to normals same as points/vertices.
                # If it were another linear transform, we would use the inverse transpose.
                self.normals = [ meshutils.rotate( self.rot_angle, self.rot_axis,
                                                   self.normals[i] )
                                                    #meshutils.norm_mesh(
                                                   #     self.normals[i], 
                                                   #     scale = self.scale
                                                   #) )
                                for i in range(len(self.pc_files)) ]
            # Duplicate if needed
            if not duplicate is None and duplicate > 0:
                logging.info("\tStarting duplications")
                self.normals       = self.normals  * duplicate
                self.points        = self.points   * duplicate
                self.pc_files      = self.pc_files * duplicate
                self.normals_files = self.normals_files * duplicate
                self.files         = self.files * duplicate
                logging.info('\tFinished duplication')

        #if not duplicate is None and duplicate > 0:
        #    self.pc_files = self.pc_files * duplicate

        self.max_nps = self.points[0].shape[0]
        logging.info('\tMaximum number of points: %d (taking samples of size %d)', self.max_nps, self.num_pc_points)

        # Precompute the index vectors into the PCs/normals for faster sampling
        self.num_random_draws = 8000 # Number of index vectors to obtain (one per PC)
        def random_choice_noreplace2(l, n_sample, num_draw):
            '''
            l: 1-D array or list
            n_sample: sample size for each draw
            num_draw: number of draws

            Intuition: Randomly generate numbers, get the index of the smallest n_sample number for each row.
            '''
            l = np.array(l)
            return l[np.argpartition(np.random.rand(num_draw,len(l)), n_sample-1,axis=-1)[:,:n_sample]]
        self.random_pc_index_sets = torch.as_tensor(
                                        random_choice_noreplace2(l        = range(self.max_nps), 
                                                                 n_sample = self.num_pc_points,
                                                                 num_draw = self.num_random_draws) 
                                    )
        logging.info('\tPre-drew PC/N indices (shape: %s)' % str(self.random_pc_index_sets.shape))

    @staticmethod
    def collate(samples):
        # The input `samples` is a list of pairs (P, N) = (points, normals).
        ps, ns = map(list, zip(*samples))
        return torch.stack(ps), torch.stack(ns) # points, normals

    def __getitem__(self, i):
        """ Returns (points, normals) for the ith mesh """
        #inds = np.random.choice(self.max_nps, size = self.num_pc_points, replace = False)
        rand_index_set = torch.randint(self.num_random_draws, size=(1,)).squeeze(0)
        inds           = self.random_pc_index_sets[rand_index_set]
        return self.points[i][inds,:], self.normals[i][inds,:]
 

class ObjToPcsNormalsDataset(DatasetCore):

    def __init__(self, 
                 folder, 
                 num_pc_points,
                 duplicate, 
                 preload_all=True,
                 transpose_pc=False,
                 pre_transform=True,
                 subset_num=None, 
                 scale=None, 
                 rot_angle=None,
                 rot_axis=None):

        assert False

        self.folder = folder
        self.num_pc_points = num_pc_points
        self.files = [ os.path.join(folder, f) for f in os.listdir(folder)
                       if f.endswith('.obj') ]
        if not subset_num is None:
            self.files = random.sample(self.files, subset_num)
            #self.files = self.files[0:subset_num]

        #if not duplicate is None and duplicate > 0:
        #    self.files = self.files * duplicate
        
        self.preload = preload_all
        self.transpose_pc = transpose_pc

        self.pre_transform = pre_transform
        if pre_transform:
            self.scale = scale
            self.rot_angle = rot_angle
            self.rot_axis = rot_axis

        self.custom_collate = True

        logging.info('Initialized OTPN shape data from %s with %d files (duping? %d)' % 
                (folder, len(self.files), duplicate))

        if self.preload:
            logging.info("\tPreloading")
            # Preloaded torch meshes (V,F)
            self.preloaded_objs = [ meshutils.read_surface_mesh(f, to_torch=True)
                                    for f in self.files ]
            # Faces
            self.faces = [ m[1] for m in self.preloaded_objs ]
            # Vertices (transformed)
            if self.pre_transform:
                self.verts = [ meshutils.rotate(
                                            self.rot_angle,
                                            self.rot_axis,
                                            meshutils.norm_mesh(
                                                self.preloaded_objs[i][0], 
                                                scale = self.scale
                                            )
                                        )
                                        for i in range(len(self.files)) ]
            else:
                self.verts = [ m[0] for m in self.preloaded_objs ]

            # Trimesh objects from the loaded meshes
            self.pl_trimesh_list = [ trimesh.Trimesh(vertices=v, faces=f)
                                     for v,f in zip(self.verts, self.faces) ]

            # Precompute the face normals
            logging.info("\tPreloading face normals")
            def read_or_gen_normals(name, mesh):
                fn_name = name.replace('.obj', '.face_normals.csv')
                if os.path.exists(fn_name): # Read file
                    _a = np.genfromtxt(fn_name, delimiter=",")
                else: # Generate, then write file
                    _a = mesh.face_normals
                    np.savetxt(fn_name, _a, delimiter=",", fmt='%.4f')
                return torch.from_numpy(_a).float()
            # Read or generate the face normals
            self.face_normals = [ read_or_gen_normals(namae, mesh) 
                                  for namae, mesh in zip(self.files, self.pl_trimesh_list) ]
            # Duplicate if needed
            if not duplicate is None and duplicate > 0:
                #self.files = self.files * duplicate
                self.face_normals    = self.face_normals    * duplicate
                self.pl_trimesh_list = self.pl_trimesh_list * duplicate
                self.verts           = self.verts           * duplicate
                self.faces           = self.faces           * duplicate
                self.preloaded_objs  = self.preloaded_objs  * duplicate
                logging.info('\tFinished duplication')

        if not duplicate is None and duplicate > 0:
            self.files = self.files * duplicate

            # # Deep graph library objects
            # self.generate_dgl_objs = generate_dgl_objs
            # if generate_dgl_objs:
            #     print('\tGenerating DGL objects')
            #     self.dgl_graphs = [ self._make_dgl_graph(v,f,t) 
            #                         for v,f,t in zip(self.verts, self.faces, self.pl_trimesh_list) ]
            # else:
            #     print('\tNot using DGL objects')
            #     self.dgl_graphs = [ None ] * len(self.faces)

    @staticmethod
    def collate(samples):
        # The input `samples` is a list of pairs (P, N) = (points, normals).
        ps, ns = map(list, zip(*samples))
        # Return points and normals
        return torch.stack(ps), torch.stack(ns)
        # Construct a batch of graphs within a single sparse tensor
        #if self.generate_dgl_objs:
        # if graphs[0] is None:
        #     batched_graph = None
        # else:
        #     batched_graph = dgl.batch(graphs)
        # #else:
        # #    batched_graph = None
        # return batched_graph, torch.stack(pcs)

    def __getitem__(self, i):
        """
        Returns (points, normals) for the ith mesh
        """
        points, f_indices = self.pl_trimesh_list[i].sample(self.num_pc_points, return_index=True)
        points = torch.from_numpy( points ).float()
        normals = (self.face_normals[i])[f_indices, :] # N_S x 3
        return points, normals
        # if self.preload:
        #     pc = torch.from_numpy( self.pl_trimesh_list[i].sample(self.num_pc_points) ).float()
        #     g = self.dgl_graphs[i]
        #     return g, pc
        # else:
        #     raise ValueError("Unimplemented")


#-------------------------------------------------------------------------------------------------#

class ObjToPcsDataset(DatasetCore):

    def __init__(self, folder, num_pc_points, 
                 preload_all=True, 
                 transpose_pc=False, 
                 duplicate=50, 
                 pre_transform=True,
                 scale=None, rot_angle=None, rot_axis=None):

        assert False

        self.folder = folder
        self.num_pc_points = num_pc_points
        self.files = [ os.path.join(folder, f) for f in os.listdir(folder) 
                       if f.endswith('.obj') ]
        if not duplicate is None and duplicate > 0:
            self.files = self.files * 50
        self.preload = preload_all
        self.transpose_pc = transpose_pc

        self.custom_collate = False

        self.pre_transform = pre_transform
        if pre_transform:
            self.scale = scale
            self.rot_angle = rot_angle
            self.rot_axis = rot_axis

        logging.info('Initialized O2P shape data with %d files (duping? %d)' % (len(self.files), duplicate))

        if self.preload:
            logging.info("\tPreloading")
            self.preloaded_objs = [ meshutils.read_surface_mesh(f, to_torch=True)
                                    for f in self.files ]
            self.faces = [ m[1] for m in self.preloaded_objs ]
            if self.pre_transform:
                self.verts = [ meshutils.rotate(
                                            self.rot_angle,
                                            self.rot_axis,
                                            meshutils.norm_mesh(
                                                self.preloaded_objs[i][0], 
                                                #pcutils.read_obj_into_pc(self.files[i], 
                                                #                         self.num_pc_points
                                                #                        ).float(), 
                                                scale = self.scale
                                            )
                                        )
                                        for i in range(len(self.files)) ]
            else:
                self.verts = [ m[0] for m in self.preloaded_objs ]
            self.pl_trimesh_list = [ trimesh.Trimesh(vertices=v, faces=f)
                                     for v,f in zip(self.verts, self.faces) ]
            
    def __getitem__(self, i):
        # Better to use trimesh or DGL
        if self.preload:
            return torch.from_numpy( self.pl_trimesh_list[i].sample(self.num_pc_points) ).float()
        else:
            if self.transpose_pc:
                return pcutils.read_obj_into_pc(self.files[i], self.num_pc_points).transpose(0,1).float()
            else:
                pc = pcutils.read_obj_into_pc(self.files[i], self.num_pc_points).float()
                return meshutils.rotate(
                        self.rot_angle,
                        self.rot_axis,
                        meshutils.norm_mesh(
                            pc, scale = self.scale
                        )
                     )

#__len__ so that len(dataset) returns the size of the dataset.
#__getitem__ to support the indexing such that dataset[i] can be used to get ith sampl


# From Kaolin
SHAPENET_SYNSET2LABEL = {  '04379243': 'table', '03211117': 'monitor', '04401088': 'phone',
                           '04530566': 'watercraft', '03001627': 'chair', '03636649': 'lamp',
                           '03691459': 'speaker', '02828884': 'bench', '02691156': 'plane',
                           '02808440': 'bathtub', '02871439': 'bookcase', '02773838': 'bag',
                           '02801938': 'basket', '02880940': 'bowl', '02924116': 'bus',
                           '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
                           '03207941': 'dishwasher', '03337140': 'file', '03624134': 'knife',
                           '03642806': 'laptop', '03710193': 'mailbox', '03761084': 'microwave',
                           '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
                           '04004475': 'printer', '04099429': 'rocket', '04256520': 'sofa',
                           '04554684': 'washer', '04090263': 'rifle', '02946921': 'can'}

def category_name_to_synset_label(name):
    for key in SHAPENET_SYNSET2LABEL:
        label = SHAPENET_SYNSET2LABEL[key]
        if label == name:
            return key
    return None






#
