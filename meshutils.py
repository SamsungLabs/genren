import meshio, kornia, torch, torch.nn as nn, numpy as np, sys, os
import trimesh, logging

def read_bff_uvs(mUV_path): # [-1,1]
    """
    Note: BFF does not produce texture coordinates *per vertex*. It does so per "unsplit" triangle corner.
    Faces are defined with (a) 3d vertex coordinates and (b) uv texture coordinates.
    These may differ and hence there may be MORE UV texture coordinate "vertices" than geometric 3d ones.
    This is nice for per-face textures, but since we are using nodal textures, this is an issue.
    As a hack, we just assign a texture vertex to *one* of its geometric counterparts in the faces.
    """
    with open(mUV_path, 'r') as uvph:
        lines = uvph.readlines()
    vtlines = [ line.strip().split(' ')  for line in lines if line.strip().startswith('vt') ]
    num_geom_verts = len([line for line in lines if line.strip().startswith('v ')])
    Uord = torch.tensor([ float(line[1]) for line in vtlines ]) # nV
    Vord = torch.tensor([ float(line[2]) for line in vtlines ]) # nV
    # We now have the texture coordinates. There may be more of these than geometric nodes.
    # e.g., f 1/1 2/2 3/510
    flines = [ line.strip().split(' ')[1:] for line in lines if line.strip().startswith('f') ]
    # Fill in the geom vert -> texture vert map
    map_v2vt = {}
    for fline in flines:
        assert len(fline) == 3
        for ni in fline: # fline = [a/b, c/d, e/f]
            assert len(ni.split("/")) == 2
            # Note: this mapping is from zero-counted geom vert ind to zero-counted tex vert ind
            map_v2vt[ int(ni.split("/")[0]) - 1 ] = int(ni.split("/")[1]) - 1
    # Now construct a map from geom vert ind -> texture vert ind -> texture coordinate
    # Thus we get our actual goal (a geom vert -> tex coord mapping)
    Us = torch.tensor( [ Uord[ map_v2vt[geom_vert_i] ] for geom_vert_i in range(num_geom_verts) ] )
    Vs = torch.tensor( [ Vord[ map_v2vt[geom_vert_i] ] for geom_vert_i in range(num_geom_verts) ] )
    # Transform [0,1] into [-1,1]
    U = (Us * 2.0) - 1.0
    V = (Vs * 2.0) - 1.0
    return U, V

def write_textured_ply(path, V, F, T):
    assert len(V.shape) == 2, V.shape
    assert len(F.shape) == 2, F.shape
    assert len(T.shape) == 2, T.shape
    M = trimesh.Trimesh(V, F, process = False)
    M.visual.vertex_colors = T
    trimesh.Trimesh.export(M, path, 'ply')

def write_obj(path, surface_v, surface_f):
    if type(surface_v) == torch.Tensor: surface_v = surface_v.numpy()
    if type(surface_f) == torch.Tensor: surface_f = surface_f.numpy()
    surface_f = { "triangle" : surface_f }
    mesh = meshio.Mesh(surface_v, surface_f)
    if not path.endswith(".obj"): path += ".obj"
    meshio.write(path, mesh)

def write_xyz(path, points, sep = ',', colours = None):
    if type(points) == torch.Tensor: points = points.numpy()
    assert points.shape[-1] == 3 and len(points.shape) == 2
    if colours is None:
        str_points = "\n".join([ str(p[0]) + sep + str(p[1]) + sep + str(p[2]) # + "\n"
                     for p in points ])
    else:
        str_points = "\n".join([ str(p[0]) + sep + str(p[1]) + sep + str(p[2]) + 
                                 str(colours[0]) + sep + str(colours[1]) + str(colours[2])
                                 for p in points ])
    with open(path, 'w') as ph:
        ph.write(str_points)

def read_surface_mesh(surface_mesh_file, to_torch=False, subdivide=False):
    """ Returns (V,f) """
    tmesh = meshio.read(surface_mesh_file)
    snodes = tmesh.points
    try:
        sfaces = tmesh.cells['triangle']
    except:
        sfaces = tmesh.cells[0].data
    if subdivide:
        logging.info('Attempting subdivision')
        snodes, sfaces = trimesh.remesh.subdivide(snodes, sfaces)
    if to_torch:
        snodes = torch.FloatTensor(snodes)
        sfaces = torch.LongTensor(sfaces)
    return (snodes, sfaces)

def read_tet_mesh(tet_mesh_file, to_torch=False):
    """ Returns (nodes, elements) """
    tetmesh = meshio.read(tet_mesh_file)
    nodes = tetmesh.points
    elems = tetmesh.cells['tetra']
    if to_torch:
        nodes = torch.FloatTensor(nodes)
        elems = torch.LongTensor(elems)
    return (nodes, elems)

def write_tet_mesh(path, nodes, elems):
    points = nodes.detach().cpu().numpy()
    cells = {
        "tetra" : elems.detach().cpu().numpy()
    }
    meshio.write_points_cells(path, points, cells)

def norm_mesh(V, scale=None, use_mean=False):
    """
    Centers the mesh and scales it.
    If scale is given, does V_centered / scale.
    Else, it compute the maximum bounding box length 
        and uses that as the scale
    """
    ND = len(V.shape)
    if ND == 2: # |V| x 3
        if use_mean:
            mu = V.mean(dim=0)
        else:
            mu = bounding_box_center(V)
        if scale is None:
            mlen = mesh_scale(V)
        else:
            mlen = scale
        return (V - mu) / mlen
    elif ND == 3: # B x |V| x 3
        if use_mean:
            mus = V.mean(dim=1)
        else:
            mus = bounding_box_center(V)
        if scale is None:
            scales = mesh_scale(V)
        else:
            scales = scale
        return (V - mus) / scales

def bounding_box_center(V):
    ND = len(V.shape)
    if ND == 2: i = 0
    elif ND == 3: i = 1
    Smax = V.max(dim=i)[0]
    Smin = V.min(dim=i)[0]
    return (Smax + Smin) / 2.0

def mesh_scale(V):
    ND = len(V.shape)
    if ND == 2: i = 0
    elif ND == 3: i = 1
    Smax = V.max(dim=i)[0]
    Smin = V.min(dim=i)[0]
    mlen = (Smax - Smin).max(dim=-1)
    return mlen[0]

def rotate(angle, axis, V):
    """
    angle: (B x 1) 1D tensor of angles
    axis: (B x 3) Batch of 3D axes
    V: (B x |V| x 3) Batch of vertices
    """
    has_B_dim = (len(V.shape) == 3)
    if not has_B_dim: V = V.unsqueeze(0)
    assert (angle.shape[0] == axis.shape[0] == V.shape[0]), (
        "Got: "+str(angle.shape)+", "+str(axis.shape)+", "+str(V.shape) 
    )
    norms = (axis**2).sum(dim=1).sqrt().unsqueeze(1)
    aas = angle * axis / norms
    R = kornia.conversions.angle_axis_to_rotation_matrix(aas) # B x 3 x 3
    VT = V.permute(0,2,1) # B x 3 x |V|
    RVT = torch.bmm(R, VT)
    if not has_B_dim:
        return RVT.permute(0,2,1).squeeze(0)
    return RVT.permute(0,2,1)

def compute_closest_inds(V1, V2):
    """
    Computes the indices of which nodes in V1 correspond to those in V2.
    Inputs: |V1| x  3, |V2| x 3
    Outputs: |V2| long indices
    Assumes no batch dimension
    """
    D = pdist(V1, V2, norm=2, retsq=True) # |V1| x |V2|
    minDs = D.min(dim=0)
    return minDs[1].long()

def pdist(sample_1, sample_2, norm=2, eps=1e-5, retsq=False):
    """
    Compute the matrix of all squared pairwise distances.
    From:
        https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py

    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``.
    """
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if abs(norm - 2.0) < 1e-5:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        if retsq:
            return torch.abs(distances_squared)
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

#def compute_closest_inds_dep(V1, V2):
#    """
#    Computes the indices of which nodes in V1 correspond to those in V2.
#    Inputs: |V1| x  3, |V2| x 3
#    Outputs: |V2| long indices
#
#    NOTE: EVEN IF THE INPUTS ARE ON THE CPU, THE OUTPUT IS ON THE GPU.
#    THIS LEADS TO MEMORY ACCESS ERRORS.
#    """
#    sided_dist = kaolin.metrics.SidedDistance()
#    indices = sided_dist(V2, V1)
#    return indices

###################################################################################

def F_to_E(F, both_directions):
    """ 
    Map a face tensor (|F| x 3) to an edge tensor (a|F| x 2), a in {3,6}.
    If both_directions is True, add all 6 edges to the output (symmetric).
    """
    nF = F.shape[0]
    if both_directions:
        return F[:, [[0,1],[0,2],[1,0],[2,0],[1,2],[2,1]] ].view(6*nF, 2)
    else:
        return F[:, [[0,1], [0,2], [1,2]] ].view(3*nF, 2)

def mean_sq_edge_len(V, E):
    """
    V : B x |V| x 3
    E : |E| x 2
    """
    V_E = V[:,E] # B x |E| x 2 x 3 -> batch x edge x end_point_vertex x vertex coordinates
    # If we threshold the squared lengths by eps, then the actual length will be masked
    # like L^2 < eps cut out of the penalty implies L < sqrt(eps) is cut out.
    # E.g., a cutoff of 1e-3=0.001 implies a true length penalty cutoff of 0.03.
    # (E.g., or 1e-4 --> 0.01)
    # Or, if we cutoff the penalty at 100 length squared, then we are cutting off the penalty 
    # for a true length of 10.
    thresh_sq = 1e-4
    sq_lengths = ( (V_E[:,:,0,:] - V_E[:,:,1,:])**2 ).sum(dim = -1) # B x |E|
    mask = (sq_lengths > thresh_sq) # Keep only penalties on the larger lengths
    return ( sq_lengths * mask ).mean()

def mean_edge_len_variance(V, E):
    """
    V: B x |V| x 3
    E: |E| x 2
    """
    V_E = V[:,E] # B x |E| x 2 x 3 -> batch x edge x end_point_vertex x vertex coordinates
    lengths = ( (V_E[:,:,0,:] - V_E[:,:,1,:])**2 + 1e-7 ).sum(dim = -1).sqrt() # B x |E|
    len_vars = lengths.var(dim=1) # Variance over edge lengths
    return len_vars.mean()

def mean_sq_edge_len_variance(V, E):
    """
    V: B x |V| x 3
    E: |E| x 2
    """
    V_E = V[:,E] # B x |E| x 2 x 3 -> batch x edge x end_point_vertex x vertex coordinates
    sq_lengths = ( (V_E[:,:,0,:] - V_E[:,:,1,:])**2 ).sum(dim = -1) # B x |E|
    sq_lens_vars = sq_lengths.var(dim=1)
    return sq_lens_vars.mean()

#-----------------------------------------------------------------------------------------------#

def compute_surface_normals_per_face_single_mesh(V, F, eps=1e-7):
    """
    Input: V (|V| x 3, real), F (|F| x 3, int)
    Output: surface normals per face (|F| x 3, real)

    Warning: potentially not consistently oriented.
    """
    # Get the vertex coordinates of the sampled faces.
    face_verts = V[F] # |F| x 3 x 3
    # Verts per coords (each is |F| x 3)
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    # Cross product between nodes
    vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
    # Normalize results
    vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(min=eps)
    return vert_normals

def compute_surface_normals_per_face_batch_template(V, F, eps=1e-7):
    """
    Input: V (B x |V| x 3, real), F (|F| x 3, int)
    Output: surface normals per face (|F| x 3, real)

    Warning: potentially not consistently oriented.
    """
    # Obtain the coordinates per ith nodal entry per face
    v0 = torch.index_select(V, 1, F[:, 0]) # B x |F| x 3
    v1 = torch.index_select(V, 1, F[:, 1])
    v2 = torch.index_select(V, 1, F[:, 2])
    # Cross product between nodes-per-face
    crosses = torch.cross( (v1 - v0), (v2 - v1), dim = 2 )
    # Normalized face normals
    n_hat = crosses / crosses.norm(dim=2, p=2, keepdim=True).clamp(min=eps)
    return n_hat # B x |F| x 3

#-----------------------------------------------------------------------------------------------#

#########################
### MESH REGULARIZERS ###
#########################

#>> From Kaolin, based on the SoftRas paper <<#

# Soft Rasterizer (SoftRas)
# 
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        faces = faces.detach().cpu().numpy()

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x
        
# NOTE slow to set up

class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average
        
        faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            for face in faces:
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-5): #eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss

########################################################################################

# Modified from Kaolin #

def sample_triangle_mesh(vertices: torch.Tensor, 
                         faces: torch.Tensor,
                         num_samples: int, 
                         eps: float = 1e-5):
    r""" Uniformly samples the surface of a mesh.

    NOTE: we assume this is a batch of different instantiations of the same mesh.
    Thus, F is merely |F| x 3 and V can be B x |V|=N x 3, since
        |V| is always the same (though V changes) and F is always the same (as is |F|).

    Args:
        vertices (torch.Tensor): Vertices of the mesh (shape:
            :math:`B x N \times 3`, where :math:`N` is the number of vertices)
        faces (torch.LongTensor): Faces of the mesh (shape: :math:`F \times 3`,
            where :math:`F` is the number of faces).
        num_samples (int): Number of points to sample
        eps (float): A small number to prevent division by zero
                     for small surface areas.
    Returns:
        (torch.Tensor): Uniformly sampled points from the triangle mesh.
    Example:
        >>> points = sample_triangle_mesh(vertices, faces, 10)
        >>> points
        tensor([[ 0.0293,  0.2179,  0.2168],
                [ 0.2003, -0.3367,  0.2187],
                [ 0.2152, -0.0943,  0.1907],
                [-0.1852,  0.1686, -0.0522],
                [-0.2167,  0.3171,  0.0737],
                [ 0.2219, -0.0289,  0.1531],
                [ 0.2217, -0.0115,  0.1247],
                [-0.1400,  0.0364, -0.1618],
                [ 0.0658, -0.0310, -0.2198],
                [ 0.1926, -0.1867, -0.2153]])
    """
    B, nV, _ = vertices.shape
    F, _ = faces.shape

    dist_uni = torch.distributions.Uniform(torch.tensor([0.]).to(
        vertices.device), torch.tensor([1.]).to(vertices.device))

    # Calculate area of each face
    # split -> like the inverse of cat. 
    #          split(T, s, d) means split T into equal size chunks along dimension d
    #          where the size of a single chunk is s
    # index_select -> same as regular bool-tensor indexing
    #                 see: https://stackoverflow.com/questions/59344751/is-there-any-diffrence-between-index-select-and-tensorsequence-in-pytorch
    #                 i.e., T[:,:,inds,:] <==> index_select(T, 2, inds)

    # Note we can index_select, rather than gather, here, since faces is identical 
    #   over the batch.

    x1, x2, x3 = torch.split(
            torch.index_select(vertices, 1, faces[:, 0])  # B x |F| x 3
            - 
            torch.index_select(vertices, 1, faces[:, 1]), 
            1, # Chunk the target dimension into individual tensors (i.e. size=1)
            dim=2) # Chunk along xyz coords dimension
    y1, y2, y3 = torch.split(
            torch.index_select(vertices, 1, faces[:, 1]) 
            - 
            torch.index_select(vertices, 1, faces[:, 2]), 
            1,     
            dim=2)
    a = (x2 * y3 - x3 * y2)**2 # Each of xi & yi is B x |F|
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    Areas = ( torch.sqrt(a + b + c + eps) / 2 ).squeeze(-1)
    # percentage of each face w.r.t. full surface area
    Areas = Areas / ( Areas.sum(dim=-1, keepdim=True) + eps) # B x |F|
    # define discrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(B,F) + eps)
    face_choices = cat_dist.sample( (num_samples,) ).T # B x N_S

    # from each face sample a point
    select_faces = faces[face_choices] # B x N_S x 3

    v1s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,0].unsqueeze(-1).expand(-1,-1,3)) 
    v2s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,1].unsqueeze(-1).expand(-1,-1,3)) 
    v3s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,2].unsqueeze(-1).expand(-1,-1,3))     
    u = torch.sqrt(dist_uni.sample([B, num_samples]))
    v = dist_uni.sample([B, num_samples])
    points = (1 - u) * v1s + (u * (1 - v)) * v2s + u * v * v3s
    return points

def sample_triangle_mesh_with_normals(vertices: torch.Tensor, 
                                      faces: torch.Tensor,
                                      num_samples: int, 
                                      eps: float = 1e-5):
    r""" Uniformly samples the surface of a mesh, and extracts the associated surface normals.

    NOTE: we assume this is a batch of different instantiations of the same mesh.
    Thus, F is merely |F| x 3 and V can be B x |V|=N x 3, since
        |V| is always the same (though V changes) and F is always the same (as is |F|).

    Args:
        vertices (torch.Tensor): Vertices of the mesh (shape:
            :math:`B x N \times 3`, where :math:`N` is the number of vertices)
        faces (torch.LongTensor): Faces of the mesh (shape: :math:`F \times 3`,
            where :math:`F` is the number of faces).
        num_samples (int): Number of points to sample
        eps (float): A small number to prevent division by zero
                     for small surface areas.
    Returns:
        V, n_hat (torch.Tensor, torch.tensor): 
            Uniformly sampled points from the triangle mesh along with their normals.
    """
    B, nV, _ = vertices.shape
    F, _ = faces.shape

    # Precompute surface normals per face across the batch
    # B x |F| x 3
    #face_surface_normals = compute_surface_normals_per_face_batch_template(V, F, eps=1e-7)

    dist_uni = torch.distributions.Uniform(torch.tensor([0.]).to(
                    vertices.device), torch.tensor([1.]).to(vertices.device))

    # Obtain the coordinates per ith nodal entry per face
    v0 = torch.index_select(vertices, 1, faces[:, 0]) # B x |F| x 3
    v1 = torch.index_select(vertices, 1, faces[:, 1])
    v2 = torch.index_select(vertices, 1, faces[:, 2])
    # Cross product between nodes-per-face
    crosses = torch.cross( (v1 - v0), (v2 - v1), dim = 2 )
    # Normalized face normals (B x |F| x 3)
    n_hat = crosses / crosses.norm(dim=2, p=2, keepdim=True).clamp(min=eps)    

    # Calculate area of each face
    x1, x2, x3 = torch.split(v0 - v1, 
            1, # Chunk the target dimension into individual tensors (i.e. size=1)
            dim=2) # Chunk along xyz coords dimension
    y1, y2, y3 = torch.split(v1 - v2,
            1,     
            dim=2)
    a = (x2 * y3 - x3 * y2)**2 # Each of xi & yi is B x |F|
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    abc_sum = (a + b + c).clamp(min=eps)
    Areas = ( torch.sqrt(abc_sum) / 2 ).squeeze(-1)
    # percentage of each face w.r.t. full surface area
    # After this, "areas" holds the proportion of each triangle in the mesh (i.e., in [0,1])
    # It seems the total area may be sufficiently large to cause some numerators to go to zero
    Areas = Areas / ( Areas.sum(dim=-1, keepdim=True).clamp(min=eps) ) # B x |F|
    # Add an additional smoothing correction to ensure positive multinomial probs
    Areas = (Areas.clamp(min=eps) + 1e-5)
    Areas = Areas / Areas.sum(dim=-1, keepdim=True)

    # NOTE TO SELF:
    # THE ERROR IS NOT HERE
    # IT WAS A NAN BEFORE-HAND, AND ARRIVED HERE
    # SEEMS TO COME FROM THE BACKWARD STEP, NOT A PARTICULAR LOSS

    #print(Areas[Areas < 0.001])
    # define discrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(B,F))
    face_choices = cat_dist.sample( (num_samples,) ).T # B x N_S

    # from each chosen face sample a point
    # faces : |F| x 3
    select_faces = faces[face_choices] # B x N_S x 3

    # Gather nodal points of the chosen faces
    v1s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,0].unsqueeze(-1).expand(-1,-1,3)) 
    v2s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,1].unsqueeze(-1).expand(-1,-1,3)) 
    v3s = torch.gather(vertices, 
                      dim=1, 
                      index=select_faces[:,:,2].unsqueeze(-1).expand(-1,-1,3))     
    u = torch.sqrt(dist_uni.sample([B, num_samples]))
    v = dist_uni.sample([B, num_samples])
    points = (1 - u) * v1s + (u * (1 - v)) * v2s + u * v * v3s

    # Gather the face normals of the chosen faces
    normals = torch.gather(n_hat, # B x |F| x 3
                           # index only selects in the face dimension 
                           dim = 1,
                           # face_choices is B x N_S -> must be duplicated along the coords axis
                           index = face_choices.unsqueeze(-1).expand(-1,-1,3) )

    return points, normals
















#
