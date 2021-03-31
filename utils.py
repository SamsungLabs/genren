import json, torch, logging, pathlib, numpy as np, time
import os, sys
import torch.nn as nn

def load_s2torch_json(file_path):
    """
    Assumes we are reading a dict mapping strings to list-form
        pytorch tensors.
    """
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return { key : torch.FloatTensor(value) 
             for key, value in data.items() }

def covariance(m, rowvar=False):
    """
    m: B x |v| (by default)
    
    Returns:
        covariance of v (|v| x |v|)
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m  = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def euclidean_random_far_point_indices(P, nT, nS = 1000):
    """
    P -> points |V| x 3
    nT -> number of samples to take
    nS -> number of random retries to do per choice
    
    Return: list of nT indices of "far apart" points
    """
    assert type(nT) is int and type(nS) is int
    assert len(P.shape) == 2, "Found " + str(P.shape)
    nV, _ = P.shape
    if nS > nV:
        nS = nV - 1
    chosen = [ int(np.random.choice(nV, 1)[0]) ] # initialization (indices)
    chosen_p = [ P[chosen[0]] ] # values
    sqdist = lambda u,v: np.inner(u-v, u-v)
    for i in range(nT - 1):
        index_samples = np.random.choice(nV, nS, replace = False)
        # Get the largest average distance across all points
        samples = P[index_samples, :] # nS x 3, positions
        # Take the new sample that is farthest from its closest existing choice
        min_dist_to_any_curr_chosen = lambda r:  min( [ sqdist(c,r) for c in chosen_p ] )
        min_dists = [ min_dist_to_any_curr_chosen(samples[i,:]) for i in range(nS) ]
        ind_of_max_dist = np.argmax(min_dists) # nS values (min dists) -> 1 index into the sample set
        new_chosen_ind = index_samples[ind_of_max_dist] # convert samples index to vertices index
        new_chosen_val = P[new_chosen_ind, :]
        chosen.append(new_chosen_ind)
        chosen_p.append(new_chosen_val)
    return chosen

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

    def sample(self):
        return next(self)

#-------------------------------------------------------------------------------------------------#

def rgb_to_xyz_vec(v):
    r = v[..., 0]
    g = v[..., 1]
    b = v[..., 2]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack((x, y, z), -1)

    return out

def rgb_to_xyz_img(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    From the Kornia library.
    Since my version is too early.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack((x, y, z), -3)

    return out

class Rgb2CieLAB(nn.Module):
    def __init__(self):
        super(Rgb2CieLAB, self).__init__()
        self.xyz2cielab = Xyz2CieLAB()
        # Gamma expansion constants
        self.thresh = 0.04045

    def forward(self, v):
        # Gamma expansion
        rgb_linear = self.gamma_expand(v)
        return self.xyz2cielab( rgb_to_xyz_vec(v) )

    def gamma_expand(self, v):
        mask = (v < self.thresh).long().nonzero()
        #a = torch.zeros(v.shape)
        a = ( (v + 0.055) / 1.055).pow(2.4)
        a[mask] = (v / 12.92)[mask]
        return a

class Xyz2CieLAB(nn.Module):

    def __init__(self, illuminant = 'd65'):
        super(Xyz2CieLAB, self).__init__()
        if illuminant == 'd65':
            self.Xn = 95.0489
            self.Yn = 100.0
            self.Zn = 108.8840
        elif illuminant == 'd50':
            self.Xn = 96.4212
            self.Yn = 100
            self.Zn = 82.5188
        # CIELAB constants 
        self.delta       = 6.0 / 29.0
        self.delta_cubed = self.delta**3
        self.c1          = 3 * self.delta**2
        self.c2          = 4.0 / 29.0

    def f(self, t):
        mask    = (t > self.delta_cubed).long().nonzero()
        p       = (t / self.c1) + self.c2
        p[mask] = t.clamp(min=1e-6).pow(1.0 / 3.0)[mask]
        return p

    def forward(self, q):
        """
        input is a colour vector (shape: S_1 x  ... x S_n x 3)
        """
        assert q.shape[-1] == 3
        return self.xyz_to_cielab(q)

    def xyz_to_cielab(self, image: torch.Tensor) -> torch.Tensor:
        X      = image[..., 0]
        Y      = image[..., 1]
        Z      = image[..., 2]
        f_y    = self.f(Y / self.Yn)
        L_star = 116.0 * f_y - 16.0
        a_star = 500.0 * ( self.f(X / self.Xn) - f_y )
        b_star = 200.0 * ( f_y - self.f(Z / self.Zn) )
        return torch.stack((L_star, a_star, b_star), dim = -1)

#-------------------------------------------------------------------------------------------------#

class Accumulator(object):
    """
    Tracks values in a dictionary.
    Note that a key-value dictionary is passed to input.
    Inputs (dict values) should be `float` or a 1D pytorch tensor.

    Call `clear()` or `reset()` to reset the object, 
         `update(input)` to add another entry,
         and `means_dict()` or `get_current_means_dict()` to get the current means.
    """
    def __init__(self):
        self.reset()

    def update(self, values_dict):
        if not self.data:
            for key in values_dict:
                self.data[key] = []
        for key in values_dict:
            v = values_dict[key]
            if not (type(v) is float):
                v = float( v.cpu().detach().numpy() )
            self.data[key].append( v )

    def clear(self):
        self.reset()

    def reset(self):
        self.data = {}

    def get_current_means_dict(self):
        return { key : _mean( self.data[key] )
                 for key in self.data }

    def means_dict(self):
        return self.get_current_means_dict()

    def names(self):
        return self.data.keys()

    def csv_means_string(self, prepend_str=""):
        names = self.names()
        loss_terms = self.means_dict()
        return ", ".join([ prepend_str + name + (': %.4f' % loss_terms[name]) for name in names ])

def _mean(L):
    return sum(L) / len(L)

class AccumTimer(Accumulator):
    
    def __init__(self):
        self.reset()
        self.start_times = {}
    
    def update(self, values_dict):
        #if not self.data:
        for key in values_dict:
            if not key in self.data.keys():
                self.data[key] = []
        for key in values_dict:
            v = values_dict[key]
            if not (type(v) is float):
                v = float( v.cpu().detach().numpy() )
            self.data[key].append( v )

    def reset(self):
        self.data = {}
        # By not resetting here, we can reset the accumulator in the middle of a tic-toc call
        #self.start_times = {}

    def start(self, name):
        tic = time.perf_counter()
        self.start_times[name] = tic

    def end(self, name):
        toc = time.perf_counter()
        tic = self.start_times[name]
        diff = toc - tic 
        self.update({name : diff})
    
    def csv_means_string(self, prepend_str=""):
        names = self.names()
        loss_terms = self.means_dict()
        #print('ll', loss_terms)
        #sys.exit(0)
        return "\n\t" + "\n\t".join([ prepend_str + name + (': %.2fms' % (loss_terms[name]*1000)) for name in names ])

#-------------------------------------------------------------------------------------------------#

### From:
### https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7

# create_graph: contructs the derivative graph, which we can backprop through
# retain_graph: prevents the computation graph (of the the value, not derivative) from 
#       being freed so we can backprop through it later

def batch_jacobian_ap(ys, xs, create_graph=True):
    B = ys.shape[0]
    return torch.stack([ jacobian_ap(ys[i], xs[i], create_graph=create_graph) 
                         for i in range(B) ])

def jacobian_ap(y, x, create_graph=False):                                                               
    print('in jac', x.shape, y.shape)
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):              
        grad_y[i] = 1.0                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.0                                                                               
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian_ap(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)     

def batch_jacobian_and_hessian(ys, xs, create_graph=True):
    B = ys.shape[0]
    jacobians = [ jacobian_ap(
                        ys[i], 
                        xs[i], 
                        create_graph=create_graph)
                  for i in range(B) ]
    hessians  = [ jacobian_ap(
                        jacobians[i], 
                        xs[i], 
                        create_graph=create_graph)
                  for i in range(B) ]
    return torch.stack(jacobians), torch.stack(hessians)

### From:
### https://github.com/mariogeiger/hessian

def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=False,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])

def batch_jacobian(ys, xs, create_graph=True):
    B = ys.shape[0]
    return torch.cat([ jacobian(ys[i], xs[i], create_graph=create_graph) 
                         for i in range(B) ], dim=0)

def jacobian(outputs, inputs, create_graph=False):
    '''
    Compute the Jacobian of `outputs` with respect to `inputs`
    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    '''
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)

def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

#####################################################################################

def batch_mm(matrix, vector_batch):
    # https://github.com/pytorch/pytorch/issues/14489
    batch_size = vector_batch.shape[0]
    # Stack the vector batch into columns. (b, n, 1) -> (n, b)
    vectors = vector_batch.transpose(0, 1).reshape(-1, batch_size)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, b) -> (b, m, 1)
    return matrix.mm(vectors).transpose(1, 0).reshape(batch_size, -1, 1)

#

def random_unit_3vectors(B):
    n = torch.randn(B, 3)
    n_hat = torch.nn.functional.normalize(n, p=2, dim=1, eps=1e-7)
    return n_hat

def rfp(a, s=True):
    # Input is a string
    if type(a) is str and s:
        return a
    elif type(a) is str and (not s):
        a = float(a)
    # 
    if a is None and s:
        return 'None'
    if type(a) is float:
        f = a
    else:
        f = float(a.detach().cpu().numpy())
    if s: f = ( '%.3f' % f )
    return f

def ppd(s='', d={}, indent=4, sort=False, rm_zeros=True):
    if rm_zeros:
        d = { k : v for k,v in d.items() if abs(v) > 1e-8 }
    logging.info( s + json.dumps(d, indent=indent, sort_keys=sort) )

def prepend_to_dict(s, d):
    return { s + k : v for k, v in d.items() }

def write_all_to_tensorboard(board_writer, dict_of_scalars, step, prepend_string=None):
    if not prepend_string is None:
        dict_of_scalars = prepend_to_dict(prepend_string, dict_of_scalars)
    for k, v in dict_of_scalars.items():
        if v is None: continue
        if type(v) is str and v.lower().strip() == 'none': continue
        board_writer.add_scalar(tag=k, scalar_value=rfp(v,s=False), global_step=step)

### Methods to get the current git commit ###

def get_git_revision(base_path):
    git_dir = pathlib.Path(base_path) / '.git'
    with (git_dir / 'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()
    with (git_dir / ref).open('r') as git_hash:
        return git_hash.readline().strip()

def execution_path():
    return pathlib.Path(__file__).parent.absolute()

def get_current_git_revision():
    return get_git_revision(execution_path())

def log_current_git_revision_safe():
    try:
        logging.info('Current git commit ' + get_current_git_revision())
    except:
        logging.info('Failed to find current git commit')

def bool_string_type(x):
    x = x.strip().lower()
    assert x in ['true', 'false']
    return True if (x == 'true') else False

#####

def main():
    with torch.no_grad():
        #r = torch.rand(500000, 3)
        r = torch.linspace(0.0, 1.0, 120)
        t1, t2, t3 = torch.meshgrid(r,r,r)
        r = torch.stack( (t1,t2,t3), -1 ).view(-1,3)
        u = Rgb2CieLAB()
        a = u(r)
        b = u( r * 2 - 1 )
        c = u( r * 255 )
        d = u( r * 128 )
        print('0,1:',   a.min(0)[0], a.max(0)[0])
        print('-1,1:',  b.min(0)[0], b.max(0)[0])
        print('0,255:', c.min(0)[0], c.max(0)[0])
        print('0,128:', d.min(0)[0], d.max(0)[0])

#-------------------------#
if __name__ == "__main__":
    main()
#-------------------------#




#
