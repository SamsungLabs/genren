import trimesh, torch

def read_obj_into_pc(path, n_points, as_torch=True):
    mesh = trimesh.load(path)
    if as_torch:
        return torch.from_numpy(mesh.sample(n_points))
    return mesh.sample(n_points) # n_points x 3






















#
