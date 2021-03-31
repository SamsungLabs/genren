import torch, torch.nn.functional as F, numpy as np

"""
Based on:
https://github.com/eifuentes/swae-pytorch/blob/master/swae/trainer.py
"""

def rand_projections(num_samples, embedding_dim):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    eps = 1e-5
    projections = [ (w + eps) / np.sqrt( ( (w + eps)**2 ).sum() )  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)

def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections,
                                 p=2):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples [n_ST x D]
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples [n_SL x D]
            num_projections (int): number of projections to approximate sliced wasserstein distance 
            p (int): power of distance metric
        Return:
            torch.Tensor: tensor of wasserstein distances of size (num_projections, 1)
    """
    # We need to have the same number of samples, else the simple sorting approach will not work.
    assert (	 len(encoded_samples.shape) 	 == 2 
    		 and len(distribution_samples.shape) == 2 
    		 and encoded_samples.shape 		 	 == distribution_samples.shape)
    # Derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # Generate random projections in latent space
    projections = rand_projections(num_projections, embedding_dim).to(encoded_samples.device) # [n_P x D]
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1)) # (n_SL x D) @ (D x n_P) -> (n_SL x n_P)
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1))) # (n_ST x D) @ (D x n_P) -> (n_ST x n_P)
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    # Note: the transpose is so that the sort (along dimension = -1) is over the samples dimension
    #	I.e., sorted_mat[i, 0] = <projection i, sample 0> is the largest one (since the sort is ascending)
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]) #
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()






#
