import dgl
import torch, torch.nn as nn, torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv as GCN

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, outdim):
        super(GCNClassifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.elu),
            GCN(hidden_dim, hidden_dim, F.elu),
            GCN(hidden_dim, hidden_dim, F.elu)
        ])
        self.classify = nn.Linear(hidden_dim, outdim)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as out_degree.
        #h = g.in_degrees().view(-1, 1).float()
        h = g.ndata['features']
        # GCN processing
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h') # Should be B x hidden_dim
        return self.classify(hg) # Now B x outdim


class GcnPatchCritic(nn.Module):
    def __init__(self, in_dim, hidden_dims, graph_structure_batch):
        super(GcnPatchCritic, self).__init__()

        assert len(hidden_dims) == 3

        self.c1 = GCN(in_dim, hidden_dims[0], activation = F.leaky_relu)
        self.f1 = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.LeakyReLU())
        self.c2 = GCN(hidden_dims[1], hidden_dims[2], activation = F.leaky_relu)
        self.f2 = nn.Linear(hidden_dims[2], 1)

        self.G = graph_structure_batch

        # self.clayers = nn.ModuleList([
        #     GCN(in_dim, hidden_dim, F.leaky_relu),
        #     GCN(hidden_dim, hidden_dim, F.leaky_relu)
        # ])
        # self.flayers = nn.ModuleList([
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, 1)
        # ])

    def forward(self, T):
        """
        Process a per-node texture with a small number of graph conv layers (to
            get a small patch critic).

        Args:
            T: a pytorch tensor (B x |V| x 3)
        """
        B, nV, _ = T.shape
        # For undirected graphs, in_degree is the same as out_degree.
        #h = g.in_degrees().view(-1, 1).float()
        #h = g.ndata['features']
        # GCN processing
        #for conv in self.clayers:
        #    h = conv(g, h)
        #g.ndata['h'] = h
        #hg = dgl.mean_nodes(g, 'h') # Should be B x 1
        T = T.view(B*nV, 3)
        h = self.c1(self.G, T)
        h = self.f1(h.view(B,nV,-1)).view(B*nV,-1)
        h = self.c2(self.G, h)
        h = self.f2(h.view(B,nV,-1)).view(B*nV,-1) # B x 1
        return h.squeeze(-1) # B
        #return h.mean() # Now B x 1 -> scalar







#
