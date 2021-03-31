import torch
import dgl

def slab_collate(samples):
    # Collator for single-label graphs
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def gcollate(graphs):
    """ Collates a list of DGL graphs """
    return dgl.batch(graphs)
    















#
