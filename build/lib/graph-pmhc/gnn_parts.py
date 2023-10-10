import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch import WeightAndSum
import itertools
import torch.nn as nn
import numpy as np
import pandas as pd

#All possible graphs (can't handle peptides longer than length 30 or shorter than 9) are generated ahead of time, and then just looked up for the given datum
def nnalign_generate_combination_lookup_table(mhc_adj,mhc_lens,bc_pad):
    #Make a tensor with all possible combinations of peptide and flank lengths, binding core starting position, and allele gene
    lengths = np.arange(8, 38)
    nflank_lengths = np.arange(1, 6)
    cflank_lengths = np.arange(1, 6)
    lengths = np.array([np.repeat(length, 30+bc_pad*2) if length!=8 else np.repeat(length, 9) for length in lengths],dtype=object)
    positions = np.array([np.array([i for i, _ in enumerate(length)]) for length in lengths], dtype=object)

    lengths = np.array(list(itertools.chain.from_iterable(lengths)))
    positions = np.array(list(itertools.chain.from_iterable(positions)))

    nflank_lengths = [np.repeat(i, len(positions)) for i in nflank_lengths]
    nflank_lengths = np.array(list(itertools.chain.from_iterable(nflank_lengths)))
    positions = np.array(list(positions) * 5)
    lengths = np.array(list(lengths) * 5)
    
    cflank_lengths = [np.repeat(i, len(positions)) for i in cflank_lengths]
    cflank_lengths = np.array(list(itertools.chain.from_iterable(cflank_lengths)))
    nflank_lengths = np.array(list(nflank_lengths) * 5)
    positions = np.array(list(positions) * 5)
    lengths = np.array(list(lengths) * 5)

    alleles = np.array([np.repeat(i, len(positions)) for i in [0,1,2]]).reshape(-1)
    positions = np.array(list(positions) * 3)
    cflank_lengths = np.array(list(cflank_lengths) * 3)
    nflank_lengths = np.array(list(nflank_lengths) * 3)
    lengths = np.array(list(lengths) * 3)

    #generate all the graphs
    graph, edge = zip(*[generate_graph_for_lookup_table(mhc_adj,mhc_lens, length, nflank_length, cflank_length, allele, pos) for length, nflank_length, cflank_length, allele, pos in
                        zip(lengths, nflank_lengths, cflank_lengths, alleles, positions)])
    #put the features that define a graph into a lookup dataframe that is aligned with the graph and edge features
    lookup_df = pd.DataFrame({'lengths':lengths, 'positions': positions, 'nflank_lengths': nflank_lengths, 'cflank_lengths': cflank_lengths, 'alleles':alleles})

    lookup_table = [lookup_df, graph, np.array(edge,dtype=object)]
    return lookup_table

#This defines a particular graph
def generate_graph_for_lookup_table(mhc_adj,mhc_lens,peptide_length, nflank_length, cflank_length, allele, position=None):
    #Different genes get different psuedosequences/adjacency matrices
    if allele ==0:
        mhc_length = mhc_lens[0]
        graph_idx =  mhc_adj[0].copy()
    elif allele == 1:
        mhc_length = mhc_lens[1]
        graph_idx =  mhc_adj[1].copy()
    elif allele == 2:
        mhc_length = mhc_lens[2]
        graph_idx =  mhc_adj[2].copy()
    #for 8mers you delete a position from the 9mers, position is the binding core starting position, 8mers not currently supported
    if peptide_length == 8:
        if len(graph_idx)>0:
            del graph_idx[position]

    #9mers are a special case because it is the length of the binding core
    if peptide_length!=9:
        #binding core starting position starts in the peptide, it gets offset by the nflank
        position = position + nflank_length
    #offset the graph idx (mhc positions) by the length of the peptide/flanks
    graph_idx = [[i + nflank_length + peptide_length + cflank_length for i in g_i] for g_i in graph_idx]
    #u are starting locations and v are ending locations for defining the edges of the graph
    #this gets the intramolecular edges, eg those between adjacent positions of the peptide, or mhc
    u = list(np.arange(nflank_length + 0, nflank_length + peptide_length - 1)) + \
        list(np.arange(nflank_length + peptide_length + cflank_length, mhc_length-1 + nflank_length + peptide_length + cflank_length)) + \
        list(np.arange(nflank_length + 1, nflank_length + peptide_length)) + \
        list(np.arange(nflank_length + peptide_length + cflank_length + 1, mhc_length-1 + nflank_length + peptide_length + cflank_length + 1))
    v = list(np.arange(nflank_length + 1, nflank_length + peptide_length)) + \
        list(np.arange(nflank_length + peptide_length + cflank_length + 1, mhc_length-1 + nflank_length + peptide_length + cflank_length + 1)) + \
        list(np.arange(nflank_length + 0, nflank_length + peptide_length - 1)) + \
        list(np.arange(nflank_length + peptide_length + cflank_length, mhc_length-1 + nflank_length + peptide_length + cflank_length))

    peptide_bonds = len(u)
    
    edge = []
    #[0,0,1] is an intramolecular edge
    edge = [edge + [0, 0, 1] for _ in range(peptide_bonds)]
    #peptide index is the positions of the binding core of the peptide, for 8mers get rid of the edges not used after deletion for 9mers
    if peptide_length == 8:
        peptide_index = list(np.arange(nflank_length, nflank_length + 9))
        peptide_index = [idx if idx < position else idx - 1 for idx in peptide_index]
        del peptide_index[(position-nflank_length)]

    else:
        peptide_index = list(np.arange(nflank_length, nflank_length + peptide_length))
    #binding core is only 9 long
    if peptide_length > 9:
        peptide_index = peptide_index[position-nflank_length:position - nflank_length + 9]
    #get the adjacencies between flanks and peptides
    nflank_u = list(np.arange(0, nflank_length)) + list(np.arange(1, nflank_length + 1))
    nflank_v = list(np.arange(1, nflank_length + 1)) + list(np.arange(0, nflank_length))

    u = nflank_u + u
    v = nflank_v + v

    cflank_u = list(np.arange(nflank_length + peptide_length-1, nflank_length + peptide_length + cflank_length-1)) + list(np.arange(nflank_length + peptide_length, cflank_length + nflank_length + peptide_length))
    cflank_v = list(np.arange(nflank_length + peptide_length, cflank_length + nflank_length + peptide_length)) + list(np.arange(nflank_length + peptide_length-1, cflank_length + nflank_length + peptide_length-1))
    u = u + cflank_u
    v = v + cflank_v
    #[0, 1, 0] is the flank edge
    edge = [[0, 0, 1] for _ in range(len(nflank_u) // 2 - 1)] + [[0, 1, 0]] + \
            [[0, 0, 1] for _ in range(len(nflank_u) // 2 - 1)] + [[0, 1, 0]] + edge

    edge = edge + [[0, 1, 0]] + [[0, 0, 1] for _ in range(len(cflank_u) // 2 - 1)] + \
             [[0, 1, 0]] + [[0, 0, 1] for _ in range(len(cflank_u) // 2 - 1)]
    
    #this gets all the intermolecular edges, those between the peptide and mhc
    [u.extend(np.repeat(i, len(g))) for i, g in zip(peptide_index, graph_idx)]
    [v.extend(g) for g in graph_idx]

    [v.extend(np.repeat(i, len(g))) for i, g in zip(peptide_index, graph_idx)]
    [u.extend(g) for g in graph_idx]

    temp = []
    #[1, 0, 0] is the intermolecular edge starting position
    edge = edge + [temp + [1, 0, 0] for _ in range(len(u) - peptide_bonds - len(nflank_u) - len(cflank_u))]
    
    #get the graph using the starting and ending positions for all edges
    graph = dgl.DGLGraph((u, v))
    #get a list of edges for the graph
    edge = np.array(edge).reshape(-1,3)
    return graph, edge

#this looks up all the graphs in a batch during the model forward
def lookup_graph(features, features_token, lengths, allele, lookup_table,bc_pad):
    allele = allele.cpu()
    features = features.cpu()
    features_token = features_token.cpu()
    for ix,length in enumerate(lengths):
        lengths[ix] = length.cpu()
    #initial length is the batch size
    initial_length = len(lengths[1]) 
    #break down lengths into the nflank, cflank and peptide lengths, 
    # which keep track of how many nodes there will be in the final graph
    nflank_lengths = lengths[0]
    cflank_lengths = lengths[2]
    lengths = lengths[1]

    #For graph enumeration we duplicate all peptides length - 8 times (eg for a 10mer there are two valid graphs), except 8mers which have 8 valid graphs
    #Bool index keeps track of how many repeats there are, it's a square matrix with a 1 for each repeat for a given length. Max 22 repeats for length 30
    #  shape (batch_size, 16)
    bool_index = torch.stack([nn.functional.pad(torch.tensor(1).repeat(length-8), (0, (30+bc_pad-length)))
                if length > torch.tensor(8)
                else nn.functional.pad(torch.tensor(1).repeat(8), (0, 14))
                for length in lengths])
    #Here we generate the index that keeps track of which peptide each graph comes from. The arange generates all the indices for the peptides
    #  and the einsum replicates the indices for each graph. Then we flatten it and remove the empties since they won't go to the model. Output shape ~8*batch_size
    index = torch.einsum('bi,b->bi', bool_index, torch.arange(1, 1+initial_length)).view(-1)
    index = index[index.nonzero()].view(-1)-1

    #Here we repeat the node features (node features are input as (batch_size, 69) in the same way and flatten them and remove padding
    #  because the final graph is just one huge graph. (output shape ~8*batch_size*~61)
    
    features = torch.einsum('bi,bjk->bijk', bool_index, features).reshape(-1,features.shape[-1])

    features_token = torch.einsum('bi,bj->bij', bool_index, features_token).view(-1)
    nonzero = features_token.nonzero()

    features = features[nonzero].squeeze()
    

    #positions tells us where the 9mer binding core starts for each peptide, we enumerate all possible starting points.
    #We enumerate it and nflank_lengths square since the peptide lengths that would make invalid graphs are set to 0 so they will be dropped afer lookup
    #output positions (batch_size* ~8)
    positions = torch.arange(22+bc_pad).repeat(initial_length)
    lengths = torch.einsum('bi,b->bi', bool_index, lengths).view(-1)
    nflank_lengths = torch.einsum('bi,b->bi', bool_index, nflank_lengths).view(-1)
    cflank_lengths = torch.einsum('bi,b->bi', bool_index, cflank_lengths).view(-1)
    allele = torch.einsum('bi,b->bi', bool_index, allele).view(-1)
    
    #Here we use pandas to vectorize the graph lookup
    #Now that we have positions, nflank length, and peptide lengths for every possible graph in the batch we need to look them up.
    #Lookup table is a list with these components [lengths, positions, nflank lengths, graphs, edges]
    #This is best described by example:               10       0             5         graph1  edge tensor1
    #                                                 10       1             5         graph2  edge tensor2
    #                                                 10       2             5         graph3  edge tensor3
    #So when length 10, position 1, nflank length 5 is presented we graph graph2 and edge tensor 2. lookup_table contains all possibilities
    condition_df= pd.DataFrame({'lengths':lengths, 'positions': positions, 'nflank_lengths': nflank_lengths, 'cflank_lengths': cflank_lengths, 'alleles':allele, 'idx':np.arange(len(positions))})
    #This merge sorts the input lengths, positions by an index of where they appear in the lookup table, and then just that index is defined, and sorted by how they originally appeared 
    condition_index = lookup_table[0].reset_index().merge(condition_df,how='right').sort_values(by=['idx'])['index']
    #All the padding will be filled with nans so dropping them gives the same shape as index above
    condition_index = condition_index.dropna()

    #Here we grab the appropriate graphs
    graph = [lookup_table[1][int(idx)] for idx in condition_index]

    #batch graphs, DGL team if you ever read this, it is super, crazy, slow
    graph = dgl.batch(graph)

    #Here we grab the appropriate edge tensors
    edge = [lookup_table[2][int(idx)] for idx in condition_index]
    #Here we flatten the edge tensors because we now have just one huge graph, output shape (~8*batch_size, 3)
    edge = torch.cat([torch.tensor(ee) for ee in edge])
    return graph, features, edge, index

#lookup one graph per data after best allele/binding core start position is known
def lookup_single_graph(features, features_token, lengths, allele, lookup_table, positions):
    allele=allele.cpu()
    features = features.cpu()
    features_token = features_token.cpu()
    for ix,length in enumerate(lengths):
        lengths[ix] = length.cpu()
    #initial length is the batch size
    initial_length = len(lengths[1]) 
    #break down lengths into the nflank, cflank and peptide lengths, 
    # which keep track of how many nodes there will be in the final graph
    nflank_lengths = lengths[0]
    cflank_lengths = lengths[2]
    lengths = lengths[1]

    condition_df= pd.DataFrame({'lengths':lengths, 'positions': positions, 'nflank_lengths': nflank_lengths, 'cflank_lengths': cflank_lengths, 'alleles':allele, 'idx':np.arange(len(positions))})
    #This merge sorts the input lengths, positions by an index of where they appear in the lookup table, and then just that index is defined, and sorted by how they originally appeared 
    condition_index = lookup_table[0].reset_index().merge(condition_df,how='right').sort_values(by=['idx'])['index']
    #All the padding will be filled with nans so dropping them gives the same shape as index above
    condition_index = condition_index.dropna()

    #Here we grab the appropriate graphs
    graph = [lookup_table[1][int(idx)] for idx in condition_index]

    #Here we batch them into one huge graph for dgl, this is about 63% of the total run time
    graph = dgl.batch(graph)

    #Here we grab the appropriate edge tensors
    edge = [lookup_table[2][int(idx)] for idx in condition_index]
    #Here we flatten the edge tensors because we now have just one huge graph, output shape (~8*batch_size, 3)
    edge = torch.cat([torch.tensor(ee) for ee in edge])
    
    features = features.view(-1,features.shape[-1])
    nonzero = features_token.view(-1).nonzero().squeeze()
    features = features[nonzero]
    return graph, features, edge



#Layers
class PositionalEncoderSimple(nn.Module):
    def __init__(self, model_dim, dropout, word_dropout=None, max_seq_len=121, concat=False):
        super(PositionalEncoderSimple, self).__init__()
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
    
    def forward(self,x):
        b, n, _, device = *x.shape, x.device
        emb = self.dropout(self.pos_emb(torch.arange(n, device = device)))
        x += emb
        return x

#These are all from the DGL team, and much of it from their implimentation of attentiveFP, which is really well done
class GlobalPool(nn.Module):
    def __init__(self, feat_size, dropout):
        super(GlobalPool, self).__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * int(feat_size), 1),
            nn.LeakyReLU()
        )
        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(int(feat_size), int(feat_size))
        )
        self.gru = nn.GRUCell(int(feat_size), int(feat_size))

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        with g.local_scope():
            g.ndata['z'] = self.compute_logits(
                torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata['a'] = dgl.softmax_nodes(g, 'z')
            g.ndata['hv'] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, 'hv', 'a')
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata['a']
            else:
                return self.gru(context, g_feats)

class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP
    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    This class computes graph representations out of node features.
    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """
    def __init__(self, feat_size, num_timesteps=2, dropout=0.):
        super(AttentiveFPReadout, self).__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g, node_feats, get_node_weight=False):
        """Computes graph representations out of node features.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        get_node_weight : bool
            Whether to get the weights of nodes in readout. Default to False.
        Returns
        -------
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Graph representations computed. G for the number of graphs.
        node_weights : list of float32 tensor of shape (V, 1), optional
            This is returned when ``get_node_weight`` is ``True``.
            The list has a length ``num_timesteps`` and ``node_weights[i]``
            gives the node weights in the i-th update.
        """
        with g.local_scope():
            g.ndata['hv'] = node_feats
            g_feats = dgl.sum_nodes(g, 'hv')

        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, node_weights
        else:
            return g_feats


class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.edge_transform[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.
    This will be used in GNN layers for updating node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[1].reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.u_mul_e('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.
    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(int(node_feat_size), int(graph_feat_size)),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(int(node_feat_size) + int(edge_feat_size), int(graph_feat_size)),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * int(graph_feat_size), 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(int(graph_feat_size), int(graph_feat_size),
                                           int(graph_feat_size), dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node[0].reset_parameters()
        self.project_edge1[0].reset_parameters()
        self.project_edge2[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges1(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges
        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GNNLayer(nn.Module):
    """GNNLayer for updating node features.
    This layer performs message passing over node representations and update them.
    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * int(node_feat_size), 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(int(node_feat_size), int(graph_feat_size), dropout)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_edge[1].reset_parameters()
        self.attentive_gru.reset_parameters()

    def apply_edges(self, edges):
        """Edge feature generation.
        Generate edge features by concatenating the features of the destination
        and source nodes.
        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.
        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.attentive_gru(g, logits, node_feats)

class AttentiveFPGNN(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(AttentiveFPGNN, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def reset_parameters(self):
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
        return node_feats


#Make an additional class for readout
class WeightedSumAndMax(nn.Module): 
    r"""Apply weighted sum and max pooling to the node
    representations and concatenate the results.
    Parameters
    ----------
    in_feats : int
        Input node feature size
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__() 
        
        # use a generic readout function from DGL (https://lifesci.dgl.ai/_modules/dgllife/model/readout/weighted_sum_and_max.html#WeightedSumAndMax)
        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """Readout
        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g