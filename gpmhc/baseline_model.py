import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastai.data.all import DataLoader
from gpmhc.data import dataset, tokenize
from gpmhc.gnn_parts import *
from gpmhc.learner import get_learner
from collections import OrderedDict
from abc import ABC, abstractmethod


NUM_WORKERS = 8

def loss_fn(y_pred, y_true):
    #y_pred shape will be (batch_size,augment_dimension (usually 1), 1 during train 32 during test)
    #During testing the chosen binding core (per allele) is sent with the predictions for each allele, first 16 are the allele EL pred
    if y_pred.shape[-1]==32:
        y_pred = y_pred[:,:16]
    #During test predictions from all alleles are sent to the model, choose the best allele
    y_pred, max_args = y_pred.max(dim=1)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    loss = criterion(y_pred, y_true.float())
    return loss.mean()

#used for generating the model
class ModelRequirements(ABC):
    @abstractmethod
    def get_model(self):
        pass

class model(ModelRequirements):
    def __init__(self, json_input, *args, **kwargs):
        #Define the loss function for this model, using above function
        self.loss_func = loss_fn
        self.json_input = json_input

        #Create the tokenizer config, which assigns used sequences and padding
        allele_seq_cols = ['mhc_dr1_1', 'mhc_dr1_2', 'mhc_dr3_1', 'mhc_dr3_2', 'mhc_dr4_1', 'mhc_dr4_2', 'mhc_dr5_1', 'mhc_dr5_2', 'mhc_dp1_1', 'mhc_dp1_2', 'mhc_dp1_3', 'mhc_dp1_4', 'mhc_dq1_1', 'mhc_dq1_2', 'mhc_dq1_3', 'mhc_dq1_4']
        self.tokenizer = OrderedDict()
        self.tokenizer['nFlank'] = {'size':5, 'side':'left', 'token':'*'}
        self.tokenizer['peptide'] = {'size':30+self.json_input['model_hyper_opts']['bc_pad']*2, 'side':'right', 'token':'*'}
        self.tokenizer['cFlank'] = {'size':5, 'side':'right', 'token':'*'}
        for col in allele_seq_cols:
            self.tokenizer[col] = {'size':self.json_input['dataloader_options']['csv_to_df']['schema_options'][6], 'side':'right', 'token':'*'}

        #The adjacency matrices for the various graphs are looked up from a table during training
        self.lookup_table = nnalign_generate_combination_lookup_table(mhc_adj = self.json_input['mhc_adj'],mhc_lens = self.json_input['mhc_lens'],bc_pad=self.json_input['model_hyper_opts']['bc_pad'])


    #This instantiates the model with the hyperparameters given
    def get_model(self, model_hyperparameters):
        model_hyperparameters = self.json_input['model_hyper_opts']
        gnn = GNN(
            tokenizer=self.tokenizer,
            lookup_table=self.lookup_table,
            bc_pad = int(model_hyperparameters['bc_pad']),
            readout = str(model_hyperparameters['readout']),
            posenc = bool(model_hyperparameters['posenc']),
            node_feat_size=int(model_hyperparameters["node_feat_size"]),
            graph_feat_size=int(model_hyperparameters["graph_feat_size"]),
            edge_feat_size=int(model_hyperparameters["edge_feat_size"]),
            gnn_layers=int(model_hyperparameters["gnn_layers"]),
            timesteps=int(model_hyperparameters["timesteps"]),
            gnn_dropout=float(model_hyperparameters["gnn_dropout"]),
            rnn_dropout=float(model_hyperparameters["rnn_dropout"]),
            classifier_dropout=float(model_hyperparameters["classifier_dropout"]),
            
        )
        model = gnn
        model.arch = self
        return model
       


    #Dataloader for inference
    def get_dataloaders_test(self, df_test):
        self.df_test = df_test
        batch_size = self.json_input["model_hyper_opts"]['batch_size']

        x = tokenize(df_test, self.tokenizer)
        y = df_test['EL'].values.astype(int)

        if x.min() < 0:
            print('Negative Value in model input, must be problem with tokenizer, perhapas illegal character in sequence?')
            raise ValueError

        test_ds = dataset(x,y)
        test_dl = DataLoader(dataset=test_ds, bs=batch_size, num_workers=NUM_WORKERS,drop_last=False, shuffle=False)
        return test_dl

    #LR scheculing for fastai
    @staticmethod
    def get_lr_schedule(lr):
        return lr

    #instantiate the learner object for fastai
    def get_learner(self, model_dir=None):
        model = self.get_model(self.json_input)
        learner = get_learner(model)
        return learner



class GNN(nn.Module):
    def __init__(
        self,
        tokenizer,
        lookup_table,
        bc_pad = 0,
        readout = 'recurrent',
        posenc=True,
        node_feat_size=256,
        graph_feat_size=256,
        edge_feat_size=3,
        gnn_layers=3,
        gnn_dropout=0.1,
        timesteps=2,
        rnn_dropout=0.2,
        classifier_dropout=0.4,


        ):
        super(GNN, self).__init__()
        vocab_size = 30
        padding_idx = 0
        #embeddings map tokens to vectors, embeddings_2 isn't used
        self.embeddings = nn.Embedding(vocab_size, node_feat_size, padding_idx=padding_idx)
        self.embeddings_2 = nn.Embedding(vocab_size, node_feat_size, padding_idx=padding_idx)
        #This is the GAT GNN layers cratd by attentivefp
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=gnn_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=gnn_dropout)
        #This is the attentive GRU created by attentivefp
        if readout == 'recurrent':
            self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=timesteps,
                                          dropout=rnn_dropout)
            self.predict = nn.Sequential(nn.Dropout(classifier_dropout),
                                          nn.Linear(int(graph_feat_size), 1))
        
        #traditional weightd node readout
        elif readout == 'weighted_node':   
            self.readout = WeightedSumAndMax(graph_feat_size)
            self.predict = nn.Sequential(nn.Dropout(classifier_dropout),
                                          nn.Linear(int(2*graph_feat_size), 1))


        #simple positional encoder assigns learned vector for each location
        self.posenc = PositionalEncoderSimple(model_dim=node_feat_size ,dropout=0.2)
        #used to break the long input into sub sequences
        self.seq_lengths = np.array([v['size'] for k,v in tokenizer.items()])
        self.seq_stop_position = np.cumsum(self.seq_lengths)
        #graph lookup table
        self.lookup_table = lookup_table
        #for longer than 9mer binding core
        self.bc_pad = bc_pad
        self.use_posenc = posenc

    
    def forward(self, x_data):
        #x_data will have each row as a long series of padded tokens with all the sequences concatenated
        #Mask out the padding
        mask = x_data == 0
        #Split up the input into the different parts
        nflank = x_data[:, :self.seq_stop_position[0]]
        peptide = x_data[:, self.seq_stop_position[0]:self.seq_stop_position[1]]
        cflank = x_data[:, self.seq_stop_position[1]:self.seq_stop_position[2]]
        mhc = [x_data[:, self.seq_stop_position[ix + 2]:self.seq_stop_position[ix + 3]] for ix in range(16)]
        #The MHC has an _ between alpha and beta chain, get rid of that
        #Split out the padding
        nflank_mask = mask[:, :self.seq_stop_position[0]]
        peptide_mask = mask[:, self.seq_stop_position[0]:self.seq_stop_position[1]]
        cflank_mask = mask[:, self.seq_stop_position[1]:self.seq_stop_position[2]]
        mhc_mask = [mask[:, self.seq_stop_position[ix + 2]:self.seq_stop_position[ix + 3]] for ix in range(16)]

        #Get the peptide, n/c flank lengths because they are used to select the graph for each datapoint
        lens = [torch.sum(~nflank_mask, dim=1), torch.sum(~peptide_mask, dim=1), torch.sum(~cflank_mask, dim=1)]
        #Repeat this over the 16 alleles
        lens_full = [torch.cat([l]*16, dim=0) for l in lens]
        #The GNN takes the node features as one long sequence. We'll have one peptide-allele pair for each allele for a given peptide
        feat_full = torch.cat([torch.cat((nflank, peptide, cflank, m), dim=1) for m in mhc], dim=0)
        mask_full = ~torch.cat([torch.cat((nflank_mask, peptide_mask, cflank_mask, m), dim=1) for m in mhc_mask], dim=0)
        feat_full_token = feat_full

        #different allele genes get different adjacency matrices, create token for tracking
        alleles_full = []
        for i in range(16):
            if i<8:
                alleles_full.append(torch.tensor([0]*len(peptide)))
            if 7<i<12:
                alleles_full.append(torch.tensor([1]*len(peptide)))
            if 11<i<16:
                alleles_full.append(torch.tensor([2]*len(peptide)))
        alleles_full = torch.cat(alleles_full, dim=0).to(feat_full.device)

        #embed and positional encode input features
        feat_full = self.embeddings(feat_full.long())
        if self.use_posenc==True:
            feat_full = self.posenc(feat_full)
        feat_full = feat_full*mask_full.unsqueeze(2)
        #batch size
        length = len(peptide)
        #We don't want to run empty alleles through the model, so we reduce it to just those that aren't empty here
        allele_presence = torch.cat(mhc, dim=0)[:,0].bool()
        presence_index = torch.arange(len(allele_presence),device=allele_presence.device)[allele_presence]
        #Get rid of the node features and lengths for the empty peptide-allele pairs
        feat = feat_full[presence_index]
        feat_token = feat_full_token[presence_index]
        lens = [l[presence_index] for l in lens_full]
        alleles = alleles_full[presence_index]
        #Lookup the graphs for each peptide-allele pair. 
        # Each of these will have multiple graphs that are valid, we enumerate over all of them
        # to find the best one. So the node feats will be replicated, index keeps track of the 
        # original peptide-allele pair that the original came from
        g, node_feats, edge_feats, index = lookup_graph(feat, feat_token, lens, alleles, self.lookup_table,self.bc_pad)
        #we're gonna run all the graph-peptide-allele pairs through the model, but only do backprop through the best one,
        #So we turn of gradient accumulation while running all of them to reduce noise/save time
        with torch.no_grad():
            #embed the residue tokens
            #cast everything to cuda
            edge_feats = edge_feats.cuda()
            node_feats = node_feats.cuda()
            device = node_feats.get_device()
            g = g.to(device)
            #update node embeddings with GNN part of the model
            node_feats = self.gnn(g, node_feats, edge_feats)
            #get a graph embedding from the RNN part of the model
            g_feats = self.readout(g, node_feats)
            #get presentation likelihoods
            preds = self.predict(g_feats).view(-1)  
            #Figure out which graph was the best for each peptide-allele pair
            graph_idx = torch.stack([torch.argmax(preds[index == idx]) for idx in torch.unique(index)])
            #reduce predictions to just the best graph
            preds = torch.stack([torch.max(preds[index == idx]) for idx in torch.unique(index)])
            device = preds.get_device()
            #Create empty tensors full of predictions for all alleles, since many didn't get 
            # predictions because the alleles were empty, fill those with low presentation value
            preds_full = torch.tensor([-1000]*16*length, dtype=torch.float, device=device)
            graph_idx_full = torch.tensor([-1]*16*length, dtype=torch.long, device=device)
            #enter the predictions from the alleles that exist
            preds_full[presence_index] = preds
            graph_idx_full[presence_index] = graph_idx
            preds_full = preds_full.view(16, -1)
            graph_idx_full = graph_idx_full.view(16, -1)
            #determine which allele was the best for each peptide
            mhc_idx = torch.argmax(preds_full, dim=0)

            if not self.training:
                preds = torch.cat((preds_full, graph_idx_full), dim=0)
                outputs = torch.transpose(preds, 1, 0)
                outputs = outputs.reshape(-1, outputs.shape[1])
                return outputs

        #This is only used in training, which is not provided in this codebase
                
        #Choose just the best peptide/allele pairs from the original token tensor
        feat = feat_full.view(16,-1, feat_full.shape[-2],feat_full.shape[-1])
        feat_token = feat_full_token.view(16,-1, feat_full_token.shape[-1])
        feat = torch.stack([feat[m,i] for i, m in enumerate(mhc_idx)])
        feat_token = torch.stack([feat_token[m,i] for i, m in enumerate(mhc_idx)])
        lens = [l.view(16,-1) for l in lens_full]
        lens = [l[0] for l in lens]
        graph_idx = torch.transpose(graph_idx_full, 1,0)
        #get the graphs from just the best alleles
        graph_idx = torch.stack([g[idx] for g, idx in zip(graph_idx, mhc_idx)])

        alleles = alleles_full.view(16,-1)
        alleles = torch.stack([alleles[m,i] for i, m in enumerate(mhc_idx)])

        #lookup the best graph for each peptide
        g, node_feats, edge_feats = lookup_single_graph(feat, feat_token, lens, alleles, self.lookup_table, graph_idx.cpu())
        node_feats = node_feats.cuda()
        #Run the model on just the best graphs/peptide-allele pairs, with gradient accumulation
        edge_feats = edge_feats.cuda()

        device = node_feats.get_device()
        g = g.to(device)

        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        output = self.predict(g_feats).view(-1)
        return output