import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import GINEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling


class GINPredictor(nn.Module):

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 readout='sum',#'mean',
                 hidden_feats=512,
                 project_dim=512):
        super(GINPredictor, self).__init__()

        self.gnn = GIN(node_in_feats=node_in_feats, 
                       edge_in_feats=edge_in_feats,
                       num_layers=num_layers,
                       emb_dim=emb_dim,
                       dropout=dropout)

        if readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'mean':
            self.readout = AvgPooling()

        self.projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, hidden_feats), nn.ReLU(),
                nn.Linear(hidden_feats, project_dim, bias=False)
            ) for _ in range(2)
        ]) # 0: product, 1: reactant, 2: reagent

    def forward(self, g, idx):

        node_feats = self.gnn(g, g.ndata['node_feats'], g.edata['edge_feats'])        
        graph_feats = self.readout(g, node_feats)
        
        output = self.projection[idx](graph_feats) #0: product, 1: reactant, 2: reagent
        output = output * (g.batch_num_nodes()>0).reshape(-1,1) #zero vector if missing

        return output
        
        
class GINLayer(nn.Module):

    def __init__(self, emb_dim, batch_norm=True, activation=None):
        super(GINLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )  
        self.conv = GINEConv()
        
        if batch_norm:
            self.bn = nn.BatchNorm1d(emb_dim)
        else:
            self.bn = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):

        node_feats = self.conv(g, node_feats, edge_feats)
        node_feats = self.mlp(node_feats)
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats


class GIN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 num_layers=5, emb_dim=300, dropout=0.5):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, emb_dim),
            nn.ReLU()
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        self.gnn_layers = nn.ModuleList()
        for layer in range(num_layers):
            if layer == num_layers - 1:
                self.gnn_layers.append(GINLayer(emb_dim))
            else:
                self.gnn_layers.append(GINLayer(emb_dim, activation=F.relu))

        self.reset_parameters()

    def reset_parameters(self):

        self.project_node_feats[0].reset_parameters()
        self.project_edge_feats[0].reset_parameters()
        self.project_edge_feats[-1].reset_parameters()

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):

        node_embeds = self.project_node_feats(node_feats)
        edge_embeds = self.project_edge_feats(edge_feats)
        
        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_embeds = self.gnn_layers[layer](g, all_layer_node_feats[layer], edge_embeds)
            node_embeds = self.dropout(node_embeds)
            all_layer_node_feats.append(node_embeds)

        final_node_feats = all_layer_node_feats[-1]

        return final_node_feats