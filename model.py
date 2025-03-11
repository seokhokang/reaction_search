"""This module defines a neural network architecture using Graph Isomorphism Networks (GIN) for the purpose of learning representations for chemical reactions.
 The architecture includes multiple GIN layers that process molecular graphs to generate molecular embeddings.
 These embeddings are then projected using a series of linear layers to generate embeddings for products, reactants, and reagents

Classes:
- GINPredictor: A model that encapsulates the GIN architecture along with projection heads for generating embeddings of reaction components.
- GINLayer: Represents a single layer of the GIN, handling node and edge features within the graph.
- GIN: A sequential container of multiple GINLayer instances, forming the complete GIN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn import GINEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling


class GINPredictor(nn.Module):
    """Representation model based on a Graph Isomorphism Network (GIN) and multiple projection heads for reaction representations.
    """

    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_layers=5,
                 emb_dim=300,
                 dropout=0.1,
                 readout='sum',
                 hidden_feats=512,
                 project_dim=512):
        """Initializes an instance of the GINPredictor class.   
        
        Args:
            node_in_feats (int): Number of input features per node.
            edge_in_feats (int): Number of input features per edge.
            num_layers (int, optional): Number of GIN layers. Default is 5.
            emb_dim (int, optional): Dimensionality of the node embeddings. Default is 300.
            dropout (float, optional): Dropout rate. Default is 0.1.
            readout (str, optional): Readout function, either 'sum' or 'mean'. Default is 'sum'.
            hidden_feats (int, optional): Hidden dimension size for the projection head. Default is 512.
            project_dim (int, optional): Output dimension of the projection layer. Default is 512.
        """    
                 
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
        """Forward pass for GINPredictor.

        Args:
            g (dgl.DGLGraph): Input molecular graph.
            idx (int): Index indicating which projection head to use (0: product, 1: reactant, 2: reagent).

        Returns:
            torch.Tensor: Projected graph embeddings of shape (batch_size, project_dim).
        """
        
        node_feats = self.gnn(g, g.ndata['node_feats'], g.edata['edge_feats'])        
        graph_feats = self.readout(g, node_feats)
        
        output = self.projection[idx](graph_feats) #0: product, 1: reactant, 2: reagent
        output = output * (g.batch_num_nodes()>0).reshape(-1,1) #zero vector if missing

        return output
        
        
class GINLayer(nn.Module):
    """A single Graph Isomorphism Network (GIN) layer.
    """
    
    def __init__(self, emb_dim, batch_norm=True, activation=None):
        """Initializes an instance of the GINLayer class.   

        Args:
            emb_dim (int): Dimension of node embeddings.
            batch_norm (bool, optional): If True, applies batch normalization. Default is True.
            activation (callable, optional): Activation function applied after batch normalization. Default is None.
        """     
    
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
        """Resets parameters
        """
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Forward pass for GINLayer.

        Args:
            g (dgl.DGLGraph): Input molecular graph.
            node_feats (torch.Tensor): Node feature tensor of shape (num_nodes, emb_dim).
            edge_feats (torch.Tensor): Edge feature tensor of shape (num_edges, emb_dim).

        Returns:
            torch.Tensor: Updated node features after message passing.
        """

        node_feats = self.conv(g, node_feats, edge_feats)
        node_feats = self.mlp(node_feats)
        if self.bn is not None:
            node_feats = self.bn(node_feats)
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats


class GIN(nn.Module):

    """Graph Isomorphism Network (GIN), consisting of multiple GIN layers.
    """

    def __init__(self, node_in_feats, edge_in_feats,
                 num_layers=5, emb_dim=300, dropout=0.5):
        """Initializes an instance of the GIN class.   
    
        Args:
            node_in_feats (int): Number of input features per node.
            edge_in_feats (int): Number of input features per edge.
            num_layers (int, optional): Number of GIN layers. Default is 5.
            emb_dim (int, optional): Dimension of node embeddings. Default is 300.
            dropout (float, optional): Dropout rate. Default is 0.5.
        """         
     
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
        """Resets parameters
        """

        self.project_node_feats[0].reset_parameters()
        self.project_edge_feats[0].reset_parameters()
        self.project_edge_feats[-1].reset_parameters()

        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Forward pass for GIN.

        Args:
            g (dgl.DGLGraph): Input molecular graph.
            node_feats (torch.Tensor): Node feature tensor of shape (num_nodes, node_in_feats).
            edge_feats (torch.Tensor): Edge feature tensor of shape (num_edges, edge_in_feats).

        Returns:
            torch.Tensor: Node embeddings of shape (num_nodes, emb_dim).
        """
  
        node_embeds = self.project_node_feats(node_feats)
        edge_embeds = self.project_edge_feats(edge_feats)
        
        all_layer_node_feats = [node_embeds]
        for layer in range(self.num_layers):
            node_embeds = self.gnn_layers[layer](g, all_layer_node_feats[layer], edge_embeds)
            node_embeds = self.dropout(node_embeds)
            all_layer_node_feats.append(node_embeds)

        final_node_feats = all_layer_node_feats[-1]

        return final_node_feats