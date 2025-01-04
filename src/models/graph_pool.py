from torch import nn, Tensor
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F
from typing import Literal
from .activation import Activation,ActivationType

class GraphPool(nn.Module):
    
    def __init__(self, 
        activation : ActivationType = 'leaky_relu',
        num_layers : int = 2,
        conv_type : Literal['gcn','gat'] = 'gcn',
        in_features : int = 384,
        hidden_dim : int = 64,
        num_clusters : int = 2             
    ) -> None:
        super().__init__()

        self.activation = activation
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters

        self.activation_fn = Activation(self.activation)

        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):

            features_in = self.in_features if i == 0 else self.hidden_dim
            features_out = self.hidden_dim
            
            if self.conv_type == 'gcn':
                self.layers.append(GCNConv(features_in, features_out))
            elif self.conv_type == 'gat':
                self.layers.append(GATConv(features_in, features_out))
                        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.25),
            Activation(self.activation),
            nn.Linear(self.hidden_dim, self.num_clusters)
        )

    def reset_parameters(self) -> None:

        for layer in self.layers:
            layer.reset_parameters()

        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, X : Tensor, edge_index : Tensor, edge_weight : Tensor) -> Tensor:
        
        for _,layer in enumerate(self.layers):
            X = layer(X,edge_index,edge_weight)
            X = self.activation_fn(X)
        
        X = self.mlp(X)
        X = F.softmax(X, dim=-1)

        return X