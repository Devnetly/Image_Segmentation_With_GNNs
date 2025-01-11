from torch import nn, Tensor
from torch_geometric.nn import GAT, GCN, ARMAConv
from torch.nn import functional as F
from typing import Literal
from .activation import Activation,ActivationType

class GraphPool(nn.Module):
    
    def __init__(self, 
        activation : ActivationType = 'leaky_relu',
        num_layers : int = 2,
        conv_type : Literal['gcn','gat','arma'] = 'gcn',
        in_features : int = 384,
        hidden_dim : int = 64,
        num_clusters : int = 2,
        normalize : bool = True         
    ) -> None:
        super().__init__()

        self.activation = activation
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.normalize = normalize

        self.convs = None

        if self.conv_type == 'gcn':

            self.convs = GCN(
                in_channels=self.in_features,
                hidden_channels=self.hidden_dim,
                normalize=self.normalize,
                num_layers=self.num_layers,
                act=Activation(self.activation)
            )

        elif self.conv_type == 'gat':

            self.convs = GAT(
                in_channels=self.in_features,
                hidden_channels=self.hidden_dim,
                num_layers=self.num_layers,
                act=Activation(self.activation),
                heads=4
            )

        elif self.conv_type == 'arma':

            self.convs = ARMAConv(
                in_channels=self.in_features,
                out_channels=self.hidden_dim,
                num_layers=self.num_layers,
                act=Activation(self.activation),
                num_stacks=2, 
                dropout=0.4,
                shared_weights=False
            )

        else:

            raise ValueError(f'Invalid convolution type {self.conv_type}')
                        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(0.25),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.num_clusters)
        )

    def reset_parameters(self) -> None:

        self.convs.reset_parameters()

        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, X : Tensor, edge_index : Tensor, edge_weight : Tensor) -> Tensor:
        
        X = self.convs(X, edge_index, edge_weight)
        X = self.mlp(X)
        X = F.softmax(X, dim=-1)

        return X