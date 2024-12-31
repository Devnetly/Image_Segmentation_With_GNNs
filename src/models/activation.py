from torch import nn, Tensor
from typing import Literal

ActivationType = Literal['relu','leaky_relu','silu','celu','gelu','selu']

class Activation(nn.Module):

    def __init__(self, name : ActivationType) -> None:
        super().__init__()

        self.name = name

        self.activation = nn.ModuleDict({
            'relu' : nn.ReLU(),
            'leaky_relu' : nn.LeakyReLU(),
            'silu' : nn.SiLU(),
            'celu' : nn.CELU(),
            'gelu' : nn.GELU(),
            'selu' : nn.SELU(),
        })

        if self.name not in self.activation:
            raise ValueError(f'{self.name} is not a valid activation function, choose from {list(self.activation.keys())}')

    def forward(self, x : Tensor) -> Tensor:
        activation_fn =  self.activation[self.name]
        return activation_fn(x)