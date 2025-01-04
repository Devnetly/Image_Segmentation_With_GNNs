import torch
import numpy as np
from torch import optim, Tensor
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal
from src.models.graph_pool import GraphPool,ActivationType
from src.feature_extraction import FeatureExtractor,FeatureExtractionConfig
from src.utils import graph_to_mask, adjacency_to_edge_list, bilateral_solver_output

@dataclass
class DeepCutConfig:
    cut : bool = True
    alpha : Optional[float] = None
    feature_extractor_config : FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    activation : ActivationType = 'leaky_relu'
    num_layers : int = 2
    conv_type : Literal['gcn','gat'] = 'gcn'
    hidden_dim : int = 64
    num_clusters : int = 2
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeepCut:
    
    def __init__(self, config : DeepCutConfig) -> None:
        super().__init__()

        self.config = config
        self.feature_extractor = FeatureExtractor(config.feature_extractor_config)

        self.graph_pool1 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=2
        ).to(self.config.device) # Foreground/Background segmentation

        self.graph_pool2 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=config.num_clusters
        ).to(self.config.device) # Foreground segmentation

        self.graph_pool3 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=2
        ).to(self.config.device) # Background segmentation

    def reset_parameters(self) -> None:
        self.graph_pool1.reset_parameters()
        self.graph_pool2.reset_parameters()
        self.graph_pool3.reset_parameters()

    def create_adjacency(self, features : Tensor) -> Tensor:
        
        W = features @ features.t()

        if self.config.cut:
            W = W * (W > 0)
            W = W / W.max()
        else:
            W = W - (W.max() / self.config.alpha)

        return W
    
    def ncut_loss(self, A : Tensor, S : Tensor) -> Tensor:
        
        ### Normalized Cut Loss
        a = torch.matmul(torch.matmul(A, S).t(), S)
        a = torch.trace(a)

        D = torch.diag(torch.sum(A, dim=-1))
        b = torch.matmul(torch.matmul(D, S).t(), S)
        b = torch.trace(b)

        ncut_loss = -(a / b)

        ### Orthogonality Loss
        St_S = torch.matmul(S.t(), S)
        St_S = St_S / torch.norm(St_S)

        I = torch.eye(S.size(1), device=S.device, dtype=S.dtype) 
        I = I / torch.norm(I)

        orth_loss = torch.norm(St_S - I)

        ### Loss
        loss = ncut_loss + orth_loss

        return loss
    
    def fit(self, model : GraphPool, X : Tensor, A : Tensor, lr : float, n_iters : int) -> Tensor:

        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        it = tqdm(range(n_iters), desc='Optimizing')

        edge_index, edge_weight = adjacency_to_edge_list(A)

        for _ in it:

            optimizer.zero_grad()

            S = model.forward(X, edge_index, edge_weight)
            loss = self.ncut_loss(A, S)
            loss.backward()
            optimizer.step()

            it.set_postfix(loss=loss.item())

        return S
    
    def segment(self, 
        image : Image.Image,
        lr : float,
        n_iters : int,
    ) -> tuple[Image.Image,Image.Image,Image.Image]:
        
        self.reset_parameters()

        processed_image = self.feature_extractor.process(image)
        X = self.feature_extractor.extract(processed_image).squeeze(0)
        A = self.create_adjacency(X)

        X = X.to(self.config.device).clone()  
        A = A.to(self.config.device)

        S = self.fit(self.graph_pool1, X, A, lr, n_iters)

        mask = graph_to_mask(
            S=S.argmax(dim=-1),
            processed_size=processed_image.shape[2:],
            og_size=image.size,
            cc=False, 
            stride=self.config.feature_extractor_config.stride
        )

        S = S.detach().cpu().numpy()

        mask = bilateral_solver_output(
            np.array(image),
            mask,
            36,
            6,
            6
        )[1]

        return mask, S

        X_f = X[S.argmax(dim=-1) == 1]
        A_f = self.create_adjacency(X_f)

        S_f = self.fit(self.graph_pool2, X_f, A_f, lr, n_iters)

        X_b = X[S.argmax(dim=-1) == 0]
        A_b = self.create_adjacency(X_b)

        S_b = self.fit(self.graph_pool3, X_b, A_b, lr, n_iters)