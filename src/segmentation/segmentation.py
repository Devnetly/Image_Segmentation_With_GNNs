import torch
import numpy as np
import sys
sys.path.append('../..')
from torch import optim, Tensor
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal
from src.models import GraphPool,ActivationType
from src.feature_extraction import FeatureExtractor,FeatureExtractionConfig
from src.utils import graph_to_mask, adjacency_to_edge_list, bilateral_solver_output
from typing import Optional,Literal

SegmentationType = Literal['ncut','cc','dmon']

@dataclass
class SegmenterConfig:
    segmentation_type : SegmentationType = 'ncut'
    alpha : float = 1.0
    threshold : float = 0.0
    feature_extractor_config : Optional[FeatureExtractionConfig] = field(default_factory=FeatureExtractionConfig)
    activation : ActivationType = 'leaky_relu'
    num_layers : int = 2
    conv_type : Literal['gcn','gat'] = 'gcn'
    hidden_dim : int = 64
    num_clusters : int = 2
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

class Segmenter:
    
    def __init__(self, config : SegmenterConfig) -> None:
        super().__init__()


        self.config = config
        self.feature_extractor = None

        if config.feature_extractor_config is not None:
            self.feature_extractor = FeatureExtractor(config.feature_extractor_config)

        normalize = config.segmentation_type != 'cc'

        self.graph_pool1 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.conf.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=self.config.num_clusters,
            normalize=normalize
        ).to(self.config.device) # Foreground/Background segmentation

        self.graph_pool2 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.conf.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=config.num_clusters,
            normalize=normalize
        ).to(self.config.device) # Foreground segmentation

        self.graph_pool3 = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.conf.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=2,
            normalize=normalize
        ).to(self.config.device) # Background segmentation

    def reset_parameters(self) -> None:
        self.graph_pool1.reset_parameters()
        self.graph_pool2.reset_parameters()
        self.graph_pool3.reset_parameters()
    
    def ncut_adjacency(self, features : Tensor) -> Tensor:
        
        W = features @ features.t()
        W = W * (W > 0.0).float()
        W = W / W.max()

        return W
    
    def cc_adjacency(self, features : Tensor) -> Tensor:

        W = features @ features.t()
        W = W - (W.max() / self.config.alpha)

        return W
    
    def cc_loss(self, A : Tensor, S : Tensor) -> Tensor:

        X = torch.matmul(S, S.t())
        cc_loss = -torch.sum(A * X)

        return cc_loss
    
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
    
    def create_adjacency(self, X : Tensor) -> Tensor:
        
        if self.config.segmentation_type == 'ncut':
            return self.ncut_adjacency(X)
        elif self.config.segmentation_type == 'cc':
            return self.cc_adjacency(X)
        elif self.config.segmentation_type == 'dmon':
            return self.dmon_adjacency(X)
        else:
            raise ValueError(f"Invalid segmentation type: {self.config.segmentation_type}")
    
    def loss(self, A : Tensor, S : Tensor) -> Tensor:

        if self.config.segmentation_type == 'ncut':
            return self.ncut_loss(A, S)
        elif self.config.segmentation_type == 'cc':
            return self.cc_loss(A, S)
        elif self.config.segmentation_type == 'dmon':
            return self.dmon_loss(A, S)
        else:
            raise ValueError(f"Invalid segmentation type: {self.config.segmentation_type}")
    
    def fit(self, 
        model : GraphPool, 
        X : Tensor, 
        A : Tensor, 
        lr : float, 
        n_iters : int,
        show_progress : bool = True
    ) -> Tensor:

        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        it = tqdm(range(n_iters), desc='Optimizing') if show_progress else range(n_iters)

        edge_index, edge_weight = adjacency_to_edge_list(A)

        for _ in it:

            optimizer.zero_grad()

            S = model.forward(X, edge_index, edge_weight)
            loss = self.loss(A, S)
            loss.backward()
            optimizer.step()

            if show_progress:
                it.set_postfix(loss=loss.item())

        return S
    
    def segment(self, 
        image : Image.Image | Tensor,
        lr : float,
        n_iters : int,
        show_progress : bool = True
    ) -> tuple[np.ndarray,np.ndarray]:
        
        self.reset_parameters()

        processed_image = self.feature_extractor.process(image)
        X = self.feature_extractor.extract(processed_image).squeeze(0)
        A = self.create_adjacency(X)

        X = X.to(self.config.device).clone()  
        A = A.to(self.config.device)

        S = self.fit(self.graph_pool1, X, A, lr, n_iters, show_progress)

        mask = graph_to_mask(
            S=S.argmax(dim=-1),
            processed_size=processed_image.shape[2:],
            og_size=image.size,
            cc=False, 
            stride=self.config.feature_extractor_config.stride
        )

        S = S.detach().cpu().numpy()

        _,mask = bilateral_solver_output(
            np.array(image),
            mask,
            36,
            6,
            6
        )

        mask = mask.astype(np.uint8) * 255

        return mask, S

        X_f = X[S.argmax(dim=-1) == 1]
        A_f = self.create_adjacency(X_f)

        S_f = self.fit(self.graph_pool2, X_f, A_f, lr, n_iters)

        X_b = X[S.argmax(dim=-1) == 0]
        A_b = self.create_adjacency(X_b)

        S_b = self.fit(self.graph_pool3, X_b, A_b, lr, n_iters)