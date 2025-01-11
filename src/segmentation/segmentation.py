import torch
import numpy as np
import sys
import cv2
sys.path.append('../..')
from torch import optim, Tensor
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal
from src.models import GraphPool,ActivationType
from src.feature_extraction import FeatureExtractor,FeatureExtractionConfig
from src.utils import graph_to_mask, adjacency_to_edge_list, bilateral_solver_output, apply_crf
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

        self.graph_pool = GraphPool(
            activation=config.activation,
            num_layers=config.num_layers,
            conv_type=config.conv_type,
            in_features=self.feature_extractor.conf.dim,
            hidden_dim=config.hidden_dim,
            num_clusters=self.config.num_clusters,
            normalize=normalize
        ).to(self.config.device) # Foreground/Background segmentation

    def reset_parameters(self) -> None:
        self.graph_pool.reset_parameters()
    
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
        
    def dmon_adjacency(self, features : Tensor) -> Tensor:
        
        features = features / torch.norm(features, dim=-1, p=2, keepdim=True)
        W = features @ features.t()
        W = W >= self.config.threshold
        W = W.float()

        return W
    
    def dmon_loss(self, A : Tensor, S : Tensor) -> Tensor:
        
        d = torch.sum(A, dim=-1)
        m = torch.sum(A)

        B = A - torch.ger(d, d) / (2 * m)

        n,k = S.size()
        k = torch.tensor(k, dtype=S.dtype, device=S.device)

        modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(S.t(), B), S))
        collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(S, dim=0), p='fro')) - 1

        loss = modularity_term + collapse_reg_term

        return loss
    
    def fit(self, 
        model : GraphPool, 
        X : Tensor, 
        A : Tensor, 
        lr : float, 
        n_iters : int,
        record_interval : Optional[int] = None,
        show_progress : bool = True
    ) -> list[Tensor]:

        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=lr)

        it = tqdm(range(n_iters), desc='Optimizing') if show_progress else range(n_iters)

        edge_index, edge_weight = adjacency_to_edge_list(A)

        results = []

        for i in it:

            optimizer.zero_grad()

            S = model.forward(X, edge_index, edge_weight)
            loss = self.loss(A, S)
            loss.backward()
            optimizer.step()

            if record_interval is not None and (i + 1) % record_interval == 0:
                results.append(S.detach().cpu().clone())

            if show_progress:
                it.set_postfix(loss=loss.item())

        if record_interval is None:
            results.append(S.detach().cpu().clone())

        return results
    
    def segment(self, 
        image : Image.Image,
        ignore : Optional[Tensor] = None,
        record_interval : Optional[int] = None,
        lr : float = 0.001,
        n_iters : int = 30,
        postprocess : bool = True,
        show_progress : bool = True
    ) -> dict | list[dict]:
        
        self.reset_parameters()

        processed_image = self.feature_extractor.process(image)
        X = self.feature_extractor.extract(processed_image).squeeze(0)

        N = X.size(0)

        if ignore is not None:
            X = X[ignore.to(X.device),:]

        A = self.create_adjacency(X)

        X = X.to(self.config.device).clone()  
        A = A.to(self.config.device)

        results = self.fit(self.graph_pool, X, A, lr, n_iters, record_interval, show_progress)
        masks = []

        for S in results:

            S = S.argmax(dim=-1)

            if ignore is not None:

                C = torch.zeros(
                    size=(N,),
                    device=S.device,
                    dtype=S.dtype
                )

                C[ignore.to(X.device)] = S
                
                S = C

            mask = graph_to_mask(
                S=S,
                processed_size=processed_image.shape[2:],
                og_size=image.size,
                cc=False, 
                stride=self.config.feature_extractor_config.stride
            )

            if postprocess:

                output_solver1,mask1 = bilateral_solver_output(
                    np.array(image),
                    mask,
                    48,
                    8,
                    8
                )

                output_solver2,mask2 = bilateral_solver_output(
                    np.array(image),
                    1 - mask,
                    48,
                    8,
                    8
                )

                a = (S == 0).sum()
                b = (S == 1).sum()
                r = max(a,b) / min(a,b)

                a1 = (mask1 == 0).sum()
                b1 = (mask1 == 1).sum()
                r1 = max(a1,b1) / min(a1,b1)

                a2 = (mask2 == 0).sum()
                b2 = (mask2 == 1).sum()
                r2 = max(a2,b2) / min(a2,b2)

                d1 = abs(r - r1)
                d2 = abs(r - r2)

                mask = mask1 if d1 < d2 else mask2
                mask = mask.astype(np.uint8) * 255
                output_solver = output_solver1 if d1 < d2 else output_solver2

            masks.append({
                'mask' : mask,
                'S' : S.detach().cpu().numpy(),
                'A' : A.detach().cpu().numpy(),
                'output_solver' : output_solver
            })

        if record_interval is None:
            return masks[0]
        
        return masks