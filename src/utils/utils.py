import cv2
import torch
import numpy as np
import pydensecrf.densecrf as dcrf
from torch import Tensor
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian

def graph_to_mask(
    S : Tensor,
    processed_size : tuple[int,int],
    og_size : tuple[int,int],
    cc : bool,
    stride : int,
) -> np.ndarray:
    
    H, W = processed_size

    minus = 1 if stride == 4 else 0

    S = S.reshape(H // stride - minus,W // stride - minus,-1)
    S = S.squeeze(-1)

    """if S[0,0] + S[0,-1] + S[-1,0] + S[-1,-1] > 2:
        S = 1 - S"""

    S_numpy = S.cpu().detach().numpy()

    if cc:
        S_numpy = largest_cc(S_numpy)

    mask = cv2.resize(S_numpy.astype(float), og_size, interpolation=cv2.INTER_NEAREST)
    mask = np.array(mask)

    return mask

def largest_cc(S : np.ndarray) -> np.ndarray:
    
    us_cc = cv2.connectedComponentsWithStats(S.astype(np.uint8), connectivity=4)

    us_cc_stats = us_cc[2]

    cc_idc = np.argsort(us_cc_stats[:,-1])[::-1]

    if np.percentile(S[us_cc[1] == cc_idc[0]], 99) == 0:
        mask : np.ndarray = np.equal(us_cc[1], cc_idc[1])
    elif np.percentile(S[us_cc[1] == cc_idc[1]], 99) == 0:
        mask : np.ndarray = np.equal(us_cc[1], cc_idc[0])
    else:
        raise NotImplementedError('No valid decision rule for cropping')
    
    return mask


def adjacency_to_edge_list(W : Tensor) -> tuple[Tensor,Tensor]:
    
    edge_index = W.nonzero(as_tuple=False).t().contiguous()
    edge_weight = W[edge_index[0], edge_index[1]]

    return edge_index, edge_weight

def seed_everything(seed : int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def apply_crf(image, mask_prob):
    h, w = image.shape[:2]
    
    # Transpose mask probabilities
    mask_prob = mask_prob.transpose(2, 0, 1)  # Shape: (n_labels, h, w)
    
    # Convert to unary energy
    unary = unary_from_softmax(mask_prob)  # Shape: (n_labels, h * w)
    unary = np.ascontiguousarray(unary)
    
    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(w, h, mask_prob.shape[0])  # n_labels inferred from mask_prob
    d.setUnaryEnergy(unary)
    
    # Add pairwise energy (optional)
    pairwise = create_pairwise_gaussian(sdims=(3, 3), shape=(h, w))
    d.addPairwiseEnergy(pairwise, compat=3)
    
    # Run inference
    refined = d.inference(10)  # Number of iterations
    return np.argmax(refined, axis=0).reshape((h, w))  # Return refined mask