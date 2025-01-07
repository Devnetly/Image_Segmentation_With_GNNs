import sys
import os
import torch
import numpy as np
import pandas as pd
import json
sys.path.append('../..')
from src.segmentation import Segmenter, SegmenterConfig, SegmentationType
from src.feature_extraction import FeatureExtractionConfig
from src.models import ActivationType
from src.dataset import ISICDataset
from src.utils import iou, dice, pixle_wise_accuracy, seed_everything
from dataclasses import dataclass, asdict
from typing import Optional, Literal
from definitions import ISIC_DIR
from tqdm.auto import tqdm
from argparse import ArgumentParser
from PIL import Image

@dataclass
class Config:

    ### Seed
    seed : int = 42

    ### Feature Extraction Configuration
    model_name : str = 'facebook/dino-vits8' # the model used to extract the features
    feature_type : Literal['cls','key','query','value'] = 'key' # the type of the feature to extract
    layer : Optional[int] = None # the encoder layer to extract the features from,None means the last layer
    stride : Optional[int] = None # the stride of the patches,None means the default stride
    resize : bool = True # whether to resize the image to the model's input size or keep the original size

    ### Model Configuration
    segmentation_type : SegmentationType = 'ncut' # the type of the segmentation
    threshold : float = 0.0 # In the case of DMON loss.
    alpha : Optional[float] = None # In the case of CC loss.
    activation : ActivationType = 'leaky_relu' # activation function
    num_layers : int = 1 # number of layers
    conv_type : Literal['gcn','gat','arma'] = 'gcn' # convolution type
    hidden_dim : int = 64 # hidden dimension
    num_clusters : int = 2 # number of clusters
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu' # device

    ### Training Configuration
    lr : float = 0.01
    n_iters : int = 10

    ### Data Configuration
    data_dir : str = ISIC_DIR # the path to the data
    output_dir : str = 'output' # the path to save the output


def evaluate(config: Config):

    seed_everything(config.seed)

    feature_extractor_config = FeatureExtractionConfig(
        model_name = config.model_name,
        feature_type = config.feature_type,
        layer = config.layer,
        stride = config.stride,
        resize = config.resize
    )

    deep_cut_config = SegmenterConfig(
        cut = config.segmentation_type,
        alpha = config.alpha,
        feature_extractor_config = feature_extractor_config,
        activation = config.activation,
        num_layers = config.num_layers,
        conv_type = config.conv_type,
        hidden_dim = config.hidden_dim,
        num_clusters = config.num_clusters,
        device = config.device
    )

    deep_cut = Segmenter(deep_cut_config)

    dataset = ISICDataset(root=config.data_dir,return_mask=True)

    metrics = {
        'filename': [],
        'iou': [],
        'dice': [],
        'pixle_wise_accuracy': [],
        'flipped': []
    }

    it = tqdm(dataset)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    for i,sample in enumerate(it):

        image = sample['image']
        target = np.array(sample['mask'])

        mask,_ = deep_cut.segment(image, config.lr, config.n_iters, show_progress=False)

        image = np.array(image).astype(np.uint8)

        iou_score = iou(mask, target)
        iou_score_inv = iou(1 - mask, target)

        metrics['flipped'].append(False)

        if iou_score_inv > iou_score:
            mask = 1 - mask
            iou_score = iou_score_inv
            metrics['flipped'][-1] = True

        Image.fromarray(mask).save(os.path.join(config.output_dir,dataset.files[i]+os.path.extsep+'png'))

        dice_score = dice(mask, target)
        pixle_wise_accuracy_score = pixle_wise_accuracy(mask, target)

        metrics['filename'].append(dataset.files[i])
        metrics['iou'].append(iou_score)
        metrics['dice'].append(dice_score)
        metrics['pixle_wise_accuracy'].append(pixle_wise_accuracy_score)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(config.output_dir,'metrics.csv'),index=False)

    with open(os.path.join(config.output_dir,'config.json'),'w') as f:
        json.dump(asdict(config),f)
    
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default=Config.model_name)
    parser.add_argument('--feature_type', type=str, default=Config.feature_type)
    parser.add_argument('--layer', type=int, default=Config.layer)
    parser.add_argument('--stride', type=int, default=Config.stride)
    parser.add_argument('--resize', type=bool, default=Config.resize)
    parser.add_argument('--cut', type=bool, default=Config.cut)
    parser.add_argument('--alpha', type=float, default=Config.alpha)
    parser.add_argument('--activation', type=str, default=Config.activation)
    parser.add_argument('--num_layers', type=int, default=Config.num_layers)
    parser.add_argument('--conv_type', type=str, default=Config.conv_type)
    parser.add_argument('--hidden_dim', type=int, default=Config.hidden_dim)
    parser.add_argument('--num_clusters', type=int, default=Config.num_clusters)
    parser.add_argument('--device', type=str, default=Config.device)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--n_iters', type=int, default=Config.n_iters)
    parser.add_argument('--data_dir', type=str, default=Config.data_dir)
    parser.add_argument('--output_dir', type=str, default=Config.output_dir)

    args = parser.parse_args()

    config = Config(**vars(args))

    evaluate(config)