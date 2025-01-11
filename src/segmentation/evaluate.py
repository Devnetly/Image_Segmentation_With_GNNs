import sys
import os
import torch
import numpy as np
import pandas as pd
import json
import time
sys.path.append('../..')
from src.segmentation import Segmenter, SegmenterConfig, SegmentationType
from src.feature_extraction import FeatureExtractionConfig
from src.models import ActivationType
from src.dataset import EMD6Dataset,ISICDataset
from src.utils import iou, dice, seed_everything
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union
from definitions import EMDS_6_DIR, OUTPUTS_DIR, ISIC_DIR
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
    threshold : float = 0.7 # In the case of DMON loss.
    alpha : Optional[float] = 5.0 # In the case of CC loss.
    activation : ActivationType = 'silu' # activation function
    num_layers : int = 1 # number of layers
    conv_type : Literal['gcn','gat','arma'] = 'arma' # convolution type
    hidden_dim : int = 32 # hidden dimension
    num_clusters : int = 2 # number of clusters
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu' # device

    ### Training Configuration
    lr : float = 0.001
    n_iters : int = 50

    ### Data Configuration
    dataset : str = 'isic2016' # the path to the data
    output_dir : str = 'output' # the path to save the output

def process(image : Image.Image):

    SIZE = 224
    h, w = image.size
    r = h / w

    new_h = min(SIZE, h)
    new_w = int(new_h / r)

    image = image.resize(size=(new_h, new_w))

    return image

def create_dataset(dataset : str) -> Union[EMD6Dataset,ISICDataset]:

    if dataset == 'emd6':
        return EMD6Dataset(root=EMDS_6_DIR)
    elif dataset == 'isic2016':
        return ISICDataset(root=ISIC_DIR, return_mask=True, img_transform=process, mask_transform=process)
    else:
        raise ValueError('Invalid dataset')

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
        segmentation_type = config.segmentation_type,
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

    dataset = create_dataset(config.dataset)

    metrics = {
        'filename': [],
        'iou': [],
        'dice': [],
        'time' : [],
        'epoch' : []
    }

    it = tqdm(dataset)

    os.makedirs(os.path.join(OUTPUTS_DIR, config.output_dir, config.dataset),exist_ok=True)

    RECORD_INTERVAL = 10

    for i,sample in enumerate(it):

        tic = time.time()

        image = sample['image']
        target = np.array(sample['mask'])

        results = deep_cut.segment(image, lr=config.lr, n_iters=config.n_iters, show_progress=False, record_interval=RECORD_INTERVAL)

        toc = time.time()

        duration = toc - tic

        for j,result in enumerate(results):

            mask = result['mask']

            image = np.array(image).astype(np.uint8)

            iou_score = iou(mask, target)
            iou_score_inv = iou(mask.max() - mask, target)

            if iou_score_inv > iou_score:
                mask = mask.max() - mask
                iou_score = iou_score_inv

            if j == len(results) - 1:
                Image.fromarray(mask).save(os.path.join(OUTPUTS_DIR, config.output_dir, config.dataset, f'{dataset.get_filename(i)}.png'))

            dice_score = dice(mask, target)

            metrics['filename'].append(dataset.files[i])
            metrics['iou'].append(iou_score)
            metrics['dice'].append(dice_score)
            metrics['time'].append(duration)
            metrics['epoch'].append(j*RECORD_INTERVAL)

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(os.path.join(OUTPUTS_DIR,config.output_dir,'metrics.csv'),index=False)

    with open(os.path.join(OUTPUTS_DIR, config.output_dir,'config.json'),'w') as f:
        json.dump(asdict(config),f)
    
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, default=Config.model_name)
    parser.add_argument('--feature_type', type=str, default=Config.feature_type)
    parser.add_argument('--layer', type=int, default=Config.layer)
    parser.add_argument('--stride', type=int, default=Config.stride)
    parser.add_argument('--resize', type=bool, default=Config.resize)
    parser.add_argument('--segmentation_type', type=str, default=Config.segmentation_type)
    parser.add_argument('--threshold', type=float, default=Config.threshold)
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