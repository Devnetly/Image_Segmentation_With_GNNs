import sys
import os
import torch
import numpy as np
sys.path.append('../..')
from src.feature_extraction import FeatureExtractionConfig, FeatureExtractor
from src.dataset import ISICDataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional,Literal
from definitions import ISIC_DIR
from argparse import ArgumentParser
from toolz import compose, partial

@dataclass
class Config:
    model_name : str = 'facebook/dino-vits8' # the model used to extract the features
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu' # device
    feature_type : Literal['cls','key','query','value'] = 'key' # the type of the feature to extract
    layer : Optional[int] = None # the encoder layer to extract the features from,None means the last layer
    stride : Optional[int] = None # the stride of the patches,None means the default stride
    resize : bool = True # whether to resize the image to the model's input size or keep the original size
    data_path : str = ISIC_DIR # the path to the data
    batch_size : int = 8 # the batch size
    num_workers : int = 4 # the number of workers
    prefetch_factor : int = 2 # the prefetch factor

def main(config: Config):
    
    feature_extractor_config = FeatureExtractionConfig(
        model_name = config.model_name,
        device = config.device,
        feature_type = config.feature_type,
        layer = config.layer,
        stride = config.stride,
        resize = config.resize
    )

    feature_extractor = FeatureExtractor(feature_extractor_config)

    dataset = ISICDataset(
        root=config.data_path, 
        img_transform=compose(
            partial(torch.squeeze, dim=0), # remove the batch dimension
            feature_extractor.process, # process the image
        ),
        return_mask=False
    )

    dataloader = DataLoader(dataset=dataset, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor, 
        shuffle=False
    )

    i = 0

    for batch in tqdm(dataloader):

        images = batch['image']
        features = feature_extractor.extract(images)
        features = features.cpu().detach().numpy()

        for feature in features:
            filename = dataset.files[i]
            feature_path = os.path.join(Config.data_path, f"{filename}_features.npy")
            np.save(feature_path, feature)
            i += 1
            
if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--model-name', type=str, default=Config.model_name)
    parser.add_argument('--device', type=str, default=Config.device)
    parser.add_argument('--feature-type', type=str, default=Config.feature_type)
    parser.add_argument('--layer', type=int, default=Config.layer)
    parser.add_argument('--stride', type=int, default=Config.stride)
    parser.add_argument('--resize', type=bool, default=Config.resize)
    parser.add_argument('--data-path', type=str, default=Config.data_path)
    parser.add_argument('--batch-size', type=int, default=Config.batch_size)
    parser.add_argument('--num-workers', type=int, default=Config.num_workers)
    parser.add_argument('--prefetch-factor', type=int, default=Config.prefetch_factor)

    args = parser.parse_args()
    config = Config(**vars(args))

    main(config)