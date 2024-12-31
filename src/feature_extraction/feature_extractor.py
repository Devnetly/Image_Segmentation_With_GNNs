import torch
from torch import Tensor
from PIL import Image
from torch.nn import functional as F
from dataclasses import dataclass,asdict
from typing import Optional,Union,Literal
from transformers import ViTImageProcessor, ViTModel

@dataclass
class FeatureExtractionConfig:
    
    model_name : str = 'facebook/dino-vits16' # the model used to extract the features
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu' # device
    feature_type : Literal['cls','key','query','value'] = 'cls' # the type of the feature to extract
    layer : Optional[int] = None # the encoder layer to extract the features from,None means the last layer
    stride : Optional[int] = None # the stride of the patches,None means the default stride
    resize : bool = True # whether to resize the image to the model's input size or keep the original size

    def __post_init__(self) -> None:
        self.stride = self.stride or int(''.join([c for c in self.model_name if c.isnumeric()]))


class FeatureExtractor:

    __dims__ = {
        "facebook/dino-vits16" : 384,
        "facebook/dino-vits8"  : 384
    }

    def __init__(self, config : FeatureExtractionConfig) -> None:
        """
            Initializes the FeatureExtractor class

            Args :
            - config : FeatureExtractionConfig : the configuration of the feature extraction

            Returns :
            - None
        """

        self.config = config
        self.dim = self.__dims__[config.model_name]
        self.model : ViTModel = ViTModel.from_pretrained(config.model_name).to(config.device).eval()
        self.processor = ViTImageProcessor.from_pretrained(config.model_name)
        self.model.config
    
    def pad(self, images : Tensor) -> Tensor:
        """
            Pads the images if the resize is set to False
            and the height and/or width of the image is not a 
            multiple of the patch size

            Args :
            - images : Tensor : the images to pad

            Returns :
            - Tensor : the padded images

        """

        _,_,H,W = images.shape
        P = self.model.config.patch_size
        N_H,N_W = H // P, W // P
        pad_H,pad_W = H - N_H * P, W - N_W * P

        images = F.pad(images, (0, pad_W, 0, pad_H), mode='constant', value=0)

        return images

    def process(self, images : Union[Image.Image, list[Image.Image]], resize : Optional[bool] = None) -> Tensor:
        """
            Process the images to be ready for feature extraction,it applies the
            loaded processor to the images and pads them if the height and/or width
            of the image is not a multiple of the patch size

            Args :
            - images : Union[Image.Image, list[Image.Image]] : the images to process

            Returns :
            - Tensor : the processed images
        """

        resize = resize if resize is not None else self.config.resize

        processed_images = self.processor(images = images, return_tensors = 'pt', do_resize = resize)
        processed_images = processed_images['pixel_values']
        processed_images = self.pad(processed_images)

        return processed_images

    def extract(self, images : Tensor, **kwargs : dict) -> Tensor:
        """
            Extract the features from the images

            Args :
            - images : Tensor : the images to extract the features from
            - **kwargs : dict : the additional arguments to override the default configuration

            Returns :
            - Tensor : the extracted features
        """

        ### Modify the configuration
        if 'model_name' in kwargs:
            raise ValueError("You can't change the model name")

        local_config = FeatureExtractionConfig(**{
            **asdict(self.config),
            **kwargs
        })

        ### Move data to the right device
        images = images.to(local_config.device)
        B,C,H,W = images.shape

        with torch.inference_mode():

            ### Patch Embeddings
            S = local_config.stride or self.model.config.patch_size
            P = self.model.config.patch_size

            # The convolution operation is used to extract the patch embeddings
            # F.conv2d is used to allow us to modify the stride of the convolution operation
            # (B,C,H,W) -> (B,C',H',W')
            features = F.conv2d(
                images,
                weight=self.model.embeddings.patch_embeddings.projection.weight,
                bias=self.model.embeddings.patch_embeddings.projection.bias,
                stride=S,
                padding=self.model.embeddings.patch_embeddings.projection.padding,
                dilation=self.model.embeddings.patch_embeddings.projection.dilation,
                groups=self.model.embeddings.patch_embeddings.projection.groups
            )

            features = features.flatten(2) # (B,C',H',W') -> (B,C',N)
            features = features.transpose(1, 2) # (B,C',N) -> (B,N,C')

            ### Add CLS token
            cls_tokens = self.model.embeddings.cls_token.expand(B, -1, -1)
            features = torch.cat((cls_tokens, features), dim=1)
            
            # if the images is not resized to the image size
            # that model expects,then we need to interpolate (resize)
            # the positional embeddings accordingly
            if local_config.resize and S == P:
                features = features + self.model.embeddings.position_embeddings
            else:
                
                ### if the stride is modified
                ### we can't use the height and width of the image
                ### directly to interpolate the positional embeddings
                H_new = int((P // S) * (1 - (P - S) / H) * H)
                W_new = int((P // S) * (1 - (P - S) / W) * W)

                features = features + self.model.embeddings.interpolate_pos_encoding(features, H_new, W_new)

            ### Encoder
            layer_id = local_config.layer or len(self.model.encoder.layer) - 1
            H = self.model.config.num_attention_heads

            for i,layer in enumerate(self.model.encoder.layer):
                
                if i != layer_id:

                    features = layer(features)
                    features = features[0]

                else:

                    if local_config.feature_type in ['query','key','value']:

                        layers = {
                            'key' : layer.attention.attention.key,
                            'query' : layer.attention.attention.query,
                            'value' : layer.attention.attention.value
                        }

                        features = layers[local_config.feature_type](features)
                        features = features[:,1:,:]

                        B,N,C = features.shape

                        features = torch.reshape(features, shape=(B, N, H, C // H))
                        features = torch.permute(features, dims=(0, 1, 2, 3))

                    else:

                        features = layer(features)
                        features = features[0]
                        features = features[:,0,:]

                    break
            
            if local_config.feature_type != 'cls':
                features = features.flatten(2)

        return features
    
    def extract_from_pil(self, images : Union[Image.Image, list[Image.Image]], **kwargs : dict) -> Tensor:
        """
            Extract the features from the PIL images,it applies the neccessary
            processing to the images before extracting the features

            Args :
            - images : Union[Image.Image, list[Image.Image]] : the images to extract the features from

            Returns :
            - Tensor : the extracted features
        """

        inputs = self.process(images = images, resize=kwargs.get('resize'))
        return self.extract(inputs, **kwargs)