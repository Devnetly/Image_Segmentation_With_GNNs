import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
# from src.models import DeepCut
from tqdm.auto import tqdm
from typing import Callable

def iou(pred : np.ndarray, target : np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union

def display_segmentation_results(
    sample : dict,
    mask : np.ndarray
) -> None:
    
    _, axes = plt.subplots(1, 4, figsize=(15, 5))
    IoU = iou(mask, np.array(sample['mask']))

    print(f"IoU: {IoU:.3f}")

    axes[0].imshow(sample['image'])
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(sample['mask'])
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(mask)
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    object_ = np.array(sample['image']) * mask.astype(np.uint8)[:, :, None]

    axes[3].imshow(object_)
    axes[3].set_title("Object")
    axes[3].axis('off')

"""def evaluate(
    dataset : Dataset,
    deep_cut : DeepCut,
    metrics : dict[str,Callable]
) -> None:
    
    values = {
        metric_name : 0.0 for metric_name in metrics.keys()
    }

    it = tqdm(dataset)
    
    for i,sample in enumerate(it):

        pred, _ = deep_cut.segment(sample['image'], lr=0.01, n_iters=20, show_progress=False)
        pred = pred.astype(np.uint8)

        target = np.array(sample['mask']).astype(np.uint8)

        locals = dict()

        for metric_name, metric in metrics.items():
            locals[metric_name] = metric(pred, target)
            values[metric_name] += locals[metric_name]

        it.set_postfix(**locals)

    for metric_name in metrics.keys():
        values[metric_name] /= len(dataset)
        print(f"{metric_name}: {values[metric_name]}")

    return values"""