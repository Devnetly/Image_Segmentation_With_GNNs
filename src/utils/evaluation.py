import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def iou(pred : np.ndarray, target : np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / union

def dice(pred : np.ndarray, target : np.ndarray) -> float:
    IoU = iou(pred, target)
    return 2 * IoU / (IoU + 1)

def pixle_wise_accuracy(pred : np.ndarray, target : np.ndarray) -> float:
    return accuracy_score(target.flatten(), pred.flatten())

def pixel_wise_precision(pred : np.ndarray, target : np.ndarray) -> float:
    return precision_score(target.flatten(), pred.flatten())

def pixel_wise_recall(pred : np.ndarray, target : np.ndarray) -> float:
    return recall_score(target.flatten(), pred.flatten())

def pixel_wise_f1_score(pred : np.ndarray, target : np.ndarray) -> float:
    return f1_score(target.flatten(), pred.flatten())

def compute_metrics(pred : np.ndarray,target : np.ndarray) -> dict:
    
    return {
        "IoU": iou(pred, target),
        "Dice": dice(pred, target),
        "Pixel-wise Accuracy": pixle_wise_accuracy(pred, target),
        "Pixel-wise Precision": pixel_wise_precision(pred, target),
        "Pixel-wise Recall": pixel_wise_recall(pred, target),
        "Pixel-wise F1 Score": pixel_wise_f1_score(pred, target)
    }

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