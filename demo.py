import sys
import gradio as gr
import numpy as np
sys.path.append('..')
from src.feature_extraction import FeatureExtractionConfig
from src.segmentation import Segmenter,SegmenterConfig
from src.utils import seed_everything, iou
from PIL import Image

# Set seed for reproducibility
SEED = 42
seed_everything(SEED)

# Define the configuration for the Segmenter

config = SegmenterConfig(
    segmentation_type="ncut",
    alpha=5.0,
    feature_extractor_config=FeatureExtractionConfig(
        model_name="facebook/dino-vits8",
        device="cuda",
        feature_type="key",
    ),
    activation="leaky_relu",
    num_layers=1,
    conv_type="arma",
    hidden_dim=32,
    num_clusters=2,
    device="cuda",
    threshold=0.3
)

# Initialize the Segmenter
segmenter = Segmenter(config)

def segment_and_display(image: np.ndarray, mask: np.ndarray):

    """Run segmentation and display results."""
    image = Image.fromarray(image)
    pred,_ = segmenter.segment(image, lr=0.01, n_iters=20, show_progress=False)
    
    # Compute IoU (intersection over union)
    IoU = iou(mask, pred)  # Replace `iou` with your actual IoU calculation function.
    
    # Prepare images for display
    object_ = np.array(image) * pred.astype(bool).astype(np.uint8)[:, :, None]
    
    return {
        "IoU": f"IoU: {IoU:.3f}",
        "Image": image,
        "Ground Truth": mask,
        "Prediction": pred,
        "Object": object_
    }

def gradio_interface(image: np.ndarray, ground_truth: np.ndarray):
    """Interface function for Gradio."""
    results = segment_and_display(image, ground_truth)
    
    return results["IoU"], results["Image"], results["Ground Truth"], results["Prediction"], results["Object"]


interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Image(type="numpy", label="Ground Truth Mask", image_mode="L")
    ],
    outputs=[
        gr.Textbox(label="IoU"),
        gr.Image(label="Input Image"),
        gr.Image(label="Ground Truth Mask"),
        gr.Image(label="Predicted Mask"),
        gr.Image(label="Segmented Object")
    ],
    title="Unsupervised Segmentation Demo",
    description="Upload an image and ground truth mask to see segmentation results using a GNN-based unsupervised model."
)

if __name__ == "__main__":
    interface.launch()
