"""
utils.py
Shared configuration and image-preprocessing helpers for the
Maize Leaf Disease Detection project.
"""

import numpy as np
from PIL import Image, ImageFilter

# ----------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------
IMG_SIZE = 224                       # VGG16 expects 224x224
CHANNELS = 3

# Folder names exactly as they appear in the Kaggle / GitHub dataset
CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Human-friendly labels + agronomic info shown in the UI
DISEASE_INFO = {
    "Blight": {
        "label": "Northern Leaf Blight",
        "pathogen": "Exserohilum turcicum (fungus)",
        "symptoms": "Long, cigar-shaped grey-green to tan lesions that run "
                    "parallel to the leaf veins and enlarge over time.",
        "action": "Use resistant hybrids, rotate crops, and apply foliar "
                  "fungicides at early tasseling if pressure is high.",
        "severity": "High",
    },
    "Common_Rust": {
        "label": "Common Rust",
        "pathogen": "Puccinia sorghi (fungus)",
        "symptoms": "Small, oval, cinnamon-brown pustules scattered on both "
                    "leaf surfaces that rupture and release rusty spores.",
        "action": "Plant resistant varieties; fungicides help when infection "
                  "appears before silking.",
        "severity": "Moderate",
    },
    "Gray_Leaf_Spot": {
        "label": "Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis (fungus)",
        "symptoms": "Narrow, rectangular grey-to-tan lesions bounded by the "
                    "leaf veins, giving a characteristic blocky look.",
        "action": "Improve airflow, rotate away from corn residue, and use "
                  "fungicides where the disease is established.",
        "severity": "High",
    },
    "Healthy": {
        "label": "Healthy Leaf",
        "pathogen": "None",
        "symptoms": "Uniform green colour with no lesions, pustules or "
                    "discolouration.",
        "action": "No treatment needed. Maintain good field hygiene and "
                  "balanced nutrition.",
        "severity": "None",
    },
}


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def load_image(file_or_path):
    """Open an image from a path or an uploaded file object as RGB."""
    img = Image.open(file_or_path).convert("RGB")
    return img


def resize_image(pil_img, size=IMG_SIZE):
    """Resize a PIL image to (size, size)."""
    return pil_img.resize((size, size))


def to_array(pil_img):
    """PIL image -> float32 numpy array with shape (H, W, 3)."""
    return np.asarray(pil_img, dtype=np.float32)


def preprocess_for_model(pil_img, model_type="cnn", size=IMG_SIZE):
    """
    Produce a model-ready batch of shape (1, size, size, 3).

    model_type:
        "cnn"  -> simple /255.0 scaling
        "vgg"  -> tf.keras VGG16 preprocess_input (mean-centering, BGR)
    """
    img = resize_image(pil_img, size)
    arr = to_array(img)

    if model_type == "vgg":
        # Imported lazily so the UI can run without TensorFlow installed
        from tensorflow.keras.applications.vgg16 import preprocess_input
        arr = preprocess_input(arr.copy())
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def preprocessing_stages(pil_img, size=IMG_SIZE):
    """
    Return a dict of intermediate images used to *visualise* the
    preprocessing pipeline in the web app.
    """
    resized = resize_image(pil_img, size)
    gray = resized.convert("L").convert("RGB")
    edges = resized.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")

    # Normalised view (scaled back to 0-255 just for display)
    norm = to_array(resized) / 255.0
    norm_display = Image.fromarray((norm * 255).astype("uint8"))

    return {
        "Original": pil_img,
        "Resized 224x224": resized,
        "Grayscale": gray,
        "Edge Map": edges,
        "Normalised": norm_display,
    }
