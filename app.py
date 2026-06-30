"""
app.py
Maize Leaf Disease Detection - interactive Streamlit web app.

Features
    * Upload a PNG / JPEG leaf image and classify it with a CNN or VGG16 model
    * Raw image  ->  preprocessing pipeline  ->  prediction + disease card
    * Dataset explorer (raw sample images per class)
    * Model performance: accuracy curves, ROC plot, confusion matrix,
      classification report
    * Professional purple theme, Cambria typography

Run locally:
    streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import (CLASS_NAMES, DISEASE_INFO, IMG_SIZE, load_image,
                   preprocess_for_model, preprocessing_stages)

MODEL_DIR = "models"
ART_DIR = "artifacts"
DATA_DIR = "Dataset"

# ----------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------
st.set_page_config(page_title="Maize Leaf Disease Detection",
                   page_icon="🌽", layout="wide",
                   initial_sidebar_state="expanded")

PURPLE_DARK = "#4A148C"
PURPLE = "#6A1B9A"
PURPLE_MID = "#7B1FA2"
PURPLE_LIGHT = "#9C27B0"
PURPLE_ACCENT = "#BA68C8"
LAVENDER = "#F3E5F5"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gelasio:wght@400;500;600;700&display=swap');

/* Cambria first, graceful serif fallbacks (Gelasio mirrors Cambria) */
html, body, [class*="css"], .stMarkdown, .stApp, p, div, span, label,
h1, h2, h3, h4, h5, h6, button, input, textarea, .stDataFrame {{
    font-family: Cambria, 'Gelasio', Georgia, 'Times New Roman', serif !important;
}}

.stApp {{
    background: linear-gradient(180deg, #FBF7FE 0%, {LAVENDER} 100%);
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {PURPLE_DARK} 0%, {PURPLE_MID} 100%);
}}
section[data-testid="stSidebar"] * {{ color: #F4E9FB !important; }}

/* Headings */
h1, h2, h3 {{ color: {PURPLE_DARK} !important; font-weight: 700 !important; }}

/* Hero banner */
.hero {{
    background: linear-gradient(120deg, {PURPLE_DARK}, {PURPLE_LIGHT});
    padding: 1.6rem 2rem; border-radius: 16px; color: #fff;
    box-shadow: 0 8px 24px rgba(74,20,140,0.28); margin-bottom: 1.2rem;
}}
.hero h1 {{ color: #fff !important; margin: 0; font-size: 2.1rem; }}
.hero p  {{ color: #EEDcFb; margin: .3rem 0 0; font-size: 1.05rem; }}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
    background: {PURPLE}; color: #fff; border: none; border-radius: 10px;
    padding: .55rem 1.4rem; font-weight: 600;
}}
.stButton>button:hover {{ background: {PURPLE_DARK}; color:#fff; }}

/* Cards */
.card {{
    background: #fff; border-radius: 14px; padding: 1.2rem 1.4rem;
    border: 1px solid #E7D4F2; box-shadow: 0 4px 14px rgba(123,31,162,0.10);
}}
.result-pill {{
    display:inline-block; background:{PURPLE_LIGHT}; color:#fff;
    padding:.35rem 1rem; border-radius:999px; font-weight:700; font-size:1.25rem;
}}
.metric-big {{ font-size:2.4rem; font-weight:700; color:{PURPLE_DARK}; }}
.sev-High   {{ color:#B71C1C; font-weight:700; }}
.sev-Moderate{{ color:#E65100; font-weight:700; }}
.sev-None   {{ color:#1B5E20; font-weight:700; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background:{LAVENDER}; border-radius:10px 10px 0 0; padding:8px 18px;
    color:{PURPLE_DARK};
}}
.stTabs [aria-selected="true"] {{ background:{PURPLE}; color:#fff !important; }}
</style>
""", unsafe_allow_html=True)

# Matplotlib purple palette
PURPLES = LinearSegmentedColormap.from_list(
    "purples", ["#F3E5F5", "#CE93D8", "#9C27B0", "#6A1B9A", "#4A148C"])
ROC_COLORS = [PURPLE_DARK, PURPLE_LIGHT, PURPLE_ACCENT, "#E1BEE7"]
plt.rcParams.update({"font.family": "serif", "axes.edgecolor": PURPLE_MID,
                     "axes.labelcolor": PURPLE_DARK, "text.color": PURPLE_DARK,
                     "xtick.color": PURPLE_DARK, "ytick.color": PURPLE_DARK})


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_type):
    path = os.path.join(MODEL_DIR, f"{model_type}_model.h5")
    if not os.path.exists(path):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)


def load_json(name):
    path = os.path.join(ART_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def sample_image_for(class_name):
    """Return path to a representative leaf image for a class, if available."""
    folder = os.path.join(DATA_DIR, class_name)
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(folder, f)
    return None


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌽 Maize Disease AI")
    page = st.radio("Navigate",
                    ["🔍 Detect Disease", "🖼️ Dataset Explorer",
                     "📊 Model Performance", "ℹ️ About"])
    st.markdown("---")
    model_type = st.selectbox("Model", ["vgg", "cnn"],
                              format_func=lambda m: "VGG16 (Transfer Learning)"
                              if m == "vgg" else "Custom CNN")
    st.caption("CNN = built from scratch.  VGG16 = ImageNet transfer learning.")


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🌽 Maize Leaf Disease Detection</h1>
  <p>Deep-learning classification with CNN &amp; VGG16 — upload a leaf image
  to identify Blight, Common Rust, Gray Leaf Spot or a Healthy leaf.</p>
</div>
""", unsafe_allow_html=True)


# ======================================================================
# PAGE 1 — DETECT
# ======================================================================
if page == "🔍 Detect Disease":
    st.subheader("Upload a leaf image")
    file = st.file_uploader("PNG or JPEG", type=["png", "jpg", "jpeg"])

    if file is None:
        st.info("Upload a maize-leaf photo (.png / .jpg / .jpeg) to begin.")
    else:
        pil = load_image(file)

        st.markdown("#### 1 · Raw image")
        st.image(pil, width=320, caption="Original upload")

        st.markdown("#### 2 · Preprocessing pipeline")
        stages = preprocessing_stages(pil)
        cols = st.columns(len(stages))
        for col, (name, img) in zip(cols, stages.items()):
            col.image(img, caption=name, use_container_width=True)

        st.markdown("#### 3 · Disease classification")
        model = load_model(model_type)
        if model is None:
            st.warning(f"No trained **{model_type.upper()}** model found in "
                       f"`./models/`. Run `python train.py` first. "
                       "Showing the workflow with a demo prediction.")
            probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
        else:
            batch = preprocess_for_model(pil, model_type)
            probs = model.predict(batch, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_cls = CLASS_NAMES[pred_idx]
        info = DISEASE_INFO[pred_cls]
        conf = float(probs[pred_idx]) * 100

        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f'<div class="card"><span class="result-pill">'
                f'{info["label"]}</span>'
                f'<p class="metric-big">{conf:.1f}%<span style="font-size:1rem">'
                f' confidence</span></p>'
                f'<p><b>Pathogen:</b> {info["pathogen"]}<br>'
                f'<b>Severity:</b> <span class="sev-{info["severity"]}">'
                f'{info["severity"]}</span></p>'
                f'<p><b>Symptoms:</b> {info["symptoms"]}</p>'
                f'<p><b>Recommended action:</b> {info["action"]}</p></div>',
                unsafe_allow_html=True)
        with right:
            ref = sample_image_for(pred_cls)
            if ref:
                st.image(ref, caption=f"Reference: {info['label']}",
                         use_container_width=True)
            else:
                st.info("Add the dataset to `./Dataset/` to show a reference "
                        "leaf image for the predicted class.")

        st.markdown("##### Class probabilities")
        prob_df = pd.DataFrame({
            "Disease": [DISEASE_INFO[c]["label"] for c in CLASS_NAMES],
            "Probability": probs})
        st.bar_chart(prob_df.set_index("Disease"), color=PURPLE_LIGHT)


# ======================================================================
# PAGE 2 — DATASET EXPLORER (raw images)
# ======================================================================
elif page == "🖼️ Dataset Explorer":
    st.subheader("Raw dataset samples")
    if not os.path.isdir(DATA_DIR):
        st.warning("Dataset not found. Download it and unzip into `./Dataset/` "
                   "with one sub-folder per class. See the README for links.")
    else:
        for cls in CLASS_NAMES:
            st.markdown(f"### {DISEASE_INFO[cls]['label']}")
            folder = os.path.join(DATA_DIR, cls)
            imgs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))][:5] \
                if os.path.isdir(folder) else []
            if imgs:
                cols = st.columns(len(imgs))
                for col, p in zip(cols, imgs):
                    col.image(p, use_container_width=True)
            else:
                st.caption("No images found for this class.")


# ======================================================================
# PAGE 3 — MODEL PERFORMANCE
# ======================================================================
elif page == "📊 Model Performance":
    st.subheader(f"Performance — "
                 f"{'VGG16' if model_type=='vgg' else 'Custom CNN'}")

    metrics = load_json(f"{model_type}_metrics.json")
    history = load_json(f"{model_type}_history.json")
    report = load_json(f"{model_type}_report.json")
    cm = load_json(f"{model_type}_confusion.json")
    roc = load_json(f"{model_type}_roc.json")

    if not any([metrics, history, report, cm, roc]):
        st.warning("No evaluation artifacts found. Run `python train.py` then "
                   "`python evaluate.py` to generate accuracy, ROC, confusion "
                   "matrix and classification report.")
    else:
        # ---- top-line accuracy ---------------------------------------
        if metrics:
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="card"><p class="metric-big">'
                        f'{metrics["accuracy"]*100:.2f}%</p>'
                        f'<b>Validation accuracy</b></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="card"><p class="metric-big">'
                        f'{metrics["n_samples"]}</p>'
                        f'<b>Validation images</b></div>',
                        unsafe_allow_html=True)

        tabs = st.tabs(["Accuracy / Loss", "ROC Curve",
                        "Confusion Matrix", "Classification Report"])

        # ---- training curves -----------------------------------------
        with tabs[0]:
            if history:
                fig, ax = plt.subplots(1, 2, figsize=(11, 4))
                ax[0].plot(history.get("accuracy", []), color=PURPLE_DARK,
                           label="train")
                ax[0].plot(history.get("val_accuracy", []), color=PURPLE_ACCENT,
                           label="val")
                ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
                ax[1].plot(history.get("loss", []), color=PURPLE_DARK, label="train")
                ax[1].plot(history.get("val_loss", []), color=PURPLE_ACCENT,
                           label="val")
                ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
                fig.tight_layout(); st.pyplot(fig)
            else:
                st.info("Training history not available.")

        # ---- ROC -----------------------------------------------------
        with tabs[1]:
            if roc:
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                for (name, d), color in zip(roc.items(), ROC_COLORS):
                    ax.plot(d["fpr"], d["tpr"], color=color, lw=2,
                            label=f'{DISEASE_INFO[name]["label"]} '
                                  f'(AUC={d["auc"]:.3f})')
                ax.plot([0, 1], [0, 1], "--", color="#BDBDBD")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC — one-vs-rest")
                ax.legend(loc="lower right", fontsize=9)
                st.pyplot(fig)
            else:
                st.info("ROC data not available.")

        # ---- confusion matrix ----------------------------------------
        with tabs[2]:
            if cm:
                mat = np.array(cm["matrix"])
                labels = [DISEASE_INFO[c]["label"] for c in cm["labels"]]
                fig, ax = plt.subplots(figsize=(6.5, 5.5))
                im = ax.imshow(mat, cmap=PURPLES)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                thresh = mat.max() / 2 if mat.max() else 0
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                                color="white" if mat[i, j] > thresh else PURPLE_DARK,
                                fontweight="bold")
                fig.colorbar(im, fraction=0.046, pad=0.04)
                fig.tight_layout(); st.pyplot(fig)
            else:
                st.info("Confusion matrix not available.")

        # ---- classification report -----------------------------------
        with tabs[3]:
            if report:
                rows = []
                for cls in CLASS_NAMES:
                    if cls in report:
                        r = report[cls]
                        rows.append({"Class": DISEASE_INFO[cls]["label"],
                                     "Precision": round(r["precision"], 3),
                                     "Recall": round(r["recall"], 3),
                                     "F1-score": round(r["f1-score"], 3),
                                     "Support": int(r["support"])})
                df = pd.DataFrame(rows)
                st.dataframe(df.style.background_gradient(
                    cmap="Purples", subset=["Precision", "Recall", "F1-score"]),
                    use_container_width=True, hide_index=True)
                if "macro avg" in report:
                    m = report["macro avg"]
                    st.caption(f"Macro avg — precision {m['precision']:.3f}, "
                               f"recall {m['recall']:.3f}, f1 {m['f1-score']:.3f}")
            else:
                st.info("Classification report not available.")


# ======================================================================
# PAGE 4 — ABOUT
# ======================================================================
else:
    st.subheader("About this project")
    st.markdown("""
This application classifies **maize (corn) leaf diseases** from a photo using
two deep-learning approaches:

* **Custom CNN** — a convolutional network trained from scratch.
* **VGG16** — transfer learning from ImageNet with a custom classification head.

**Pipeline:** raw image → resize to 224×224 → model-specific normalisation →
softmax over four classes → disease card with agronomic guidance.

**Dataset** (≈4,000 images, 4 classes):
[Kaggle — Corn/Maize Leaf Disease](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
· [GitHub mirror](https://github.com/R4j4n/Maize-Diseases-Detection).
""")
    st.markdown("### The four classes")
    for cls in CLASS_NAMES:
        info = DISEASE_INFO[cls]
        st.markdown(
            f'<div class="card" style="margin-bottom:.7rem">'
            f'<b style="color:{PURPLE_DARK};font-size:1.1rem">{info["label"]}</b>'
            f' &nbsp;<span class="sev-{info["severity"]}">'
            f'({info["severity"]} severity)</span><br>'
            f'<b>Pathogen:</b> {info["pathogen"]}<br>{info["symptoms"]}</div>',
            unsafe_allow_html=True)
