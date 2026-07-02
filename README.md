# 🌽 Maize Leaf Disease Detection — CNN & VGG16

An interactive **Streamlit** web app that classifies maize (corn) leaf diseases
from an uploaded photo using two deep-learning models: a **custom CNN** and
**VGG16 transfer learning**. It shows the raw image, the preprocessing pipeline,
the predicted disease with a reference leaf image, and full evaluation
visualisations — accuracy curves, **ROC plot**, **confusion matrix** and
**classification report** — in a professional purple theme with Cambria
typography.

**Classes:** `Blight` (Northern Leaf Blight) · `Common_Rust` ·
`Gray_Leaf_Spot` · `Healthy`

---

## 1. Project structure

```
maize-disease-detection/
├── app.py                 # Streamlit web app (purple theme + Cambria)
├── train.py               # Train custom CNN and VGG16
├── evaluate.py            # ROC, confusion matrix, classification report
├── utils.py               # Config + preprocessing helpers
├── requirements.txt
├── Dockerfile             # For Cloud Run
├── .dockerignore
├── .streamlit/config.toml # Base theme
├── models/                # Saved models (created by train.py)
├── artifacts/             # Eval JSON read by the app (created by evaluate.py)
└── Dataset/               # You download this (one folder per class)
```

---

## 2. Get the dataset

Download from either source and unzip so the folders look like
`Dataset/<ClassName>/*.jpg`:

- Kaggle: <https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset>
- GitHub: <https://github.com/R4j4n/Maize-Diseases-Detection>

```bash
# Kaggle CLI option
pip install kaggle
kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset
unzip corn-or-maize-leaf-disease-dataset.zip -d Dataset
```

> Make sure the four sub-folders are named exactly
> `Blight`, `Common_Rust`, `Gray_Leaf_Spot`, `Healthy`
> (rename if the download uses spaces). This matches `CLASS_NAMES` in `utils.py`.

---

## 3. Set up and run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py --model both --epochs 20    # trains CNN + VGG16
python evaluate.py --model both             # generates ROC / CM / report
streamlit run app.py                        # open http://localhost:8501
```

The app runs even **before** training — it shows the full workflow with a demo
prediction and tells you what to train. After `train.py` + `evaluate.py`, the
**Model Performance** page fills in with real metrics.

> **Font note:** Cambria is a Microsoft font and may be absent on Linux servers.
> The CSS falls back to *Gelasio* (a metric-compatible Google font) then Georgia,
> so the app looks consistent everywhere while using Cambria where installed.

---

## 4. Deploy free on Google Cloud (Cloud Run)

**Cloud Run** has a perpetual free tier (2M requests, 360k GB-seconds of memory
and 180k vCPU-seconds per month) — ideal for a demo app. You need a Google
account and the dataset is **not** shipped (only `models/` + `artifacts/`), so
the image stays small.

### 4.1 One-time setup
1. Create / select a project at <https://console.cloud.google.com>.
   Note the **Project ID**.
2. Enable billing (required even for free tier — you won't be charged within
   the limits) and enable the **Cloud Run** and **Cloud Build** APIs.
3. Install the **gcloud CLI**: <https://cloud.google.com/sdk/docs/install>, then:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 4.2 Make sure trained artifacts are present
Cloud Run can't train the model, so commit your trained files first:

```bash
ls models/        # cnn_model.h5, vgg_model.h5
ls artifacts/     # *_history.json, *_roc.json, *_confusion.json, *_report.json, *_metrics.json
```

> Optionally drop a few sample images into `Dataset/<class>/` so the predicted
> "reference leaf" and Dataset Explorer work. Keep it tiny (the `.dockerignore`
> excludes `Dataset/` from the build by default — remove that line if you want
> to include a small sample set).

### 4.3 Deploy (single command — builds + deploys)

```bash
gcloud run deploy maize-disease \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --port 8080
```

- `--source .` lets Cloud Build build the `Dockerfile` for you (no local Docker needed).
- TensorFlow needs memory — **2Gi** avoids out-of-memory crashes on cold start.
- `--allow-unauthenticated` makes it a public URL.

When it finishes, gcloud prints a **Service URL** like
`https://maize-disease-xxxxx-el.a.run.app` — that's your live app.

### 4.4 Stay within the free tier
- Set **max instances** to cap usage: add `--max-instances 1`.
- Set **min instances to 0** (default) so it scales to zero when idle (no cost).
- Cold starts take a few seconds while TensorFlow loads — that's expected.

### 4.5 Update later
Re-run the same `gcloud run deploy` command; it builds a new revision.

### Alternative free options
- **Google App Engine (Standard)** also has a free tier, but the 2 GB memory
  cap and TensorFlow size make Cloud Run the smoother choice.
- For a zero-config public demo, **Streamlit Community Cloud**
  (<https://share.streamlit.io>) deploys straight from a GitHub repo for free,
  though it isn't part of Google Cloud.

---

## 5. How it works

| Stage | CNN | VGG16 |
|------|-----|-------|
| Input | 224×224×3 | 224×224×3 |
| Normalisation | `/255.0` | `vgg16.preprocess_input` |
| Backbone | 4 Conv blocks (32→128) | Frozen ImageNet conv base |
| Head | GAP → Dense(128) → softmax(4) | GAP → Dense(256) → softmax(4) |
| Training | from scratch | transfer learning (lr 1e-4) |

Augmentation (flip, rotation, zoom) is applied to the training split only.
Evaluation uses a fixed 20% validation split with `shuffle=False` so the
confusion matrix and ROC curves are reproducible.

---

## 6. Troubleshooting

- **`trained model found`** → run `python train.py` first.
- **Out-of-memory on Cloud Run** → increase `--memory` to `2Gi`/`4Gi`.
- **Wrong class names** → fix folder names to match `CLASS_NAMES` in `utils.py`.
- **Fonts look different on the server** → expected; Cambria falls back to a
  serif of the same metrics.

---

*Built for educational and research use. Field decisions should be confirmed by
an agronomist or plant-pathology lab.*
