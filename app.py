import sys
import os

# Auto-redirect to project .venv streamlit if executed via python app.py
_project_dir = os.path.dirname(os.path.abspath(__file__))
_venv_streamlit = os.path.join(_project_dir, ".venv", "Scripts", "streamlit.exe")
if os.path.exists(_venv_streamlit):
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_streamlit = get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False

    if not _in_streamlit and "streamlit" not in os.path.basename(sys.argv[0]).lower():
        import socket
        import subprocess
        import webbrowser

        def _is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                return s.connect_ex(('127.0.0.1', port)) == 0

        target_port = 8502
        if _is_port_in_use(target_port):
            print(f"\n[OK] Bird Species app is ALREADY running on http://localhost:{target_port}!")
            print(f"[OK] Opening http://localhost:{target_port} in your browser...\n")
            webbrowser.open(f"http://localhost:{target_port}")
            sys.exit(0)

        sys.exit(subprocess.call([_venv_streamlit, "run", os.path.abspath(__file__), "--server.port", str(target_port)] + sys.argv[1:]))

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import os


# ─── 0)class NAMES───_____─────────────
class_names = [
    "Barn Swallow", "Bay breasted Warbler", "Black and white Warbler", "Black billed Cuckoo",
    "Black throated Blue Warbler", "Black throated Sparrow", "Blue Grosbeak", "Blue Jay",
    "Bobolink", "Bohemian Waxwing", "Bronzed Cowbird", "Brown Creeper",
    "Brown Pelican", "Brown Thrasher", "Canada Warbler", "Cape Glossy Starling",
    "Cape May Warbler", "Cardinal", "Carolina Wren", "Caspian Tern",
    "Cedar Waxwing", "Cerulean Warbler", "Chuck will Widow", "Clark Nutcracker",
    "Common Yellowthroat", "Eared Grebe", "Eastern Towhee", "European Goldfinch",
    "Evening Grosbeak", "Forsters Tern", "Fox Sparrow", "Geococcyx",
    "Golden winged Warbler", "Gray Kingbird", "Gray crowned Rosy Finch", "Green Jay",
    "Green Violetear", "Green tailed Towhee", "Harris Sparrow", "Heermann Gull",
    "Hooded Merganser", "Hooded Oriole", "Hooded Warbler", "Horned Grebe",
    "Horned Lark", "Horned Puffin", "Ivory Gull", "Lazuli Bunting",
    "Le Conte Sparrow", "Least Auklet", "Loggerhead Shrike", "Magnolia Warbler",
    "Mallard", "Myrtle Warbler", "Nashville Warbler", "Nelson Sharp tailed Sparrow",
    "Nighthawk", "Ovenbird", "Pacific Loon", "Painted Bunting",
    "Palm Warbler", "Parakeet Auklet", "Pied Kingfisher", "Pied billed Grebe",
    "Pine Grosbeak", "Pine Warbler", "Prairie Warbler", "Prothonotary Warbler",
    "Purple Finch", "Red bellied Woodpecker", "Red cockaded Woodpecker", "Red eyed Vireo",
    "Red winged Blackbird", "Rhinoceros Auklet", "Rose breasted Grosbeak", "Ruby throated Hummingbird",
    "Rufous Hummingbird", "Savannah Sparrow", "Sayornis", "Scarlet Tanager",
    "Scissor tailed Flycatcher", "Spotted Catbird", "Summer Tanager", "Tree Swallow",
    "Tropical Kingbird", "Vermilion Flycatcher", "Vesper Sparrow", "Warbling Vireo",
    "Western Meadowlark", "Western Wood Pewee", "Whip poor Will", "White Pelican",
    "White breasted Kingfisher", "White breasted Nuthatch", "White crowned Sparrow", "White throated Sparrow",
    "Yellow Warbler", "Yellow breasted Chat", "Yellow headed Blackbird", "Yellow throated Vireo"
]

# ─── Check which models are available ────────────────────────────────────────
YOLO_MODEL_PATH = "models/epoch45.pt"
RESNET_MODEL_PATH = "models/model_state_dict_best.pth"

HAS_YOLO = os.path.exists(YOLO_MODEL_PATH)
HAS_RESNET = os.path.exists(RESNET_MODEL_PATH)

# ─── 1) CACHING MODEL LOADERS (Lazy-imported for instant app startup) ─────────

@st.cache_resource
def load_yolo_model(path="models/epoch45.pt"):
    from ultralytics import YOLO
    return YOLO(path)

@st.cache_resource
def load_resnet_model(path="models/model_state_dict_best.pth", device="cpu"):
    import torch
    from torchvision import models

    # 1) Build the bare ResNet152
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    # 2) Load the raw state_dict
    raw_sd = torch.load(path, map_location="cpu")

    # 3) Strip "module." prefixes if present
    new_sd = {}
    for k, v in raw_sd.items():
        name = k
        if k.startswith("module."):
            name = k[len("module."):]
        new_sd[name] = v

    # 4) LOAD INTO THE MODEL
    model.load_state_dict(new_sd)

    # 5) Wrap for multi‑GPU (if you'll run on GPU) and move to device
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_swin_pipeline():
    from transformers import pipeline
    return pipeline("image-classification", model="Emiel/cub-200-bird-classifier-swin", device=-1)

# ─── 2) TRANSFORM FOR RESNET ─────────────────────────────────────────────────

def get_resnet_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ─── 3) STREAMLIT UI ────────────────────────────────────────────────────────

st.set_page_config(page_title="Bird Species Classification", page_icon="🐦", layout="wide")

# Custom styling for high-end presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 24px;
    }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🐦 Bird Species Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Hierarchical Swin Transformer Architecture · CUB-200-2011 Dataset</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🧠 Model Architecture")
    st.markdown("**Backbone:** Swin Transformer (`Swin-B`)")
    st.markdown("**Task:** Fine-Grained 200 Bird Species Recognition")
    st.markdown("**Benchmark Dataset:** Caltech-UCSD Birds-200-2011")
    st.markdown("**Taxonomy:** 200 North American Species")
    st.divider()
    st.header("⚡ Pipeline Highlights")
    st.markdown("• **Shifted Windows:** Self-attention within local windows with linear complexity")
    st.markdown("• **Hierarchical Maps:** Multi-scale feature representations")
    st.markdown("• **Inference Mode:** Softmax Top-5 Probability Distribution")

uploaded = st.file_uploader("Upload a bird image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])
if not uploaded:
    st.info("👆 Please upload a bird image above to run classification.")
    st.stop()

img = Image.open(uploaded).convert("RGB")

# Resize large images for fast CPU inference
inf_img = img.copy()
inf_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

col_img, col_pred = st.columns([1, 1.2])

with col_img:
    st.subheader("📸 Uploaded Image")
    st.image(img, use_container_width=True)

with col_pred:
    with st.spinner("Classifying bird species..."):
        hf_pipe = load_swin_pipeline()
        results = hf_pipe(inf_img, top_k=5)

    top_label = results[0]["label"].replace("_", " ").title()
    top_score = results[0]["score"]

    st.subheader("🎯 Primary Prediction")
    st.success(f"### {top_label}")
    st.metric(label="Prediction Confidence", value=f"{top_score:.2%}")

    st.subheader("📊 Top-5 Species Probabilities")
    for r in results:
        clean_name = r["label"].replace("_", " ").title()
        score = r["score"]
        st.progress(score, text=f"{clean_name}: {score:.2%}")
