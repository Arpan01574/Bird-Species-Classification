import sys
import os

_project_dir = os.path.dirname(os.path.abspath(__file__))

# Auto-redirect to project .venv streamlit if executed via python app.py
if __name__ == "__main__":
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

            sys.exit(subprocess.call(
                [_venv_streamlit, "run", os.path.abspath(__file__), "--server.port", str(target_port)] + sys.argv[1:],
                cwd=_project_dir
            ))

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

# ─── 0) CLASS NAMES (100 CUB Species) ─────────────────────────────────────────
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

# ─── Model Paths & Availability ───────────────────────────────────────────────
YOLO_MODEL_PATH = os.path.join(_project_dir, "models", "epoch45.pt")
RESNET_MODEL_PATH = os.path.join(_project_dir, "models", "model_state_dict_best.pth")

HAS_YOLO = os.path.exists(YOLO_MODEL_PATH)
HAS_RESNET = os.path.exists(RESNET_MODEL_PATH)

# ─── 1) CACHING MODEL LOADERS ────────────────────────────────────────────────

@st.cache_resource
def load_yolo_model(path=YOLO_MODEL_PATH):
    from ultralytics import YOLO
    return YOLO(path)

@st.cache_resource
def load_resnet_model(path=RESNET_MODEL_PATH, device=None):
    from torchvision import models
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    raw_sd = torch.load(path, map_location=device)
    new_sd = { (k[7:] if k.startswith("module.") else k): v for k, v in raw_sd.items() }
    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_swin_pipeline():
    from transformers import pipeline
    return pipeline("image-classification", model="Emiel/cub-200-bird-classifier-swin", device=-1)

# ─── 2) IMAGE TRANSFORMS ──────────────────────────────────────────────────────

def get_resnet_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ─── 3) STREAMLIT UI SETUP ────────────────────────────────────────────────────

st.set_page_config(page_title="Bird Species Classification & Detection", page_icon="🐦", layout="wide")

# Custom styling
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
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .model-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-yolo { background: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-resnet { background: #1e1b4b; color: #818cf8; border: 1px solid #4f46e5; }
    .badge-swin { background: #3b0764; color: #c084fc; border: 1px solid #9333ea; }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🐦 Bird Species Classification & Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-Stage Computer Vision Pipeline: Custom YOLOv8 Detector · Custom ResNet-152 · Swin Transformer Benchmark</p>', unsafe_allow_html=True)

# ─── 4) SIDEBAR ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🧠 Pipeline Status")

    if HAS_YOLO:
        st.success("✅ **YOLOv8 Detector** (`epoch45.pt`)")
    else:
        st.warning("⚠️ YOLO weights missing (`models/epoch45.pt`)")

    if HAS_RESNET:
        st.success("✅ **Custom ResNet-152** (`model_state_dict_best.pth`)")
    else:
        st.warning("⚠️ ResNet weights missing (`models/model_state_dict_best.pth`)")

    st.success("✅ **Swin-B Transformer** (Hugging Face Cache)")

    st.divider()
    st.header("⚙️ Detection Settings")
    conf_thresh = st.slider("YOLO Detection Confidence:", min_value=0.1, max_value=0.9, value=0.25, step=0.05)
    crop_padding = st.slider("Crop Margin Padding (%):", min_value=0, max_value=25, value=10, step=5)

    st.divider()
    st.header("📐 Architecture Details")
    st.markdown("""
    - **Detector:** YOLOv8 Fine-tuned on CUB Bounding Boxes
    - **Primary Classifier:** ResNet-152 trained on 100 CUB Species
    - **Benchmark Ensemble:** Swin Transformer (`Swin-B`)
    - **Inference Hardware:** PyTorch CPU / CUDA
    """)

# ─── 5) SAMPLE IMAGES & UPLOADER ─────────────────────────────────────────────

test_images_dir = os.path.join(_project_dir, "test_images")
sample_files = []
if os.path.exists(test_images_dir):
    sample_files = [f for f in sorted(os.listdir(test_images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

col_source1, col_source2 = st.columns([1, 1])

with col_source1:
    uploaded = st.file_uploader("Upload a bird image (JPG, PNG, WebP):", type=["jpg", "jpeg", "png", "webp"])

with col_source2:
    selected_sample = None
    if sample_files:
        sample_options = ["None (Upload custom image)"] + [s.replace("_", " ").replace(".jpg", "") for s in sample_files]
        choice = st.selectbox("Or choose a sample bird from dataset:", sample_options)
        if choice != "None (Upload custom image)":
            chosen_filename = sample_files[sample_options.index(choice) - 1]
            selected_sample = os.path.join(test_images_dir, chosen_filename)

# Determine active image
active_img = None
active_source_label = ""

if uploaded:
    active_img = Image.open(uploaded).convert("RGB")
    active_source_label = f"Uploaded: {uploaded.name}"
elif selected_sample:
    active_img = Image.open(selected_sample).convert("RGB")
    active_source_label = f"Sample: {os.path.basename(selected_sample)}"

if active_img is None:
    st.info("👆 Please upload a bird image or select a sample image above to run detection and classification.")
    st.stop()

# ─── 6) INFERENCE PIPELINE ───────────────────────────────────────────────────

col_left, col_right = st.columns([1, 1.15])

with col_left:
    st.subheader(f"📸 Bird Detection ({active_source_label})")

    # Run YOLO detection
    detected_boxes = []
    annotated_img = active_img.copy()

    if HAS_YOLO:
        with st.spinner("Running YOLOv8 bird detector..."):
            yolo_model = load_yolo_model()
            yolo_res = yolo_model(active_img, conf=conf_thresh, verbose=False)
            boxes_data = yolo_res[0].boxes

            if len(boxes_data) > 0:
                xyxy = boxes_data.xyxy.cpu().numpy()
                confs = boxes_data.conf.cpu().numpy()

                draw = ImageDraw.Draw(annotated_img)
                w_img, h_img = active_img.size

                for i, (b, c) in enumerate(zip(xyxy, confs)):
                    x1, y1, x2, y2 = b
                    # Add margin padding
                    pad_x = (x2 - x1) * (crop_padding / 100.0)
                    pad_y = (y2 - y1) * (crop_padding / 100.0)
                    cx1 = max(0, int(x1 - pad_x))
                    cy1 = max(0, int(y1 - pad_y))
                    cx2 = min(w_img, int(x2 + pad_x))
                    cy2 = min(h_img, int(y2 + pad_y))

                    detected_boxes.append({
                        "box": [cx1, cy1, cx2, cy2],
                        "conf": float(c),
                        "crop": active_img.crop((cx1, cy1, cx2, cy2))
                    })

                    # Draw clean bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="#10b981", width=4)
                    draw.rectangle([x1, max(0, y1 - 24), x1 + 140, y1], fill="#10b981")
                    draw.text((x1 + 6, max(0, y1 - 20)), f"Bird: {c:.1%}", fill="#ffffff")

    st.image(annotated_img, use_container_width=True)

    if detected_boxes:
        st.success(f"🎯 **YOLOv8 detected {len(detected_boxes)} bird region(s)!** Top detection confidence: `{detected_boxes[0]['conf']:.1%}`")
        # Crop preview
        with st.expander("🔍 View Cropped Region for Classification", expanded=False):
            st.image(detected_boxes[0]["crop"], caption=f"Cropped ROI ({detected_boxes[0]['crop'].size[0]}x{detected_boxes[0]['crop'].size[1]}px)", use_container_width=True)
    else:
        st.info("ℹ️ No distinct bounding box detected above threshold — analyzing full image frame.")

# ─── 7) CLASSIFICATION INFERENCE ─────────────────────────────────────────────

# Classification input image: use cropped bird if detected, else full image
eval_crop = detected_boxes[0]["crop"] if detected_boxes else active_img

# Resize for fast Swin inference
inf_swin_img = active_img.copy()
inf_swin_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

with col_right:
    tab_overview, tab_resnet, tab_swin = st.tabs(["✨ Side-by-Side Comparison", "🔬 Custom ResNet-152", "🌐 Swin Transformer"])

    # --- Run ResNet-152 ---
    resnet_results = []
    if HAS_RESNET:
        with st.spinner("Classifying with your custom ResNet-152..."):
            resnet_model = load_resnet_model()
            transform = get_resnet_transform()
            inp = transform(eval_crop).unsqueeze(0)
            device = next(resnet_model.parameters()).device
            inp = inp.to(device)

            with torch.no_grad():
                out = resnet_model(inp)
                probs = torch.softmax(out, dim=1)[0]
                top5_p, top5_i = torch.topk(probs, 5)
                for p, idx in zip(top5_p, top5_i):
                    resnet_results.append({
                        "label": class_names[idx.item()],
                        "score": float(p.item())
                    })

    # --- Run Swin Transformer ---
    swin_results = []
    with st.spinner("Classifying with Swin-B Transformer..."):
        swin_pipe = load_swin_pipeline()
        raw_swin = swin_pipe(inf_swin_img, top_k=5)
        for r in raw_swin:
            swin_results.append({
                "label": r["label"].replace("_", " ").title(),
                "score": float(r["score"])
            })

    # --- TAB 1: SIDE-BY-SIDE OVERVIEW ---
    with tab_overview:
        st.subheader("🎯 Primary Model Predictions")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<span class="model-badge badge-resnet">CUSTOM RESNET-152</span>', unsafe_allow_html=True)
            if resnet_results:
                r_top = resnet_results[0]
                st.success(f"**{r_top['label']}**")
                st.metric("Confidence", f"{r_top['score']:.2%}")
            else:
                st.warning("ResNet weights not found.")

        with c2:
            st.markdown('<span class="model-badge badge-swin">SWIN-B TRANSFORMER</span>', unsafe_allow_html=True)
            if swin_results:
                s_top = swin_results[0]
                st.success(f"**{s_top['label']}**")
                st.metric("Confidence", f"{s_top['score']:.2%}")

        # Agreement / Consensus check
        if resnet_results and swin_results:
            st.divider()
            r_label_clean = resnet_results[0]['label'].lower()
            s_label_clean = swin_results[0]['label'].lower()
            if r_label_clean == s_label_clean or r_label_clean in s_label_clean or s_label_clean in r_label_clean:
                st.balloons()
                st.info(f"🤝 **Consensus Reached!** Both your Custom ResNet-152 and Swin Transformer agree on: **{resnet_results[0]['label']}**")
            else:
                st.caption(f"💡 ResNet predicted **{resnet_results[0]['label']}**, while Swin predicted **{swin_results[0]['label']}**.")

        st.subheader("📊 Top Predictions Comparison")
        st.markdown("**Your Custom ResNet-152:**")
        if resnet_results:
            for r in resnet_results[:3]:
                st.progress(r["score"], text=f"{r['label']}: {r['score']:.2%}")

        st.markdown("**Swin Transformer:**")
        if swin_results:
            for s in swin_results[:3]:
                st.progress(s["score"], text=f"{s['label']}: {s['score']:.2%}")

    # --- TAB 2: RESNET-152 DEEP DIVE ---
    with tab_resnet:
        st.markdown('<span class="model-badge badge-resnet">CUSTOM TRAINED WEIGHTS</span>', unsafe_allow_html=True)
        st.caption("ResNet-152 model trained directly on the CUB dataset with custom head and data augmentation.")
        if resnet_results:
            for i, r in enumerate(resnet_results, 1):
                st.write(f"**{i}. {r['label']}** — `{r['score']:.2%}`")
                st.progress(r["score"])
        else:
            st.warning("ResNet weights not loaded.")

    # --- TAB 3: SWIN TRANSFORMER DEEP DIVE ---
    with tab_swin:
        st.markdown('<span class="model-badge badge-swin">SWIN TRANSFORMER BENCHMARK</span>', unsafe_allow_html=True)
        st.caption("Hierarchical Swin-B vision transformer for fine-grained 200 species recognition.")
        if swin_results:
            for i, s in enumerate(swin_results, 1):
                st.write(f"**{i}. {s['label']}** — `{s['score']:.2%}`")
                st.progress(s["score"])
