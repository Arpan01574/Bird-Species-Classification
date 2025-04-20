import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO
from transformers import pipeline
import tempfile

# ─── 0) PAGE CONFIG & STYLES ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bird Species Classification",
    page_icon="🐦",
    layout="wide"
)
# Inject custom CSS for background and fonts
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(120deg, #e0f7fa, #ffe0b2);
        }
        h1 {
            font-family: 'DejaVu Sans', sans-serif;
            font-size: 2.5rem;
            color: #37474f;
            text-align: center;
        }
        .stImage {
            border: 2px solid #ccc;
            border-radius: 10px;
            padding: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── 1) CLASS NAMES ─────────────────────────────────────────────────────────
class_names = [
    # ... (same list as before) ...
]

# ─── 2) SIDEBAR SETTINGS ────────────────────────────────────────────────────
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.01
)

# ─── 3) CACHE MODEL LOADERS ─────────────────────────────────────────────────
@st.cache_resource
def load_yolo_model(path="models/epoch45.pt"):
    return YOLO(path)

@st.cache_resource
def load_resnet_model(path="models/model_state_dict_best.pth", device="cpu"):
    model = models.resnet152(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    raw_sd = torch.load(path, map_location="cpu")
    new_sd = {k[len("module."):]:v if k.startswith("module.") else v for k,v in raw_sd.items()}
    model.load_state_dict(new_sd)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_swin_pipeline():
    return pipeline("image-classification", model="Emiel/cub-200-bird-classifier-swin")

# ─── 4) TRANSFORM FOR RESNET ─────────────────────────────────────────────────
resnet_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ─── 5) STREAMLIT UI ────────────────────────────────────────────────────────
st.title("🐦 Bird Species Classification")

uploaded = st.file_uploader("Upload a bird image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload an image.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
st.image(img, caption="Uploaded Image", use_container_width=True)

# Load models
yolo_model   = load_yolo_model()
resnet_model = load_resnet_model()
hf_pipe      = load_swin_pipeline()

# Pre-load a TrueType font for labels
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=18)
except IOError:
    font = ImageFont.load_default()

# Run YOLO detection
results = yolo_model(np.array(img))
boxes   = results[0].boxes.xyxy.cpu().numpy()
scores_full = hf_pipe(img)[0]["score"]

annotated = img.copy()
draw = ImageDraw.Draw(annotated)

# Precompute HF result on the full image
hf_res_full = hf_pipe(img)
hf_label, hf_conf = hf_res_full[0]["label"], hf_res_full[0]["score"]

for box in boxes:
    x1,y1,x2,y2 = map(int, box)
    crop = img.crop((x1,y1,x2,y2))

    # ResNet prediction on the crop
    inp = resnet_transform(crop).unsqueeze(0)
    with torch.no_grad():
        out   = resnet_model(inp)
        probs = torch.softmax(out, dim=1)[0]
        ridx  = torch.argmax(probs).item()
        rlabel, rconf = class_names[ridx], probs[ridx].item()

    # Choose best
    if hf_conf > rconf:
        label, score, method = hf_label, hf_conf, "HF"
    else:
        label, score, method = rlabel, rconf, "ResNet"

    # Filter by threshold
    if score < confidence_threshold:
        continue

    # Draw box
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)

    # Prepare text background
    text = f"{label} ({method}, {score:.2f})"
    text_w, text_h = draw.textsize(text, font=font)
    text_origin = (x1, max(y1 - text_h - 6, 0))
    # Background rectangle for text
    draw.rectangle(
        [
            text_origin,
            (text_origin[0] + text_w + 6, text_origin[1] + text_h + 6)
        ], fill="red"
    )
    # Text label
    draw.text(
        (text_origin[0] + 3, text_origin[1] + 3),
        text,
        fill="white",
        font=font
    )

st.image(annotated, caption="Detections & Predictions", use_container_width=True)
