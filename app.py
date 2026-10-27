import os
import sys
import time
import glob
import torch
import numpy as np
from PIL import Image, ImageDraw

# Ensure local imports resolve regardless of launch working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import streamlit as st
import streamlit.components.v1 as components
from model.loader import load_generator_model, set_dropout_eval_mode
from utils.image_ops import (
    FACADE_CLASSES,
    preprocess_layout,
    postprocess_facade,
    create_side_by_side_comparison,
    create_triplet_comparison,
    analyze_layout_colors,
    pil_to_bytes,
)

# -----------------------------------------------------------------------------
# Page Configuration & Architectural Dark Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Architectural Blueprint to Facade Renderer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, #161b22 0%, #1a2230 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f0f6fc;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 16px;
        line-height: 1.5;
    }
    
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .arch-badge {
        display: inline-flex;
        align-items: center;
        background-color: #21262d;
        color: #f59e0b;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .arch-badge.blue {
        color: #38bdf8;
    }
    
    .arch-badge.green {
        color: #4ade80;
    }
    
    .arch-badge.purple {
        color: #c084fc;
    }

    /* Cards */
    .arch-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }
    
    /* Metric Card */
    .metric-pill {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f59e0b;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Swatch element */
    .swatch-box {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 10px;
        background: #161b22;
        border-radius: 6px;
        border: 1px solid #30363d;
        margin-bottom: 6px;
    }
    .swatch-color {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid rgba(255,255,255,0.2);
        flex-shrink: 0;
    }
    .swatch-name {
        font-size: 0.82rem;
        font-weight: 600;
        color: #c9d1d9;
    }
    .swatch-hex {
        font-size: 0.72rem;
        color: #8b949e;
        margin-left: auto;
        font-family: monospace;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .primary-btn button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        color: #0f172a !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.3) !important;
    }
    .primary-btn button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.45) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "sample_page" not in st.session_state:
    st.session_state["sample_page"] = 0

sample_dir = os.path.join(current_dir, "samples")
all_sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))

if "active_blueprint" not in st.session_state or st.session_state["active_blueprint"] is None:
    if all_sample_files:
        default_sample = all_sample_files[0]
        st.session_state["active_blueprint"] = Image.open(default_sample).convert("RGB")
        sample_bname = os.path.splitext(os.path.basename(default_sample))[0]
        st.session_state["blueprint_source"] = f"Sample: {sample_bname}"
        matching_jpg = default_sample.replace(".png", ".jpg")
        if os.path.exists(matching_jpg):
            st.session_state["ground_truth_photo"] = Image.open(matching_jpg).convert("RGB")
        else:
            st.session_state["ground_truth_photo"] = None
    else:
        st.session_state["active_blueprint"] = None
        st.session_state["blueprint_source"] = None
        st.session_state["ground_truth_photo"] = None

if "drawing_elements" not in st.session_state:
    # Initialize studio drawing canvas layers with a baseline classical facade
    st.session_state["drawing_elements"] = [
        {"class": "facade", "bbox": (12, 35, 244, 255)},
        {"class": "cornice", "bbox": (10, 24, 246, 36)},
        # Floor 3 windows
        {"class": "window", "bbox": (32, 50, 68, 86)},
        {"class": "sill", "bbox": (30, 86, 70, 90)},
        {"class": "window", "bbox": (92, 50, 128, 86)},
        {"class": "sill", "bbox": (90, 86, 130, 90)},
        {"class": "window", "bbox": (152, 50, 188, 86)},
        {"class": "sill", "bbox": (150, 86, 190, 90)},
        # Floor 2 windows with balconies
        {"class": "window", "bbox": (32, 105, 68, 145)},
        {"class": "balcony", "bbox": (28, 140, 72, 155)},
        {"class": "window", "bbox": (92, 105, 128, 145)},
        {"class": "balcony", "bbox": (88, 140, 132, 155)},
        {"class": "window", "bbox": (152, 105, 188, 145)},
        {"class": "balcony", "bbox": (148, 140, 192, 155)},
        # Floor 1 windows
        {"class": "window", "bbox": (32, 168, 68, 204)},
        {"class": "sill", "bbox": (30, 204, 70, 208)},
        {"class": "window", "bbox": (152, 168, 188, 204)},
        {"class": "sill", "bbox": (150, 204, 190, 208)},
        # Ground floor entrance door & storefront
        {"class": "door", "bbox": (90, 175, 130, 255)},
        {"class": "shop", "bbox": (24, 218, 76, 255)},
        {"class": "shop", "bbox": (144, 218, 232, 255)},
    ]

# -----------------------------------------------------------------------------
# Load Model & Handle Checkpoint State
# -----------------------------------------------------------------------------
model_loaded = False
load_error = None
try:
    model, metadata, device = load_generator_model("model/facade_generator.pkl")
    model_loaded = True
except Exception as e:
    load_error = str(e)

# -----------------------------------------------------------------------------
# Hero Header Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🏛️ Architectural Blueprint to Facade Renderer</div>
        <div class="hero-subtitle">
            Translate 2D semantic building segmentation layouts into photorealistic architectural facades 
            using a Conditional Generative Adversarial Network (Pix2Pix U-Net Generator).
        </div>
        <div class="badge-bar">
            <span class="arch-badge">Pix2Pix cGAN</span>
            <span class="arch-badge blue">U-Net Generator (8 Down / 8 Up)</span>
            <span class="arch-badge green">70x70 PatchGAN Discriminator</span>
            <span class="arch-badge purple">CMP Facades Dataset</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_loaded:
    st.error(
        f"⚠️ **Model Loading Error**: {load_error}\n\n"
        "Please ensure `facade_generator.pkl` is located at `model/facade_generator.pkl`."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar: About the Model, Metrics, and Class Palette Legend
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Status")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-val">{metadata['device_name']}</div>
                <div class="metric-lbl">Accelerator</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_d2:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-val">{metadata['input_size']}×{metadata['input_size']}</div>
                <div class="metric-lbl">Resolution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Test Split Metrics")
    metrics = metadata.get("test_metrics", {})
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-val">{metrics.get('L1', 0.1847):.4f}</div>
                <div class="metric-lbl">L1 Loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-val">{metrics.get('PSNR', 12.56):.2f} dB</div>
                <div class="metric-lbl">PSNR</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-pill">
                <div class="metric-val">{metrics.get('SSIM', 0.2288):.4f}</div>
                <div class="metric-lbl">SSIM</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🧠 How Pix2Pix Works")
    st.markdown(
        """
        **Pix2Pix** is a supervised image-to-image conditional GAN framework:
        
        1. **U-Net Generator**: 8 downsampling and 8 upsampling blocks. Skip connections 
           between mirrors pass fine geometric blueprint edges directly across the bottleneck, 
           preventing structural loss.
        2. **70×70 PatchGAN Discriminator**: Classifies local 70×70 patches as real or fake, 
           penalizing high-frequency blur and forcing crisp brick, masonry, and glass textures.
        3. **Compound Objective**:
        """
    )
    st.latex(r"\mathcal{G}^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)")
    st.markdown(
        """
        The $\\mathcal{L}_{L1}$ term (weight $\\lambda=100$) anchors pixel-level color fidelity, 
        while $\\mathcal{L}_{cGAN}$ synthesizes photorealistic details.
        
        > **Note:** Outputs are highest in quality when input layouts adhere to the CMP Facades color palette.
        """
    )

    st.markdown("---")
    st.markdown("### 🎨 CMP Facades Palette Legend")
    for cls in FACADE_CLASSES:
        st.markdown(
            f"""
            <div class="swatch-box" title="{cls['desc']}">
                <div class="swatch-color" style="background-color: {cls['hex']};"></div>
                <span class="swatch-name">{cls['name']}</span>
                <span class="swatch-hex">{cls['hex']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Main Section: Input Modes (Tabs)
# -----------------------------------------------------------------------------
st.markdown("### 1. Select or Create Building Blueprint")
tab_sample, tab_upload, tab_draw = st.tabs(
    ["🏛️ Sample Layouts Gallery", "📤 Upload Custom Blueprint", "✏️ Blueprint Studio & Draw Layout"]
)

# --- Tab 1: Sample Layouts Gallery ---
with tab_sample:
    sample_dir = os.path.join(current_dir, "samples")
    sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
    
    if not sample_files:
        st.warning("No sample layout images found in `samples/` directory.")
    else:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="color:#8b949e; font-size:0.95rem;">
                    📁 <strong>{len(sample_files)}</strong> building blueprints loaded from the CMP Facades dataset.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Quick Search / Jump Dropdown
        with st.expander("🔍 Search & Jump to Any Sample by ID (378 Available)", expanded=False):
            sample_labels = [os.path.splitext(os.path.basename(p))[0] for p in sample_files]
            chosen_sample_name = st.selectbox(
                "Choose sample ID",
                sample_labels,
                index=0,
                key="sample_selector_dropdown"
            )
            chosen_path = os.path.join(sample_dir, f"{chosen_sample_name}.png")
            chosen_jpg = os.path.join(sample_dir, f"{chosen_sample_name}.jpg")
            
            c_prev1, c_prev2 = st.columns(2)
            with c_prev1:
                st.image(Image.open(chosen_path), caption=f"Blueprint: {chosen_sample_name}", use_container_width=True)
            with c_prev2:
                if os.path.exists(chosen_jpg):
                    st.image(Image.open(chosen_jpg), caption="Ground Truth Photo (Dataset)", use_container_width=True)
                else:
                    st.info("No ground truth photo found.")
            if st.button(f"✓ Select {chosen_sample_name} as Active Blueprint", type="primary", use_container_width=True):
                st.session_state["active_blueprint"] = Image.open(chosen_path).convert("RGB")
                st.session_state["blueprint_source"] = f"Sample: {chosen_sample_name}"
                if os.path.exists(chosen_jpg):
                    st.session_state["ground_truth_photo"] = Image.open(chosen_jpg).convert("RGB")
                else:
                    st.session_state["ground_truth_photo"] = None
                st.rerun()

        # Paginated Grid Gallery
        samples_per_page = 8
        total_pages = max(1, (len(sample_files) + samples_per_page - 1) // samples_per_page)
        
        # Page navigation controls
        pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 2, 2])
        with pag_col1:
            if st.button("◀ Prev", disabled=(st.session_state["sample_page"] == 0), use_container_width=True):
                st.session_state["sample_page"] = max(0, st.session_state["sample_page"] - 1)
                st.rerun()
        with pag_col2:
            if st.button("Next ▶", disabled=(st.session_state["sample_page"] >= total_pages - 1), use_container_width=True):
                st.session_state["sample_page"] = min(total_pages - 1, st.session_state["sample_page"] + 1)
                st.rerun()
        with pag_col3:
            st.markdown(f"<div style='padding-top:6px; color:#c9d1d9; font-weight:600;'>Page {st.session_state['sample_page'] + 1} of {total_pages}</div>", unsafe_allow_html=True)
        with pag_col4:
            selected_page = st.selectbox(
                "Jump to Page",
                range(1, total_pages + 1),
                index=st.session_state["sample_page"],
                label_visibility="collapsed",
                key="sample_page_selector"
            )
            if selected_page - 1 != st.session_state["sample_page"]:
                st.session_state["sample_page"] = selected_page - 1
                st.rerun()
                
        # Current page slice
        start_idx = st.session_state["sample_page"] * samples_per_page
        end_idx = min(len(sample_files), start_idx + samples_per_page)
        page_samples = sample_files[start_idx:end_idx]
        
        cols_per_row = 4
        rows = [page_samples[i:i + cols_per_row] for i in range(0, len(page_samples), cols_per_row)]
        
        for r_idx, row in enumerate(rows):
            cols = st.columns(cols_per_row)
            for c_idx, sample_path in enumerate(row):
                with cols[c_idx]:
                    filename = os.path.basename(sample_path)
                    clean_name = os.path.splitext(filename)[0]
                    sample_img = Image.open(sample_path)
                    
                    is_active = (st.session_state.get("blueprint_source") == f"Sample: {clean_name}")
                    
                    st.image(sample_img, caption=f"Layout: {clean_name}", use_container_width=True)
                    if st.button(f"{'✓ Active' if is_active else 'Select'}", key=f"btn_p_{start_idx + r_idx * cols_per_row + c_idx}", use_container_width=True, type="primary" if is_active else "secondary"):
                        st.session_state["active_blueprint"] = sample_img.convert("RGB")
                        st.session_state["blueprint_source"] = f"Sample: {clean_name}"
                        jpg_path = sample_path.replace(".png", ".jpg")
                        if os.path.exists(jpg_path):
                            st.session_state["ground_truth_photo"] = Image.open(jpg_path).convert("RGB")
                        else:
                            st.session_state["ground_truth_photo"] = None
                        st.rerun()

# --- Tab 2: Upload Custom Blueprint ---
with tab_upload:
    st.caption("Upload a 2D building segmentation image (PNG or JPG). The image will be processed at 256×256.")
    uploaded_file = st.file_uploader(
        "Choose blueprint image file",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        try:
            uploaded_img = Image.open(uploaded_file).convert("RGB")
            up_col1, up_col2 = st.columns([1, 2])
            with up_col1:
                st.image(uploaded_img, caption="Uploaded Layout Preview", use_container_width=True)
                if st.button("Use This Upload as Active Blueprint", type="primary", use_container_width=True):
                    st.session_state["active_blueprint"] = uploaded_img
                    st.session_state["blueprint_source"] = f"Upload: {uploaded_file.name}"
                    st.session_state["ground_truth_photo"] = None
                    st.success("Uploaded blueprint set as active!")
                    st.rerun()
            with up_col2:
                st.markdown("#### Detected Semantic Architectural Classes")
                detected = analyze_layout_colors(uploaded_img)
                if not detected:
                    st.info("No dominant standard CMP Facades classes detected. The model will still attempt inference, but results are optimal with calibrated palette colors.")
                else:
                    for d in detected:
                        st.markdown(
                            f"""
                            <div class="swatch-box" style="margin-bottom: 4px;">
                                <div class="swatch-color" style="background-color: {d['hex']};"></div>
                                <span class="swatch-name">{d['name']}</span>
                                <span style="font-size: 0.8rem; color: #8b949e;">{d['coverage']:.1f}% area</span>
                                <span class="swatch-hex">{d['hex']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")

# --- Tab 3: Blueprint Studio & Draw Layout ---
with tab_draw:
    st.caption(
        "Design custom architectural layouts with the standard CMP Facades palette. "
        "Use preset templates or add custom elements with exact label colors."
    )
    
    studio_col_left, studio_col_right = st.columns([1, 1.3])
    
    # Helper to render the current drawing layers
    def render_studio_blueprint(elements):
        img = Image.new("RGB", (256, 256), (0, 0, 170))  # background — exact CMP Facades color
        draw = ImageDraw.Draw(img)
        class_map = {c["id"]: c["rgb"] for c in FACADE_CLASSES}
        for el in elements:
            rgb = class_map.get(el["class"], (0, 0, 255))
            draw.rectangle(el["bbox"], fill=rgb)
        return img

    with studio_col_left:
        st.markdown("#### 🎨 Active Blueprint Canvas")
        current_studio_img = render_studio_blueprint(st.session_state["drawing_elements"])
        st.image(current_studio_img, caption="Studio Canvas (256×256 pixels)", use_container_width=True)
        
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("↩️ Undo", use_container_width=True):
                if st.session_state["drawing_elements"]:
                    st.session_state["drawing_elements"].pop()
                    st.rerun()
        with btn_c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["drawing_elements"] = []
                st.rerun()
        with btn_c3:
            if st.button("🚀 Render This", type="primary", use_container_width=True):
                st.session_state["active_blueprint"] = current_studio_img
                st.session_state["blueprint_source"] = "Studio: Custom Architectural Design"
                st.session_state["ground_truth_photo"] = None
                st.success("Set as active blueprint!")
                st.rerun()

    with studio_col_right:
        st.markdown("#### 📐 Architecture Presets & Element Tools")
        preset_choice = st.selectbox(
            "Load Architectural Blueprint Preset",
            [
                "— Choose a Preset —",
                "Classic 3-Story Residence",
                "Commercial Storefront with Lofts",
                "Grand Symmetrical Palazzo",
                "Modern Balconied Apartments",
                "Blank Canvas (Background Only)",
            ]
        )
        
        if preset_choice != "— Choose a Preset —":
            if st.button("Apply Selected Preset"):
                if preset_choice == "Blank Canvas (Background Only)":
                    st.session_state["drawing_elements"] = []
                elif preset_choice == "Classic 3-Story Residence":
                    st.session_state["drawing_elements"] = [
                        {"class": "facade", "bbox": (15, 30, 241, 255)},
                        {"class": "cornice", "bbox": (12, 22, 244, 34)},
                        {"class": "window", "bbox": (35, 55, 70, 95)},
                        {"class": "sill", "bbox": (33, 95, 72, 99)},
                        {"class": "window", "bbox": (95, 55, 130, 95)},
                        {"class": "sill", "bbox": (93, 95, 132, 99)},
                        {"class": "window", "bbox": (155, 55, 190, 95)},
                        {"class": "sill", "bbox": (153, 95, 192, 99)},
                        {"class": "window", "bbox": (35, 120, 70, 160)},
                        {"class": "balcony", "bbox": (30, 155, 75, 170)},
                        {"class": "window", "bbox": (95, 120, 130, 160)},
                        {"class": "balcony", "bbox": (90, 155, 135, 170)},
                        {"class": "window", "bbox": (155, 120, 190, 160)},
                        {"class": "balcony", "bbox": (150, 155, 195, 170)},
                        {"class": "door", "bbox": (95, 185, 130, 255)},
                        {"class": "shop", "bbox": (25, 205, 80, 255)},
                        {"class": "shop", "bbox": (145, 205, 230, 255)},
                    ]
                elif preset_choice == "Commercial Storefront with Lofts":
                    st.session_state["drawing_elements"] = [
                        {"class": "facade", "bbox": (10, 25, 246, 255)},
                        {"class": "cornice", "bbox": (8, 16, 248, 28)},
                        {"class": "molding", "bbox": (10, 85, 246, 92)},
                        {"class": "window", "bbox": (30, 40, 65, 80)},
                        {"class": "window", "bbox": (85, 40, 120, 80)},
                        {"class": "window", "bbox": (140, 40, 175, 80)},
                        {"class": "window", "bbox": (195, 40, 230, 80)},
                        {"class": "molding", "bbox": (10, 150, 246, 156)},
                        {"class": "window", "bbox": (30, 105, 65, 145)},
                        {"class": "window", "bbox": (85, 105, 120, 145)},
                        {"class": "window", "bbox": (140, 105, 175, 145)},
                        {"class": "window", "bbox": (195, 105, 230, 145)},
                        {"class": "shop", "bbox": (20, 175, 115, 255)},
                        {"class": "shop", "bbox": (140, 175, 235, 255)},
                        {"class": "door", "bbox": (118, 185, 138, 255)},
                    ]
                elif preset_choice == "Grand Symmetrical Palazzo":
                    st.session_state["drawing_elements"] = [
                        {"class": "facade", "bbox": (16, 30, 240, 255)},
                        {"class": "cornice", "bbox": (12, 18, 244, 32)},
                        {"class": "pillar", "bbox": (20, 30, 36, 255)},
                        {"class": "pillar", "bbox": (220, 30, 236, 255)},
                        {"class": "window", "bbox": (50, 50, 85, 90)},
                        {"class": "window", "bbox": (110, 50, 145, 90)},
                        {"class": "window", "bbox": (170, 50, 205, 90)},
                        {"class": "window", "bbox": (50, 110, 85, 150)},
                        {"class": "balcony", "bbox": (100, 140, 155, 158)},
                        {"class": "window", "bbox": (110, 110, 145, 150)},
                        {"class": "window", "bbox": (170, 110, 205, 150)},
                        {"class": "door", "bbox": (105, 180, 150, 255)},
                        {"class": "deco", "bbox": (115, 162, 140, 176)},
                    ]
                elif preset_choice == "Modern Balconied Apartments":
                    st.session_state["drawing_elements"] = [
                        {"class": "facade", "bbox": (20, 35, 236, 255)},
                        {"class": "cornice", "bbox": (18, 25, 238, 38)},
                        {"class": "window", "bbox": (40, 55, 90, 100)},
                        {"class": "balcony", "bbox": (35, 95, 95, 110)},
                        {"class": "window", "bbox": (145, 55, 195, 100)},
                        {"class": "balcony", "bbox": (140, 95, 200, 110)},
                        {"class": "window", "bbox": (40, 125, 90, 170)},
                        {"class": "balcony", "bbox": (35, 165, 95, 180)},
                        {"class": "window", "bbox": (145, 125, 195, 170)},
                        {"class": "balcony", "bbox": (140, 165, 200, 180)},
                        {"class": "door", "bbox": (105, 190, 145, 255)},
                    ]
                st.rerun()

        st.markdown("##### ➕ Add Custom Architectural Element")
        class_options = [c["name"] for c in FACADE_CLASSES]
        selected_class_name = st.selectbox("Architectural Class", class_options, index=2)
        selected_class_id = next(c["id"] for c in FACADE_CLASSES if c["name"] == selected_class_name)
        selected_hex = next(c["hex"] for c in FACADE_CLASSES if c["name"] == selected_class_name)
        
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <div style="width:20px; height:20px; background-color:{selected_hex}; border-radius:4px; border:1px solid #555;"></div>
                <span style="font-size:0.85rem; color:#f59e0b; font-weight:600;">Active Class: {selected_class_name} ({selected_hex})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        pos_c1, pos_c2 = st.columns(2)
        with pos_c1:
            x_min = st.slider("X Position", 0, 250, 40, step=2)
            y_min = st.slider("Y Position", 0, 250, 60, step=2)
        with pos_c2:
            elem_w = st.slider("Element Width", 4, 256, 40, step=2)
            elem_h = st.slider("Element Height", 4, 256, 45, step=2)
            
        x_max = min(256, x_min + elem_w)
        y_max = min(256, y_min + elem_h)
        
        if st.button("Draw Rectangle", use_container_width=True):
            st.session_state["drawing_elements"].append({
                "class": selected_class_id,
                "bbox": (x_min, y_min, x_max, y_max)
            })
            st.rerun()

# -----------------------------------------------------------------------------
# Active Blueprint Status Banner
# -----------------------------------------------------------------------------
st.markdown("---")
active_bp = st.session_state.get("active_blueprint")
source_label = st.session_state.get("blueprint_source", "None selected")

col_banner1, col_banner2 = st.columns([3, 1])
with col_banner1:
    st.markdown(
        f"""
        <div style="background:#161b22; border-left:4px solid #f59e0b; padding:12px 16px; border-radius:0 8px 8px 0;">
            <span style="color:#8b949e; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px;">Active Input Layout</span><br>
            <strong style="font-size:1.1rem; color:#f0f6fc;">{source_label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_banner2:
    pass

# -----------------------------------------------------------------------------
# Inference Controls: Variations Slider, Upscale, and Generate Button
# -----------------------------------------------------------------------------
st.markdown("### 2. Render Settings & Generation")
ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.5, 1.5, 2])

with ctrl_c1:
    num_variations = st.slider(
        "Stochastic Variations",
        min_value=1,
        max_value=4,
        value=1,
        help="Runs generator with dropout enabled in decoder layers (BatchNorm in eval mode) to synthesize diverse architectural textures.",
    )

with ctrl_c2:
    upscale_toggle = st.checkbox(
        "Upscale Output to 512×512 (Lanczos)",
        value=False,
        help="Applies high-quality Lanczos resampling for display and download.",
    )

with ctrl_c3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    generate_clicked = st.button(
        "✨ Generate Architectural Facade",
        type="primary",
        use_container_width=True,
        disabled=(active_bp is None),
    )

# -----------------------------------------------------------------------------
# Generation Execution & Results Showcase
# -----------------------------------------------------------------------------
if generate_clicked and active_bp is not None:
    with st.spinner("Synthesizing architectural facade via Pix2Pix U-Net generator..."):
        try:
            # Preprocess blueprint to [-1, 1] tensor
            input_tensor, resized_blueprint = preprocess_layout(
                active_bp, target_size=256, device=device
            )
            
            generated_images = []
            inference_times = []
            
            # Primary deterministic run (or stochastic if multiple)
            set_dropout_eval_mode(model, enable_dropout=(num_variations > 1))
            
            for v_idx in range(num_variations):
                t_start = time.perf_counter()
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                t_infer = (time.perf_counter() - t_start) * 1000.0
                inference_times.append(t_infer)
                
                facade_pil = postprocess_facade(output_tensor, upscale_512=upscale_toggle)
                generated_images.append(facade_pil)
            
            st.session_state["rendered_facades"] = generated_images
            st.session_state["rendered_blueprint"] = resized_blueprint
            st.session_state["inference_times"] = inference_times
            st.session_state["upscaled"] = upscale_toggle
            
        except Exception as e:
            st.error(f"Inference error: {str(e)}")

# Display Rendered Results if available
if "rendered_facades" in st.session_state:
    facades = st.session_state["rendered_facades"]
    blueprint = st.session_state["rendered_blueprint"]
    times = st.session_state.get("inference_times", [38.0])
    is_upscaled = st.session_state.get("upscaled", False)
    
    avg_time = sum(times) / len(times)
    res_str = "512×512 px (Upscaled)" if is_upscaled else "256×256 px (Native)"
    
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h3 style="margin:0;">3. Rendered Facade Output</h3>
            <div style="display:flex; gap:8px;">
                <span class="arch-badge green">⚡ {avg_time:.1f} ms / image</span>
                <span class="arch-badge blue">📐 {res_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    primary_facade = facades[0]
    
    has_ground_truth = (st.session_state.get("ground_truth_photo") is not None)
    inspection_options = ["Side-by-Side Comparison", "Interactive Dissolve Overlay"]
    if has_ground_truth:
        inspection_options.append("Evaluation Triplet (Blueprint | AI Generated | Real Photo)")
    
    view_mode = st.radio(
        "Inspection Mode",
        inspection_options,
        horizontal=True,
    )
    
    if view_mode == "Side-by-Side Comparison":
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("##### 🟦 2D Semantic Blueprint")
            st.image(blueprint, use_container_width=True)
        with col_res2:
            st.markdown("##### 🏛️ Synthesized Realistic Facade")
            st.image(primary_facade, use_container_width=True)
    elif view_mode == "Evaluation Triplet (Blueprint | AI Generated | Real Photo)":
        gt_col1, gt_col2, gt_col3 = st.columns(3)
        with gt_col1:
            st.markdown("##### 🟦 1. Input Blueprint")
            st.image(blueprint, use_container_width=True)
        with gt_col2:
            st.markdown("##### 🏛️ 2. AI Synthesized Facade")
            st.image(primary_facade, use_container_width=True)
        with gt_col3:
            st.markdown("##### 📸 3. Real Ground Truth Photo")
            st.image(st.session_state["ground_truth_photo"], use_container_width=True)
    else:
        st.markdown("##### 🔀 Blueprint to Photorealism Dissolve")
        blend_factor = st.slider(
            "Slide to blend between Blueprint and Realized Facade",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.02,
        )
        
        # Match dimensions for blending
        b_sized = blueprint.resize(primary_facade.size, Image.Resampling.NEAREST)
        blended = Image.blend(b_sized, primary_facade, alpha=blend_factor)
        
        col_bl1, col_bl2, col_bl3 = st.columns([1, 2, 1])
        with col_bl2:
            st.image(
                blended,
                caption=f"Blend: {int((1-blend_factor)*100)}% Blueprint | {int(blend_factor*100)}% Facade",
                use_container_width=True,
            )
            
    # Download Buttons
    st.markdown("#### 📥 Export Deliverables")
    dl_cols = st.columns(3 if has_ground_truth else 2)
    with dl_cols[0]:
        facade_bytes = pil_to_bytes(primary_facade, "PNG")
        st.download_button(
            "⬇️ Download Facade Image (PNG)",
            data=facade_bytes,
            file_name="synthesized_facade.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_cols[1]:
        side_by_side_img = create_side_by_side_comparison(blueprint, primary_facade, upscale=is_upscaled)
        sbs_bytes = pil_to_bytes(side_by_side_img, "PNG")
        st.download_button(
            "⬇️ Download Side-by-Side Presentation (PNG)",
            data=sbs_bytes,
            file_name="blueprint_and_facade_presentation.png",
            mime="image/png",
            use_container_width=True,
        )
    if has_ground_truth:
        with dl_cols[2]:
            triplet_img = create_triplet_comparison(blueprint, primary_facade, st.session_state["ground_truth_photo"], upscale=is_upscaled)
            trip_bytes = pil_to_bytes(triplet_img, "PNG")
            st.download_button(
                "⬇️ Download Triplet Presentation (PNG)",
                data=trip_bytes,
                file_name="blueprint_generated_and_ground_truth.png",
                mime="image/png",
                use_container_width=True,
            )

    # If multiple variations were generated, display them in an aesthetic grid
    if len(facades) > 1:
        st.markdown("---")
        st.markdown(f"#### 🎭 Stochastic Facade Variations ({len(facades)} Generative Passes)")
        st.caption("Dropout layers sampled diverse structural nuances and window reflections:")
        
        var_cols = st.columns(len(facades))
        for v_idx, v_img in enumerate(facades):
            with var_cols[v_idx]:
                st.image(v_img, caption=f"Variation {v_idx + 1} ({times[v_idx]:.1f}ms)", use_container_width=True)
                v_bytes = pil_to_bytes(v_img, "PNG")
                st.download_button(
                    f"Download #{v_idx + 1}",
                    data=v_bytes,
                    file_name=f"facade_variation_{v_idx + 1}.png",
                    mime="image/png",
                    key=f"dl_var_{v_idx}",
                    use_container_width=True,
                )
