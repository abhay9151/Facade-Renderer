# Architectural Blueprint to Facade Renderer 🏛️

A modern, showcase-ready computer vision application that translates 2D semantic building blueprints into photorealistic architectural facades using a **Pix2Pix Conditional Generative Adversarial Network (cGAN)**.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch (CPU or GPU; automatically leverages Apple Silicon MPS or NVIDIA CUDA if available)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Pre-trained Model

The model file `facade_generator.pkl` (~208 MB) is excluded from git due to size.  
Place it at **`model/facade_generator.pkl`** before launching.

> Download it from the [GitHub Releases page](https://github.com/abhay9151/Facade-Renderer/releases) or ask the project author for the file.

### 4. Launch the Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to start rendering facades!

---

## 🎨 What the App Does

Users provide or design a 2D color-coded architectural layout (segmentation blueprint representing walls, windows, doors, balconies, roof cornices, sill ledges, and ground-floor shops). The app executes inference through a pre-trained Pix2Pix U-Net generator to synthesize a photorealistic photographic building facade in real-time (~38 ms per image on Apple Silicon MPS).

### Key Features
1. **CMP Facades Blueprint Gallery**: Browse and select from 378 architectural blueprints from the CMP Facades Dataset (with paired real-photo ground truth).
2. **Custom Blueprint Upload**: Upload any custom 2D layout image (PNG/JPG) with automatic semantic class detection and color coverage analysis.
3. **Interactive Blueprint Studio**: Compose and draw custom building layouts with instant visual feedback, presets, exact CMP Facades color swatches, rectangle tool, undo, and clear.
4. **Stochastic Variations**: Synthesize 1 to 4 diverse stylistic variations using active decoder dropout layers while keeping BatchNorm in evaluation mode.
5. **Interactive Before / After Comparison**: Inspect layouts and synthesized facades with side-by-side views, triplet evaluation (Blueprint | AI | Real Photo), or an interactive dissolve slider.
6. **Optional 512×512 Lanczos Upscaling**: Export high-resolution presentation renders.
7. **One-Click Deliverables Export**: Download the standalone synthesized facade PNG, a side-by-side presentation board, or a full evaluation triplet.

---

## 🧠 Model Architecture & Methodology

The model implements the standard **Pix2Pix** conditional GAN framework benchmarked on the **CMP Facades Database**:

| Component | Specification |
|---|---|
| **Framework** | Pix2Pix Conditional GAN (cGAN) |
| **Generator** | 8-level U-Net with skip connections ($C_{in}=3 \to C_{out}=3$, $N_f=64$, Tanh output) |
| **Discriminator** | $70 \times 70$ PatchGAN conditioned on the blueprint input |
| **Objective** | $\mathcal{G}^* = \arg\min_G \max_D \mathcal{L}_{cGAN}(G, D) + \lambda \cdot \mathcal{L}_{L1}(G)$, with $\lambda = 100$ |
| **Input Resolution** | $256 \times 256$ pixels, normalized to $[-1, 1]$ |
| **Interpolation** | `NEAREST` resampling on input to prevent label boundary blur |
| **Performance** | L1: `0.1847` \| PSNR: `12.56 dB` \| SSIM: `0.2288` |

---

## 🎨 CMP Facades Label Palette

The model was trained on the standardized CMP Facades dataset palette. Use **exactly** these colors in your blueprints for best results:

| Class Name | Element Description | Hex | RGB |
|---|---|---|---|
| **Background** | Sky, roof silhouette, ground context | `#0000aa` | `(0, 0, 170)` |
| **Facade (Wall)** | Main structural masonry and walls | `#0000ff` | `(0, 0, 255)` |
| **Window** | Window glazing, panes, interior shadows | `#0055ff` | `(0, 85, 255)` |
| **Door** | Building portals, entrances, doorways | `#aa0000` | `(170, 0, 0)` |
| **Cornice** | Horizontal roof overhangs and entablatures | `#ff5500` | `(255, 85, 0)` |
| **Window Sill** | Horizontal ledges below window openings | `#00ffff` | `(0, 255, 255)` |
| **Balcony** | Projecting platforms, railings, parapets | `#aaff55` | `(170, 255, 85)` |
| **Blind / Shutter** | Sunshades, window shutters, blinds | `#55ffaa` | `(85, 255, 170)` |
| **Deco** | Medallions, decorative sculptures, reliefs | `#ffff00` | `(255, 255, 0)` |
| **Molding** | Decorative horizontal trim bands | `#ffaa00` | `(255, 170, 0)` |
| **Pillar** | Vertical columns, pilasters, piers | `#ff0000` | `(255, 0, 0)` |
| **Shopfront** | Ground floor retail storefronts and vitrines | `#00aaff` | `(0, 170, 255)` |

> **Note:** These colors were empirically verified from all 378 images in the CMP Facades dataset. Inputs using different colors will produce degraded results.

---

## 📁 Repository Structure

```
Facade-Renderer/
├── app.py                      # Main Streamlit web application
├── model/
│   ├── generator.py            # Pix2Pix U-Net generator architecture
│   ├── loader.py               # Checkpoint loader with caching & stochastic dropout
│   └── facade_generator.pkl    # ⚠️  NOT in git — download from Releases (208 MB)
├── utils/
│   └── image_ops.py            # Preprocessing, postprocessing, color analytics & comparison
├── samples/                    # 378 CMP Facades blueprint PNGs + paired real-photo JPGs
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and run instructions
```

---

## 📦 Requirements

```
streamlit
torch
torchvision
Pillow
numpy
```

See `requirements.txt` for pinned versions.

---

## 📄 License

This project is for educational and research purposes. The CMP Facades dataset is © Radim Tyleček and Radim Šára (Czech Technical University in Prague).
