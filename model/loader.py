import os
import pickle
import torch
import torch.nn as nn
import streamlit as st
from model.generator import UNetGenerator

# ---------------------------------------------------------------------------
# GitHub Release URL for the model weights (used for Streamlit Cloud deploy)
# ---------------------------------------------------------------------------
MODEL_RELEASE_URL = (
    "https://github.com/abhay9151/Facade-Renderer/releases/download/v1.0/facade_generator.pkl"
)

def _download_model(url: str, dest: str):
    """Download a file from *url* to *dest* with a Streamlit progress bar."""
    import urllib.request, shutil

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_path = dest + ".part"

    st.info("⬇️ Downloading model weights from GitHub Releases (~208 MB)…")
    progress_bar = st.progress(0, text="Starting download…")

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size, 1.0)
            progress_bar.progress(pct, text=f"Downloaded {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB")

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        shutil.move(tmp_path, dest)
        progress_bar.progress(1.0, text="Download complete ✅")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to download model from {url}: {e}\n"
            "Please download 'facade_generator.pkl' manually and place it in the 'model/' directory."
        )


def get_device():
    """Detect available accelerator: CUDA -> Apple Silicon MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@st.cache_resource(show_spinner=False)
def load_generator_model(model_path="model/facade_generator.pkl"):
    """
    Load Pix2Pix UNetGenerator from pickle checkpoint with caching.
    If the checkpoint is not found locally, it is automatically downloaded
    from the GitHub Releases page (for Streamlit Cloud deployment).
    Returns:
        model: UNetGenerator instance on active device
        metadata: dict with training epochs, input_size, test_metrics, etc.
        device: torch.device
    """
    # Check alternate paths if run from different working directories
    possible_paths = [
        model_path,
        os.path.join(os.path.dirname(__file__), "facade_generator.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "facade_generator.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "model", "facade_generator.pkl"),
    ]
    
    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    # Auto-download when running on Streamlit Cloud (model not in repo)
    if not resolved_path:
        default_dest = os.path.join(os.path.dirname(__file__), "facade_generator.pkl")
        _download_model(MODEL_RELEASE_URL, default_dest)
        resolved_path = default_dest

    try:
        with open(resolved_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file at '{resolved_path}': {str(e)}")

    if not isinstance(data, dict) or "state_dict" not in data:
        raise ValueError(
            "Invalid checkpoint structure: expected dictionary containing 'state_dict'."
        )

    device = get_device()
    model = UNetGenerator(in_ch=3, out_ch=3, nf=64).to(device)
    
    try:
        model.load_state_dict(data["state_dict"])
    except Exception as e:
        raise RuntimeError(f"State dict mismatch when loading generator weights: {str(e)}")

    model.eval()

    metadata = {
        "model": data.get("model", "UNetGenerator (Pix2Pix)"),
        "input_size": data.get("input_size", 256),
        "normalization": data.get("normalization", "mean=0.5, std=0.5 ([-1, 1])"),
        "epochs": data.get("epochs", 30),
        "test_metrics": data.get("test_metrics", {"L1": 0.1847, "PSNR": 12.56, "SSIM": 0.2288}),
        "checkpoint_path": resolved_path,
        "device_name": str(device).upper(),
    }

    return model, metadata, device

def set_dropout_eval_mode(model, enable_dropout=False):
    """
    Configure model for deterministic evaluation or stochastic variations.
    When enable_dropout=True, keeps BatchNorm in eval mode while setting
    Dropout layers to train mode to generate varied facade textures.
    """
    model.eval()
    if enable_dropout:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
