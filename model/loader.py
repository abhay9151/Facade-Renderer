import os
import pickle
import torch
import torch.nn as nn
import streamlit as st
from model.generator import UNetGenerator

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

    if not resolved_path:
        raise FileNotFoundError(
            f"Model checkpoint not found. Expected at '{model_path}'. "
            "Please ensure 'facade_generator.pkl' is placed in the 'model/' directory."
        )

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
