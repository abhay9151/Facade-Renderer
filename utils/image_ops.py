import io
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# CMP Facades dataset semantic classes — palette extracted empirically from all 378 images.
# These exact RGB values are what the model was trained on. Do NOT change them.
FACADE_CLASSES = [
    {
        "id": "background",
        "name": "Background",
        "rgb": (0, 0, 170),
        "hex": "#0000aa",
        "desc": "Sky, roofline silhouette, ground level context",
    },
    {
        "id": "facade",
        "name": "Facade (Wall)",
        "rgb": (0, 0, 255),
        "hex": "#0000ff",
        "desc": "Main structural wall and masonry",
    },
    {
        "id": "window",
        "name": "Window",
        "rgb": (0, 85, 255),
        "hex": "#0055ff",
        "desc": "Glazing, glass panes, interior shadows",
    },
    {
        "id": "door",
        "name": "Door",
        "rgb": (170, 0, 0),
        "hex": "#aa0000",
        "desc": "Building entrances and doorways",
    },
    {
        "id": "cornice",
        "name": "Cornice",
        "rgb": (255, 85, 0),
        "hex": "#ff5500",
        "desc": "Horizontal decorative roof overhang",
    },
    {
        "id": "sill",
        "name": "Window Sill",
        "rgb": (0, 255, 255),
        "hex": "#00ffff",
        "desc": "Horizontal ledges beneath window frames",
    },
    {
        "id": "balcony",
        "name": "Balcony",
        "rgb": (170, 255, 85),
        "hex": "#aaff55",
        "desc": "Projecting platforms and parapets",
    },
    {
        "id": "blind",
        "name": "Blind / Shutter",
        "rgb": (85, 255, 170),
        "hex": "#55ffaa",
        "desc": "Window shutters, shades, blinds",
    },
    {
        "id": "deco",
        "name": "Deco (Ornament)",
        "rgb": (255, 255, 0),
        "hex": "#ffff00",
        "desc": "Decorative ornaments and medallions",
    },
    {
        "id": "molding",
        "name": "Molding",
        "rgb": (255, 170, 0),
        "hex": "#ffaa00",
        "desc": "Architectural trim bands and stringcourses",
    },
    {
        "id": "pillar",
        "name": "Pillar / Column",
        "rgb": (255, 0, 0),
        "hex": "#ff0000",
        "desc": "Vertical support pillars and pilasters",
    },
    {
        "id": "shop",
        "name": "Shopfront",
        "rgb": (0, 170, 255),
        "hex": "#00aaff",
        "desc": "Ground floor retail storefronts and vitrines",
    },
]

def preprocess_layout(image: Image.Image, target_size: int = 256, device=None) -> tuple[torch.Tensor, Image.Image]:
    """
    Convert layout to RGB, resize to target_size using NEAREST interpolation
    to preserve exact semantic color boundaries, normalize to [-1, 1], and prepare batch tensor.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # NEAREST resampling is mandatory to avoid antialiasing blur on label colors
    resized_pil = image.resize((target_size, target_size), Image.Resampling.NEAREST)
    
    # Convert PIL to Tensor [C, H, W] in [0, 1]
    arr = np.array(resized_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) # [3, H, W]
    
    # Normalize mean=0.5, std=0.5 -> [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    
    # Add batch dimension -> [1, 3, H, W]
    batch_tensor = tensor.unsqueeze(0)
    
    if device is not None:
        batch_tensor = batch_tensor.to(device)
        
    return batch_tensor, resized_pil

def postprocess_facade(output_tensor: torch.Tensor, upscale_512: bool = False) -> Image.Image:
    """
    Postprocess generator output tensor from [-1, 1] to a [0, 255] PIL Image.
    Optionally applies high-fidelity Lanczos upscaling to 512x512.
    """
    if output_tensor.dim() == 4:
        output_tensor = output_tensor.squeeze(0)
        
    # Unnormalize: [-1, 1] -> [0, 1]
    tensor = output_tensor.detach().cpu() * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to numpy uint8 [H, W, C]
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    
    if upscale_512:
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
    return image

def create_side_by_side_comparison(
    layout_img: Image.Image, 
    facade_img: Image.Image, 
    upscale: bool = False
) -> Image.Image:
    """
    Combine input blueprint and generated facade side-by-side with subtle styling.
    """
    size = 512 if upscale else 256
    
    # Ensure both images match the size
    l_img = layout_img.resize((size, size), Image.Resampling.NEAREST)
    f_img = facade_img.resize((size, size), Image.Resampling.LANCZOS if upscale else Image.Resampling.BILINEAR)
    
    divider_width = 8
    border = 16
    total_w = size * 2 + divider_width + border * 2
    header_h = 44
    total_h = size + header_h + border * 2
    
    # Deep architectural charcoal canvas
    canvas = Image.new("RGB", (total_w, total_h), (18, 22, 28))
    draw = ImageDraw.Draw(canvas)
    
    # Place layout and facade
    x_layout = border
    y_img = border + header_h
    canvas.paste(l_img, (x_layout, y_img))
    
    x_facade = border + size + divider_width
    canvas.paste(f_img, (x_facade, y_img))
    
    # Elegant divider line
    div_x = border + size + (divider_width // 2)
    draw.line([(div_x, y_img), (div_x, y_img + size)], fill=(45, 55, 72), width=2)
    
    # Labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    # Text headers
    draw.text((x_layout + 10, border + 14), "INPUT BLUEPRINT (2D LAYOUT)", fill=(203, 213, 225), font=font)
    draw.text((x_facade + 10, border + 14), "SYNTHESIZED FACADE (PIX2PIX cGAN)", fill=(245, 158, 11), font=font)
    
    return canvas

def create_triplet_comparison(
    layout_img: Image.Image,
    facade_img: Image.Image,
    real_img: Image.Image,
    upscale: bool = False
) -> Image.Image:
    """
    Combine blueprint, generated facade, and real photo side-by-side (Evaluation Triplet).
    """
    size = 512 if upscale else 256
    l_img = layout_img.resize((size, size), Image.Resampling.NEAREST)
    f_img = facade_img.resize((size, size), Image.Resampling.LANCZOS if upscale else Image.Resampling.BILINEAR)
    r_img = real_img.resize((size, size), Image.Resampling.LANCZOS if upscale else Image.Resampling.BILINEAR)

    divider_width = 8
    border = 16
    total_w = size * 3 + divider_width * 2 + border * 2
    header_h = 44
    total_h = size + header_h + border * 2

    canvas = Image.new("RGB", (total_w, total_h), (18, 22, 28))
    draw = ImageDraw.Draw(canvas)

    y_img = border + header_h
    x0 = border
    x1 = border + size + divider_width
    x2 = border + (size + divider_width) * 2

    canvas.paste(l_img, (x0, y_img))
    canvas.paste(f_img, (x1, y_img))
    canvas.paste(r_img, (x2, y_img))

    # Dividers
    draw.line([(x1 - divider_width // 2, y_img), (x1 - divider_width // 2, y_img + size)], fill=(45, 55, 72), width=2)
    draw.line([(x2 - divider_width // 2, y_img), (x2 - divider_width // 2, y_img + size)], fill=(45, 55, 72), width=2)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((x0 + 10, border + 14), "1. INPUT BLUEPRINT", fill=(203, 213, 225), font=font)
    draw.text((x1 + 10, border + 14), "2. SYNTHESIZED (PIX2PIX)", fill=(245, 158, 11), font=font)
    draw.text((x2 + 10, border + 14), "3. REAL GROUND TRUTH", fill=(74, 222, 128), font=font)

    return canvas

def analyze_layout_colors(image: Image.Image) -> list[dict]:
    """
    Examine an input layout image to identify which CMP Facades classes are present
    and measure coverage percentage.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    small = image.resize((128, 128), Image.Resampling.NEAREST)
    arr = np.array(small, dtype=np.float32)
    pixels = arr.reshape(-1, 3)
    total_pixels = len(pixels)
    
    results = []
    for cls in FACADE_CLASSES:
        target_rgb = np.array(cls["rgb"], dtype=np.float32)
        # Compute Euclidean distance in RGB color space
        distances = np.linalg.norm(pixels - target_rgb, axis=1)
        # Match within threshold of 40 to tolerate JPEG compression artifacts
        matches = np.sum(distances < 40)
        coverage_pct = (matches / total_pixels) * 100.0
        if coverage_pct >= 0.2:  # Found in layout
            results.append({
                "id": cls["id"],
                "name": cls["name"],
                "hex": cls["hex"],
                "coverage": coverage_pct,
                "desc": cls["desc"],
            })
            
    # Sort by coverage descending
    results.sort(key=lambda x: x["coverage"], reverse=True)
    return results

def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to downloadable bytes buffer."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()
