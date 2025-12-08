import gc
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision.transforms.functional import normalize
from diffusers import StableDiffusionInpaintPipeline
import os
from pathlib import Path

# 全局变量缓存
_SEG_MODEL = None
_PIPE = None

# 缓存路径
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "models"


def load_seg_model(device):
    global _SEG_MODEL
    if _SEG_MODEL is None:
        print("✨ [Generator] Loading Segmentation Model (RMBG-1.4)...")
        from transformers import AutoModelForImageSegmentation
        try:
            _SEG_MODEL = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-1.4",
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            _SEG_MODEL.to(device)
            _SEG_MODEL.eval()
        except Exception as e:
            print(f"❌ Failed to load RMBG-1.4: {e}")
            raise e
    return _SEG_MODEL


def load_inpainting_pipeline(device):
    global _PIPE
    if _PIPE is None:
        print("✨ [Generator] Loading Stable Diffusion Inpainting...")
        from diffusers import StableDiffusionInpaintPipeline
        try:
            _PIPE = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=CACHE_DIR
            )
            _PIPE.to(device)
            _PIPE.enable_model_cpu_offload()
        except Exception as e:
            print(f"❌ Failed to load SD Inpainting: {e}")
            raise e
    return _PIPE


def preprocess_image(image_pil: Image.Image, model_input_size: tuple) -> torch.Tensor:
    """Preprocess image matching reference logic"""
    image_resized = image_pil.resize(model_input_size, Image.Resampling.BILINEAR)
    im_arr = np.array(image_resized)

    # Handle Grayscale images
    if len(im_arr.shape) == 2:
        im_arr = np.stack([im_arr] * 3, axis=-1)

    im_tensor = torch.tensor(im_arr, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = torch.unsqueeze(im_tensor, 0)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image


def postprocess_image(result: torch.Tensor, orig_size: tuple) -> Image.Image:
    """Postprocess result matching reference logic"""
    result = F.interpolate(result, size=orig_size[::-1], mode='bilinear')
    result = torch.squeeze(result, 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return Image.fromarray(im_array)


def generate_background_ai(image_pil: Image.Image,
                           prompt: str = "background, water, lake, ripples, nature, high quality, realistic",
                           device="cuda") -> tuple[Image.Image, Image.Image]:
    """
    生成AI背景并返回主体mask
    返回: (背景图像, 主体mask)
    """
    try:
        # 强制转RGB
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # === 1. 分割 (RMBG-1.4) ===
        seg_model = load_seg_model(device)
        orig_w, orig_h = image_pil.size

        input_tensor = preprocess_image(image_pil, (1024, 1024)).to(device)

        with torch.no_grad():
            outputs = seg_model(input_tensor)
            # 提取 Tensor [Reference Logic]
            if isinstance(outputs, (list, tuple)):
                preds = outputs[0][0]
            else:
                preds = outputs

        # 后处理
        mask = postprocess_image(preds, (orig_w, orig_h))

        # 二值化
        mask = mask.point(lambda p: 255 if p > 10 else 0).convert("L")

        # === [新增] 面积阈值检测 ===
        mask_arr = np.array(mask)
        # 计算白色像素占比 (255)
        coverage_ratio = np.sum(mask_arr > 128) / mask_arr.size

        print(f"🐛 [Debug] Mask coverage: {coverage_ratio:.2%} (Threshold: 30%)")

        # 保存原始mask（用于shader）
        subject_mask = mask.copy()
        
        # 如果主体超过 30%，认为遮挡太多，不适合 Inpainting
        if coverage_ratio > 0.30:
            print(f"⚠️ [Generator] Subject is too large ({coverage_ratio:.1%}). Skipping Inpainting.")
            # 返回原图作为背景 (图层重叠)
            return image_pil, subject_mask

        # === 继续处理 ===
        # 保存 Mask
        mask.save("debug_mask.png")

        # 膨胀 Mask
        mask = mask.filter(ImageFilter.MaxFilter(25))

        # === 2. 修复 (Stable Diffusion) ===
        pipe = load_inpainting_pipeline(device)

        def align_8(x):
            return x - (x % 8)

        sd_w, sd_h = align_8(orig_w), align_8(orig_h)

        sd_in_img = image_pil.resize((sd_w, sd_h), Image.Resampling.LANCZOS)
        sd_in_mask = mask.resize((sd_w, sd_h), Image.Resampling.NEAREST)

        result = pipe(
            prompt=prompt,
            negative_prompt="foreground object, person, animal, text, watermark, ugly, distorted, low quality, glitch, wall, concrete, ground",
            image=sd_in_img,
            mask_image=sd_in_mask,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        result = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        return result, subject_mask

    except Exception as e:
        print(f"❌ AI Generation Error: {e}")
        import traceback
        traceback.print_exc()
        # 返回黑色背景和空mask
        empty_mask = Image.new("L", image_pil.size, 0)
        return Image.new("RGB", image_pil.size, (0, 0, 0)), empty_mask