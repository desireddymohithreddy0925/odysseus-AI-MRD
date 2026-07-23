"""Image generation model registry and VRAM fitting for Cookbook."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

# Curated registry of image generation models supported by diffusers.
# ONLY verified HuggingFace repo IDs.
# VRAM estimates are for inference (single image generation).
IMAGE_MODEL_REGISTRY = [
    # ── Z-Image (Alibaba Tongyi) ──
    {
        "id": "Tongyi-MAI/Z-Image-Turbo",
        "name": "Z-Image Turbo",
        "provider": "Tongyi",
        "params_b": 6.0,
        "vram_bf16": 19.0,
        "vram_fp8": 10.0,
        "vram_q4": 6.0,
        "default_quant": "BF16",
        "quant_repos": {
            "FP8": "drbaph/Z-Image-Turbo-FP8",
        },
        "capabilities": ["text-to-image"],
        "description": "6B distilled, 8-step. Sub-second on H800. Apache 2.0.",
        "quality": 92,
        "speed": 95,
        "released": "2025-12",
    },
    {
        "id": "Tongyi-MAI/Z-Image",
        "name": "Z-Image",
        "provider": "Tongyi",
        "params_b": 6.0,
        "vram_bf16": 19.0,
        "vram_fp8": 10.0,
        "vram_q4": 6.0,
        "default_quant": "BF16",
        "quant_repos": {
            "FP8": "drbaph/Z-Image-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "Full undistilled model. Highest creative freedom. Apache 2.0.",
        "quality": 93,
        "speed": 70,
        "released": "2025-12",
    },
    # ── Qwen Image ──
    {
        "id": "Qwen/Qwen-Image-2512",
        "name": "Qwen Image 2512",
        "provider": "Qwen",
        "params_b": 20.0,
        "vram_bf16": 42.0,
        "vram_fp8": 22.0,
        "vram_q4": 14.0,
        "default_quant": "FP8",
        "quant_repos": {},
        "capabilities": ["text-to-image", "text-rendering"],
        "description": "Dec 2025 update. Better humans, finer detail, strong text. Apache 2.0.",
        "quality": 95,
        "speed": 50,
        "released": "2025-12",
    },
    {
        "id": "Qwen/Qwen-Image",
        "name": "Qwen Image",
        "provider": "Qwen",
        "params_b": 20.0,
        "vram_bf16": 42.0,
        "vram_fp8": 22.0,
        "vram_q4": 14.0,
        "default_quant": "FP8",
        "quant_repos": {},
        "capabilities": ["text-to-image", "text-rendering"],
        "description": "20B foundation. Best text rendering in images. Apache 2.0.",
        "quality": 94,
        "speed": 50,
        "released": "2025-08",
    },
    {
        "id": "Qwen/Qwen-Image-Edit-2511",
        "name": "Qwen Image Edit",
        "provider": "Qwen",
        "params_b": 20.0,
        "vram_bf16": 42.0,
        "vram_fp8": 22.0,
        "vram_q4": 14.0,
        "default_quant": "FP8",
        "quant_repos": {},
        "capabilities": ["image-editing", "inpainting"],
        "description": "Dedicated editing. Style transfer, object removal. Apache 2.0.",
        "quality": 92,
        "speed": 50,
        "released": "2025-11",
    },
    # -- Apple MLX utility/edit models --
    {
        "id": "mlx-community/DDColor-modelscope-fp16",
        "name": "DDColor ModelScope",
        "provider": "mlx-community",
        "params_b": 0.2,
        "vram_bf16": 1.2,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["colorize", "image-to-image"],
        "description": "Apple MLX DDColor automatic colorization default tier. Requires the mlx-ddcolor-swift runtime.",
        "quality": 78,
        "speed": 90,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_ddcolor_swift",
    },
    {
        "id": "mlx-community/DDColor-paper-tiny-fp16",
        "name": "DDColor Tiny",
        "provider": "mlx-community",
        "params_b": 0.055,
        "vram_bf16": 0.8,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["colorize", "image-to-image"],
        "description": "Fast Apple MLX DDColor colorization tier. Requires the mlx-ddcolor-swift runtime.",
        "quality": 70,
        "speed": 98,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_ddcolor_swift",
    },
    {
        "id": "mlx-community/DDColor-artistic-fp16",
        "name": "DDColor Artistic",
        "provider": "mlx-community",
        "params_b": 0.2,
        "vram_bf16": 1.2,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["colorize", "image-to-image"],
        "description": "Stylized Apple MLX DDColor colorization with a more saturated palette. Requires the mlx-ddcolor-swift runtime.",
        "quality": 76,
        "speed": 88,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_ddcolor_swift",
    },
    {
        "id": "mlx-community/LaMa-bf16",
        "name": "LaMa Inpainting",
        "provider": "mlx-community",
        "params_b": 0.051,
        "vram_bf16": 1.0,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {},
        "capabilities": ["inpainting", "object-removal", "image-to-image"],
        "description": "Apple MLX Big-LaMa inpainting quality tier. Uses image + white mask. Requires the mlx-lama-swift runtime.",
        "quality": 74,
        "speed": 90,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_lama_swift",
    },
    {
        "id": "mlx-community/MI-GAN-512-places2-fp16",
        "name": "MI-GAN 512 Inpainting",
        "provider": "mlx-community",
        "params_b": 0.007,
        "vram_bf16": 0.6,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["inpainting", "object-removal", "image-to-image"],
        "description": "Fast Apple MLX MI-GAN 512 Places2 inpainting. Uses image + white mask. Requires the mlx-lama-swift runtime.",
        "quality": 64,
        "speed": 98,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_lama_swift",
    },
    {
        "id": "mlx-community/MI-GAN-256-places2-fp16",
        "name": "MI-GAN 256 Places2",
        "provider": "mlx-community",
        "params_b": 0.006,
        "vram_bf16": 0.5,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["inpainting", "object-removal", "image-to-image"],
        "description": "Very fast Apple MLX MI-GAN 256 Places2 inpainting. Requires the mlx-lama-swift runtime.",
        "quality": 58,
        "speed": 99,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_lama_swift",
    },
    {
        "id": "mlx-community/MI-GAN-256-ffhq-fp16",
        "name": "MI-GAN 256 FFHQ",
        "provider": "mlx-community",
        "params_b": 0.006,
        "vram_bf16": 0.5,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "FP16",
        "quant_repos": {},
        "capabilities": ["inpainting", "object-removal", "face-editing", "image-to-image"],
        "description": "Very fast Apple MLX MI-GAN 256 face inpainting. Requires the mlx-lama-swift runtime.",
        "quality": 58,
        "speed": 99,
        "released": "2026-06",
        "mlx_only": True,
        "runtime": "mlx_lama_swift",
    },
    # ── Stable Diffusion (dedicated inpainting) ──
    {
        "id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "name": "SDXL Inpainting",
        "provider": "Stability AI",
        "params_b": 3.5,
        "vram_bf16": 12.0,
        "vram_fp8": 8.0,
        "vram_q4": 6.0,
        "default_quant": "BF16",
        "quant_repos": {},
        "capabilities": ["inpainting", "image-editing"],
        "description": "SDXL fine-tuned for inpainting (9-channel UNet). Best SD-family fill quality; fits a 24GB card comfortably.",
        "quality": 86,
        "speed": 68,
        "released": "2023-11",
    },
    {
        "id": "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "name": "SD 1.5 Inpainting",
        "provider": "Stability AI",
        "params_b": 1.1,
        "vram_bf16": 4.0,
        "vram_fp8": 3.0,
        "vram_q4": 2.5,
        "default_quant": "BF16",
        "quant_repos": {},
        "capabilities": ["inpainting"],
        "description": "Classic SD 1.5 inpaint. Very light and fast; lower fidelity than SDXL.",
        "quality": 70,
        "speed": 92,
        "released": "2022-10",
    },
    # ── FLUX ──
    {
        "id": "black-forest-labs/FLUX.1-dev",
        "name": "FLUX.1 Dev",
        "provider": "Black Forest Labs",
        "params_b": 12.0,
        "vram_bf16": 33.0,
        "vram_fp8": 17.0,
        "vram_q4": 10.0,
        "default_quant": "FP8",
        "quant_repos": {
            "FP8": "diffusers/FLUX.1-dev-torchao-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "High quality, detailed. Popular community model. Non-commercial.",
        "quality": 92,
        "speed": 55,
        "released": "2024-08",
    },
    {
        "id": "black-forest-labs/FLUX.1-schnell",
        "name": "FLUX.1 Schnell",
        "provider": "Black Forest Labs",
        "params_b": 12.0,
        "vram_bf16": 33.0,
        "vram_fp8": 17.0,
        "vram_q4": 10.0,
        "default_quant": "FP8",
        "quant_repos": {
            "FP8": "Kijai/flux-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "Fast 4-step variant. Apache 2.0 license.",
        "quality": 85,
        "speed": 90,
        "released": "2024-08",
    },
    # ── Stable Diffusion ──
    {
        "id": "stabilityai/stable-diffusion-3.5-medium",
        "name": "SD 3.5 Medium",
        "provider": "Stability AI",
        "params_b": 2.5,
        "vram_bf16": 12.0,
        "vram_fp8": 7.0,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {
            "FP8": "Comfy-Org/stable-diffusion-3.5-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "2.5B lightweight, fast. Fits almost any GPU.",
        "quality": 75,
        "speed": 95,
        "released": "2024-10",
    },
    {
        "id": "stabilityai/stable-diffusion-3.5-large",
        "name": "SD 3.5 Large",
        "provider": "Stability AI",
        "params_b": 8.1,
        "vram_bf16": 22.0,
        "vram_fp8": 12.0,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {
            "FP8": "Comfy-Org/stable-diffusion-3.5-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "8B high quality. Good balance of speed and quality.",
        "quality": 85,
        "speed": 70,
        "released": "2024-10",
    },
    {
        "id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "name": "SD 3.5 Large Turbo",
        "provider": "Stability AI",
        "params_b": 8.1,
        "vram_bf16": 22.0,
        "vram_fp8": 12.0,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {
            "FP8": "Comfy-Org/stable-diffusion-3.5-fp8",
        },
        "capabilities": ["text-to-image"],
        "description": "Distilled for few-step inference. Fastest large SD.",
        "quality": 80,
        "speed": 92,
        "released": "2024-10",
    },
    {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "name": "SDXL",
        "provider": "Stability AI",
        "params_b": 3.5,
        "vram_bf16": 10.0,
        "vram_fp8": 6.0,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {},
        "capabilities": ["text-to-image"],
        "description": "Classic workhorse. Huge LoRA ecosystem. Fits 8GB+.",
        "quality": 72,
        "speed": 90,
        "released": "2023-07",
    },
    # ── Hunyuan ──
    {
        "id": "tencent/HunyuanImage-3.0",
        "name": "HunyuanImage 3.0",
        "provider": "Tencent",
        "params_b": 13.0,
        "vram_bf16": 30.0,
        "vram_fp8": 16.0,
        "vram_q4": 9.0,
        "default_quant": "FP8",
        "quant_repos": {
            "Q4": "wikeeyang/Hunyuan-Image-30-Qint4",
            "NF4": "EricRollei/HunyuanImage-3.0-Instruct-NF4",
        },
        "capabilities": ["text-to-image", "text-rendering"],
        "description": "Strong text rendering. Bilingual Chinese/English. 13B activated per token.",
        "quality": 88,
        "speed": 60,
        "released": "2025-09",
    },
    {
        "id": "tencent/HunyuanImage-3.0-Instruct-Distil",
        "name": "HunyuanImage 3.0 Distil",
        "provider": "Tencent",
        "params_b": 13.0,
        "vram_bf16": 30.0,
        "vram_fp8": 16.0,
        "vram_q4": 9.0,
        "default_quant": "FP8",
        "quant_repos": {},
        "capabilities": ["text-to-image", "text-rendering"],
        "description": "Distilled variant, fewer steps. Faster with comparable quality.",
        "quality": 85,
        "speed": 80,
        "released": "2026-01",
    },
]

HF_IMAGE_COLLECTIONS = [
    "stabilityai/image",
    "stabilityai/stable-diffusion-35",
    "black-forest-labs/flux2",
]

HF_MLX_IMAGE_COLLECTIONS = [
    "mlx-community/flux2-klein-mlx",
    "mlx-community/inpainting-mlx",
    "mlx-community/ddcolor-mlx",
    "mlx-community/boogu-image-01-mlx",
]

HF_MLX_IMAGE_REPO_SEEDS: list[str] = []
HF_IMAGE_REPO_SEEDS: list[str] = []

_HF_COLLECTION_CACHE = {"ts": 0.0, "models": []}
_HF_COLLECTION_TTL = 30 * 60
_HF_VARIANT_CACHE: dict[str, dict[str, str]] = {}
_HF_SEARCH_DISABLED_UNTIL = 0.0


def _repo_display_name(repo_id: str) -> str:
    name = str(repo_id or "").split("/")[-1]
    return name.replace("-", " ").replace("_", " ").strip() or repo_id


def _provider_from_repo(repo_id: str) -> str:
    owner = str(repo_id or "").split("/", 1)[0].lower()
    return {
        "stabilityai": "Stability AI",
        "black-forest-labs": "Black Forest Labs",
        "tongyi-mai": "Tongyi",
        "qwen": "Qwen",
        "mlx-community": "mlx-community",
    }.get(owner, owner.replace("-", " ").title() if owner else "HuggingFace")


def _infer_capabilities(item: dict[str, Any], repo_id: str) -> list[str]:
    tasks = set()
    pipeline = str(item.get("pipeline_tag") or "").strip().lower()
    if pipeline:
        tasks.add(pipeline)
    for provider in item.get("availableInferenceProviders") or []:
        if isinstance(provider, dict) and provider.get("task"):
            tasks.add(str(provider["task"]).strip().lower())
    text = f"{repo_id} {' '.join(tasks)}".lower()
    caps = []
    if "image-to-image" in tasks or "edit" in text or "inpaint" in text:
        caps.append("image-editing")
    if "inpaint" in text:
        caps.append("inpainting")
    if "text-to-image" in tasks or not caps:
        caps.append("text-to-image")
    return caps


def _estimate_image_model(repo_id: str) -> dict[str, Any]:
    text = str(repo_id or "").lower()
    if any(k in text for k in ("mi-gan", "big-lama", "lama-")):
        return {"params_b": 0.01, "bf16": 1.0, "fp8": 0.7, "q4": 0.5, "quality": 65, "speed": 98, "quant": "BF16"}
    if "qwen-image-2512" in text:
        return {"params_b": 20.0, "bf16": 42.0, "fp8": 34.0, "q4": 18.0, "quality": 95, "speed": 46, "quant": "FP8"}
    if "hidream-o1-image-dev" in text:
        return {"params_b": 17.0, "bf16": 36.0, "fp8": 22.0, "q4": 14.0, "quality": 90, "speed": 48, "quant": "BF16"}
    if "flux.2-klein-9b" in text or "flux2-klein-9b" in text:
        return {"params_b": 9.0, "bf16": 29.0, "fp8": 18.0, "q4": 11.0, "quality": 88, "speed": 60, "quant": "BF16"}
    if "flux.2-klein" in text or "flux2-klein" in text:
        return {"params_b": 4.0, "bf16": 10.0, "fp8": 6.0, "q4": 3.0, "quality": 82, "speed": 88, "quant": "BF16"}
    if "boogu-image" in text:
        return {"params_b": 4.0, "bf16": 10.0, "fp8": 6.0, "q4": 3.0, "quality": 76, "speed": 82, "quant": "Q4" if "4bit" in text else "BF16"}
    if "krea-2" in text:
        return {"params_b": 14.0, "bf16": 32.0, "fp8": 17.0, "q4": 10.0, "quality": 94, "speed": 88, "quant": "FP8"}
    if "qwen-image" in text:
        return {"params_b": 20.0, "bf16": 42.0, "fp8": 22.0, "q4": 14.0, "quality": 92, "speed": 50, "quant": "FP8"}
    if "z-image" in text:
        return {"params_b": 6.0, "bf16": 19.0, "fp8": 10.0, "q4": 6.0, "quality": 92, "speed": 90, "quant": "BF16"}
    if "flux.2" in text or "flux2" in text:
        return {"params_b": 32.0, "bf16": 64.0, "fp8": 34.0, "q4": 20.0, "quality": 96, "speed": 45, "quant": "FP8"}
    if "flux" in text:
        return {"params_b": 12.0, "bf16": 33.0, "fp8": 17.0, "q4": 10.0, "quality": 90, "speed": 60, "quant": "FP8"}
    if "stable-diffusion-3.5-large" in text:
        return {"params_b": 8.1, "bf16": 22.0, "fp8": 12.0, "q4": None, "quality": 84, "speed": 74, "quant": "BF16"}
    if "stable-diffusion-3.5-medium" in text or "stable-diffusion-3-medium" in text:
        return {"params_b": 2.5, "bf16": 12.0, "fp8": 7.0, "q4": None, "quality": 76, "speed": 94, "quant": "BF16"}
    if "stable-diffusion-xl" in text or "sdxl" in text:
        return {"params_b": 3.5, "bf16": 10.0, "fp8": 6.0, "q4": None, "quality": 72, "speed": 90, "quant": "BF16"}
    if "stable-diffusion" in text:
        return {"params_b": 1.5, "bf16": 6.0, "fp8": 4.0, "q4": None, "quality": 68, "speed": 92, "quant": "BF16"}
    return {"params_b": 8.0, "bf16": 24.0, "fp8": 13.0, "q4": 8.0, "quality": 78, "speed": 60, "quant": "BF16"}


def _params_b_from_item(item: dict[str, Any]) -> float | None:
    raw = item.get("numParameters")
    if isinstance(raw, (int, float)) and raw > 0:
        return max(0.01, round(float(raw) / 1_000_000_000.0, 3))
    return None


def _mlx_quantize_estimate(repo_id: str, est: dict[str, Any]) -> dict[str, Any]:
    text = str(repo_id or "").lower()
    out = dict(est)
    if "3bit" in text or "4bit" in text or "q4" in text:
        out["quant"] = "Q4"
        out["bf16"] = None
        out["fp8"] = None
    elif "8bit" in text:
        out["quant"] = "FP8"
        out["bf16"] = None
    elif "6bit" in text or "5bit" in text:
        out["quant"] = "Q4"
        out["bf16"] = None
        out["fp8"] = out.get("fp8") or out.get("q4")
    elif "bf16" in text or "fp16" in text:
        out["quant"] = "BF16"
        out["fp8"] = None
        out["q4"] = None
    return out


def _collection_item_to_model(item: dict[str, Any], collection_title: str = "", mlx_only: bool = False) -> dict[str, Any] | None:
    repo_id = str(item.get("id") or "").strip()
    if "/" not in repo_id:
        return None
    typ = str(item.get("type") or item.get("itemType") or "model").lower()
    if typ not in {"", "model"}:
        return None
    est = _estimate_image_model(repo_id)
    item_params_b = _params_b_from_item(item)
    if item_params_b is not None:
        est = {
            **est,
            "params_b": item_params_b,
            "bf16": max(0.5, round(item_params_b * 2.4 + 0.8, 1)),
            "fp8": max(0.5, round(item_params_b * 1.3 + 0.5, 1)),
            "q4": max(0.4, round(item_params_b * 0.8 + 0.4, 1)),
        }
    if mlx_only:
        est = _mlx_quantize_estimate(repo_id, est)
    caps = _infer_capabilities(item, repo_id)
    gated = item.get("gated")
    desc_bits = []
    if collection_title:
        desc_bits.append(f"HF collection: {collection_title}.")
    if gated:
        desc_bits.append("Gated on HuggingFace.")
    out = {
        "id": repo_id,
        "name": _repo_display_name(repo_id),
        "provider": _provider_from_repo(repo_id),
        "params_b": est["params_b"],
        "vram_bf16": est["bf16"],
        "vram_fp8": est["fp8"],
        "vram_q4": est["q4"],
        "default_quant": est["quant"],
        "quant_repos": {},
        "capabilities": caps,
        "description": " ".join(desc_bits).strip() or "Imported from HuggingFace collection.",
        "quality": est["quality"],
        "speed": est["speed"],
        "released": "",
    }
    if mlx_only:
        out["mlx_only"] = True
        out["description"] = (out["description"] + " Apple Silicon / MLX only.").strip()
    return out


def _fetch_hf_image_collection_models() -> list[dict[str, Any]]:
    now = time.time()
    if now - float(_HF_COLLECTION_CACHE.get("ts") or 0) < _HF_COLLECTION_TTL:
        return list(_HF_COLLECTION_CACHE.get("models") or [])
    models: list[dict[str, Any]] = []
    for slug, mlx_only in [(slug, False) for slug in HF_IMAGE_COLLECTIONS] + [(slug, True) for slug in HF_MLX_IMAGE_COLLECTIONS]:
        url = f"https://huggingface.co/api/collections/{slug}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Odysseus-Cookbook/1.0"})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            continue
        title = str(data.get("title") or slug)
        for item in data.get("items") or []:
            if isinstance(item, dict):
                model = _collection_item_to_model(item, title, mlx_only=mlx_only)
                if model:
                    models.append(model)
    _HF_COLLECTION_CACHE["ts"] = now
    _HF_COLLECTION_CACHE["models"] = models
    return list(models)


def _hf_model_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    global _HF_SEARCH_DISABLED_UNTIL
    now = time.time()
    if now < _HF_SEARCH_DISABLED_UNTIL:
        return []
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode({
        "search": query,
        "limit": str(limit),
    })
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Odysseus-Cookbook/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, list) else []
    except Exception:
        _HF_SEARCH_DISABLED_UNTIL = now + 10 * 60
        return []


def _variant_score(candidate: dict[str, Any], base_repo: str, want: str) -> float:
    rid = str(candidate.get("id") or candidate.get("modelId") or "")
    text = " ".join([
        rid,
        str(candidate.get("library_name") or ""),
        str(candidate.get("pipeline_tag") or ""),
        " ".join(str(t) for t in candidate.get("tags") or []),
    ]).lower()
    base = base_repo.lower()
    base_short = base_repo.rsplit("/", 1)[-1].lower()
    if want == "gguf" and "gguf" not in text:
        return -1
    if want == "fp8" and not any(k in text for k in ("fp8", "nvfp4", "mxfp8", "mxfp4")):
        return -1
    score = float(candidate.get("downloads") or 0) / 1000.0 + float(candidate.get("likes") or 0)
    if f"base_model:{base}" in text or f"base_model:quantized:{base}" in text:
        score += 10000
    elif base_short and base_short in rid.lower():
        score += 1000
    else:
        score -= 200
    if "diffusers" in text:
        score += 50
    if str(candidate.get("private")).lower() == "true":
        score -= 10000
    return score


def _best_variant_repo(base_repo: str, want: str) -> str:
    base_short = str(base_repo or "").rsplit("/", 1)[-1]
    candidates = _hf_model_search(f"{base_short} {want}", limit=12)
    scored = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id") or item.get("modelId") or "").strip()
        if "/" not in rid or rid.lower() == base_repo.lower():
            continue
        score = _variant_score(item, base_repo, want)
        if score >= 0:
            scored.append((score, rid))
    scored.sort(reverse=True)
    return scored[0][1] if scored else ""


def _should_discover_variants(repo_id: str) -> bool:
    text = str(repo_id or "").lower()
    return any(k in text for k in (
        "krea-2",
        "qwen-image",
        "z-image",
        "flux.2",
        "flux2",
        "stable-diffusion-3.5",
    ))


def _discover_quant_repos(repo_id: str, need_fp8: bool = True, need_gguf: bool = True) -> dict[str, str]:
    key = str(repo_id or "").strip()
    if not key:
        return {}
    cache_key = f"{key.lower()}|fp8={int(need_fp8)}|gguf={int(need_gguf)}"
    if cache_key in _HF_VARIANT_CACHE:
        return dict(_HF_VARIANT_CACHE[cache_key])
    found: dict[str, str] = {}
    if need_fp8:
        fp8 = _best_variant_repo(key, "fp8")
        if fp8:
            found["FP8"] = fp8
    if need_gguf:
        gguf = _best_variant_repo(key, "gguf")
        if gguf:
            # The image-model fitter's smallest bucket is Q4; most HF image GGUF
            # repos expose Q4/Q5/Q8 files under one repo, so use it as the low-VRAM
            # download source while preserving the explicit GGUF label for callers.
            found["Q4"] = gguf
            found["GGUF"] = gguf
    _HF_VARIANT_CACHE[cache_key] = found
    return dict(found)


def _merge_quant_repos(model: dict[str, Any]) -> dict[str, Any]:
    out = dict(model)
    existing = dict(out.get("quant_repos") or {})
    repo_id = str(out.get("id") or "")
    if _should_discover_variants(repo_id):
        discovered = _discover_quant_repos(
            repo_id,
            need_fp8="FP8" not in existing,
            need_gguf="Q4" not in existing and "GGUF" not in existing,
        )
        for k, v in discovered.items():
            existing.setdefault(k, v)
    out["quant_repos"] = existing
    return out


def get_image_models():
    """Return the image model registry."""
    merged = [_merge_quant_repos(m) for m in IMAGE_MODEL_REGISTRY]
    seen = {str(m.get("id") or "").lower() for m in merged if isinstance(m, dict)}
    for model in _fetch_hf_image_collection_models():
        key = str(model.get("id") or "").lower()
        if key and key not in seen:
            merged.append(_merge_quant_repos(model))
            seen.add(key)
    return merged


def _is_apple_image_system(system: dict[str, Any]) -> bool:
    backend = str(system.get("backend") or "").lower()
    gpu_name = str(system.get("gpu_name") or "").lower()
    cpu_name = str(system.get("cpu_name") or "").lower()
    platform = str(system.get("platform") or "").lower()
    return (
        bool(system.get("unified_memory"))
        or backend in {"metal", "mps", "apple"}
        or "apple" in gpu_name
        or "apple" in cpu_name
        or platform == "darwin"
    )


def rank_image_models(system, search=None, sort="fit"):
    """Score and rank image models against detected hardware.

    Returns list of models with fit info (vram needed, fits, recommended quant).
    """
    if not isinstance(system, dict):
        system = {}
    gpu_vram = system.get("gpu_vram_gb", 0) or 0
    has_gpu = system.get("has_gpu", False)
    ram_gb = system.get("available_ram_gb") or system.get("total_ram_gb") or 0
    budget_gb = gpu_vram if has_gpu and gpu_vram > 0 else ram_gb
    budget_kind = "gpu" if has_gpu and gpu_vram > 0 else "ram"
    apple_system = _is_apple_image_system(system)
    results = []

    for model in get_image_models():
        if apple_system and not (model.get("mlx_only") or model.get("apple_ok")):
            continue
        if model.get("mlx_only") and not apple_system:
            continue
        # Filter by search
        if isinstance(search, str) and search:
            s = search.lower()
            if s not in model["name"].lower() and s not in model["id"].lower() and s not in model.get("description", "").lower():
                continue

        # Determine best quant that fits
        quant = None
        vram_needed = None
        fits = False
        quant_repo = None

        if budget_gb > 0:
            # Try BF16 first, then FP8, then Q4
            for q, vram_key in [("BF16", "vram_bf16"), ("FP8", "vram_fp8"), ("Q4", "vram_q4")]:
                v = model.get(vram_key)
                if v is not None and v <= budget_gb * 0.90:  # 10% headroom
                    quant = q
                    vram_needed = v
                    fits = True
                    quant_repo = model.get("quant_repos", {}).get(q)
                    break
            # If nothing fits, show what it needs
            if not fits:
                quant = model["default_quant"]
                vram_needed = model.get("vram_bf16", 0)

        # Fit label
        if budget_gb <= 0:
            fit = "no_gpu"
            fit_label = "No GPU"
        elif fits:
            headroom = budget_gb - vram_needed
            if headroom > budget_gb * 0.3:
                fit = "perfect"
                fit_label = "Perfect"
            elif headroom > budget_gb * 0.1:
                fit = "good"
                fit_label = "Good"
            else:
                fit = "tight"
                fit_label = "Tight"
        else:
            fit = "no_fit"
            fit_label = "Too large"

        # Score: quality * speed * fit bonus
        score = model["quality"] * 0.6 + model["speed"] * 0.2
        if fit == "perfect":
            score += 20
        elif fit == "good":
            score += 10
        elif fit == "tight":
            score += 5
        elif fit == "no_fit":
            score -= 30

        results.append({
            "id": model["id"],
            "name": model["name"],
            "provider": model["provider"],
            "params_b": model["params_b"],
            "vram_needed": vram_needed,
            "quant": quant,
            "quant_repo": quant_repo,
            "fits": fits,
            "fit": fit,
            "fit_label": fit_label,
            "fit_budget": budget_kind,
            "quality": model["quality"],
            "speed": model["speed"],
            "score": round(score, 1),
            "capabilities": model["capabilities"],
            "description": model["description"],
            "released": model.get("released", ""),
        })

    # Sort
    if sort == "quality":
        results.sort(key=lambda x: (-x["quality"], -x["score"]))
    elif sort == "speed":
        results.sort(key=lambda x: (-x["speed"], -x["score"]))
    elif sort == "vram":
        results.sort(key=lambda x: (x["vram_needed"] or 999, -x["score"]))
    else:  # fit (default)
        results.sort(key=lambda x: (-x["score"],))

    return results
