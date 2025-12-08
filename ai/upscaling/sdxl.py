"""
Image Upscaling with Stable Diffusion Upscalers

This module provides ×2 and ×4 upscaling using official Stability AI upscaler models.

Basic Usage:
    from ai.upscaling.sdxl import make_pipe, upscale

    # Load ×4 upscaler (default)
    pipe = make_pipe(scale=4)
    upscaled = upscale(pipe, image="input.png", prompt="high quality photo")
    upscaled.save("output.png")

    # Load ×2 upscaler
    pipe = make_pipe(scale=2)
    upscaled = upscale(pipe, image="input.png")

With PIL Image:
    from PIL import Image
    from ai.upscaling.sdxl import make_pipe, upscale

    pipe = make_pipe()
    img = Image.open("input.png")
    upscaled = upscale(pipe, image=img, prompt="detailed photograph")

Available Models:
    - ×4: stabilityai/stable-diffusion-x4-upscaler (SD 1.x based, prompt-guided)
    - ×2: stabilityai/sd-x2-latent-upscaler (latent-space upscaling)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Defaulting to unsafe serialization.*")

import torch
from diffusers import (
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionUpscalePipeline,
)
from PIL import Image as PILImageModule

from utils.seed import new_seed

if TYPE_CHECKING:
    from PIL.Image import Image


# =============================================================================
# Constants
# =============================================================================

# Official Stability AI upscaler models
UPSCALER_X4_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALER_X2_MODEL_ID = "stabilityai/sd-x2-latent-upscaler"

# Type alias for pipeline union
UpscalerPipeline = Union[
    StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
]


# =============================================================================
# Pipeline Construction
# =============================================================================


def make_pipe(scale: Literal[2, 4] = 4) -> UpscalerPipeline:
    """
    Create and return an upscaler pipeline.

    This function loads the appropriate Stability AI upscaler model based on
    the desired scale factor. Uses CPU offloading to minimize VRAM usage.

    Args:
        scale: Upscaling factor - 2 for ×2 or 4 for ×4 upscaling.
               Default is 4 for highest quality upscaling.

    Returns:
        Upscaler pipeline ready for inference.

    Example:
        # ×4 upscaler (default, best quality)
        pipe = make_pipe(scale=4)

        # ×2 upscaler (faster, latent-space)
        pipe = make_pipe(scale=2)
    """
    if scale == 4:
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            UPSCALER_X4_MODEL_ID,
            torch_dtype=torch.float16,
        )
    elif scale == 2:
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            UPSCALER_X2_MODEL_ID,
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"scale must be 2 or 4, got {scale}")

    # Enable CPU offloading to minimize VRAM usage
    pipe.enable_model_cpu_offload()

    return pipe


# =============================================================================
# Image Loading
# =============================================================================


def _load_image(image: Image | str | Path) -> Image:
    """
    Load and validate an input image.

    Args:
        image: PIL Image, or path to an image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If image path doesn't exist.
        TypeError: If image is not a valid type.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return PILImageModule.open(path).convert("RGB")

    # Assume it's a PIL Image
    try:
        return image.convert("RGB")
    except AttributeError:
        raise TypeError(
            f"image must be a PIL Image or path, got {type(image).__name__}"
        )


# =============================================================================
# Upscaling
# =============================================================================


def upscale(
    pipe: UpscalerPipeline,
    image: Image | str | Path,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, artifacts, noise",
    seed: int = 0,
    steps: int = 20,
    guidance: float = 7.5,
) -> Image:
    """
    Upscale an image using the loaded upscaler pipeline.

    The ×4 upscaler is prompt-guided, meaning it can enhance details based on
    a text description. The ×2 upscaler works in latent space and is faster
    but less controllable.

    Args:
        pipe: Upscaler pipeline from make_pipe().
        image: Input image - PIL Image or path to image file.
        prompt: Text description to guide upscaling (×4 only).
                Leave empty for generic enhancement.
        negative_prompt: Things to avoid in the output.
        seed: Random seed for reproducibility. 0 = random seed.
        steps: Number of denoising steps. More = higher quality but slower.
        guidance: Classifier-free guidance scale. Higher = closer to prompt.

    Returns:
        Upscaled PIL Image.

    Example:
        # Basic upscaling
        upscaled = upscale(pipe, "input.png")

        # With prompt guidance (×4 upscaler)
        upscaled = upscale(
            pipe,
            image="portrait.png",
            prompt="high resolution portrait photo, sharp details",
            steps=30,
        )

        # From PIL Image
        from PIL import Image
        img = Image.open("input.png")
        upscaled = upscale(pipe, image=img)
    """
    # Load input image
    input_image = _load_image(image)

    # Set up the generator with seed
    generator = torch.Generator(device="cuda").manual_seed(
        new_seed() if seed == 0 else seed
    )

    with torch.inference_mode():
        if isinstance(pipe, StableDiffusionUpscalePipeline):
            # ×4 upscaler - prompt-guided
            result = pipe(
                prompt=prompt if prompt else "high quality, detailed",
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        else:
            # ×2 latent upscaler
            result = pipe(
                prompt=prompt if prompt else "high quality",
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        output_image = result.images[0]
        del result

    return output_image

