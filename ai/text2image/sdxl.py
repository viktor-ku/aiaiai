"""
SDXL Text-to-Image Generation with Optional Pose Control and LoRA Support

This module provides SDXL image generation with optional ControlNet-based pose copying
and LoRA (Low-Rank Adaptation) inference support.

For LoRA training, see ai.lora.sdxl module.

Basic Usage (text-to-image):
    from ai.text2image.sdxl import make_pipe, snap
    
    pipe = make_pipe()
    image = snap(pipe, prompt="a woman in a red dress")
    image.save("output.png")

With LoRA (trained via ai.lora.sdxl):
    from ai.text2image.sdxl import make_pipe, snap, load_lora
    
    # Load LoRA at pipeline creation
    pipe = make_pipe(lora_path="output/lora_2025-12-07_233755")
    image = snap(pipe, prompt="a woman in a red dress", lora_scale=0.8)
    
    # Or load LoRA after pipeline creation
    pipe = make_pipe()
    load_lora(pipe, "output/lora_2025-12-07_233755")
    image = snap(pipe, prompt="a woman in a red dress", lora_scale=0.8)

Pose-Copying Usage:
    from ai.text2image.sdxl import make_pipe, snap, Pose
    
    pipe = make_pipe()
    
    # From a photo (pose will be extracted automatically)
    pose = Pose(image_path="/path/to/reference.jpg", source="photo", strength=1.0)
    image = snap(pipe, prompt="a woman in a red dress", pose=pose)
    
    # From an existing pose map (skeleton image)
    pose = Pose(image_path="/path/to/pose_map.png", source="pose_map", strength=0.8)
    image = snap(pipe, prompt="a woman in a red dress", pose=pose)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", message=".*mediapipe.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.*")
warnings.filterwarnings("ignore", message=".*Overwriting tiny_vit.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*Defaulting to unsafe serialization.*")
warnings.filterwarnings("ignore", message=".*upcast_vae.*is deprecated.*")

import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline
from PIL import Image as PILImageModule

from utils.seed import new_seed

if TYPE_CHECKING:
    from PIL.Image import Image


# =============================================================================
# Constants
# =============================================================================

# Base SDXL model
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# OpenPose ControlNet for SDXL (handles human pose conditioning)
CONTROLNET_OPENPOSE_MODEL_ID = "thibaud/controlnet-openpose-sdxl-1.0"


# =============================================================================
# Pose Configuration Class
# =============================================================================


@dataclass
class Pose:
    """
    Configuration for pose-controlled image generation.

    This class encapsulates all settings needed to condition SDXL generation
    on a reference pose. The pose can come from either:
    - A regular photo (pose will be extracted using OpenPose)
    - A pre-made pose map/skeleton image

    Attributes:
        image: Direct PIL Image to use as pose reference. Mutually exclusive with image_path.
        image_path: Path to the pose reference image. Mutually exclusive with image.
        source: Type of input - "photo" for regular images (pose will be extracted),
                "pose_map" for pre-existing skeleton/pose images.
        strength: ControlNet conditioning scale (0.0-2.0). Higher = stronger pose adherence.
                  1.0 is a good default; lower for more creative freedom.
        resize_mode: How to handle size mismatch - "resize" stretches to target,
                     "crop" center-crops to target dimensions.

    Example:
        # From a photo - pose will be auto-extracted
        pose = Pose(image_path="reference.jpg", source="photo", strength=1.0)

        # From an existing pose skeleton
        pose = Pose(image_path="skeleton.png", source="pose_map", strength=0.8)

        # Using PIL Image directly
        from PIL import Image
        img = Image.open("ref.jpg")
        pose = Pose(image=img, source="photo")
    """

    image: Image | None = None
    image_path: str | Path | None = None
    source: Literal["photo", "pose_map"] = "photo"
    strength: float = 1.0
    resize_mode: Literal["resize", "crop"] = "resize"

    def __post_init__(self) -> None:
        """Validate that exactly one of image or image_path is provided."""
        if self.image is None and self.image_path is None:
            raise ValueError("Either 'image' or 'image_path' must be provided")
        if self.image is not None and self.image_path is not None:
            raise ValueError("Only one of 'image' or 'image_path' should be provided, not both")
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError(f"strength must be between 0.0 and 2.0, got {self.strength}")

    def get_image(self) -> Image:
        """
        Resolve and return the reference image.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If image_path doesn't exist.
        """
        if self.image is not None:
            return self.image.convert("RGB")

        path = Path(self.image_path)  # type: ignore
        if not path.exists():
            raise FileNotFoundError(f"Pose reference image not found: {path}")

        return PILImageModule.open(path).convert("RGB")


# =============================================================================
# Pipeline Container
# =============================================================================


@dataclass
class SDXLPipelines:
    """
    Container holding both base SDXL and ControlNet SDXL pipelines.

    This is returned by make_pipe() and passed to snap(). It allows snap()
    to choose the appropriate pipeline based on whether pose control is requested.

    Attributes:
        base: Standard SDXL text-to-image pipeline.
        pose: SDXL ControlNet pipeline for pose-controlled generation.
              May be None if ControlNet loading fails or is disabled.
        pose_detector: OpenPose detector for extracting poses from photos.
                       Only loaded when needed.
    """

    base: StableDiffusionXLPipeline
    pose: StableDiffusionXLControlNetPipeline | None = None
    _pose_detector: object | None = field(default=None, repr=False)

    def get_pose_detector(self):
        """
        Lazily load and return the OpenPose detector.

        The detector is only loaded on first use to save memory when
        pose extraction isn't needed.

        Returns:
            OpenposeDetector instance from controlnet_aux.
        """
        if self._pose_detector is None:
            # Suppress controlnet_aux import warnings before importing
            import warnings

            warnings.filterwarnings("ignore", message=".*mediapipe.*")
            warnings.filterwarnings("ignore", message=".*Importing from timm.*")
            warnings.filterwarnings("ignore", message=".*Overwriting tiny_vit.*")

            from controlnet_aux import OpenposeDetector

            self._pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        return self._pose_detector


# =============================================================================
# Pose Preprocessing
# =============================================================================


def _prepare_conditioning_image(
    pose: Pose,
    width: int,
    height: int,
    pipe: SDXLPipelines,
) -> Image:
    """
    Convert a Pose configuration into a ControlNet-ready conditioning image.

    For photos, this runs OpenPose detection to extract the skeleton.
    For pose maps, this just resizes/crops the existing skeleton image.

    Args:
        pose: Pose configuration with reference image.
        width: Target image width.
        height: Target image height.
        pipe: Pipeline container (needed to access the pose detector).

    Returns:
        RGB PIL Image of the pose skeleton, sized to width x height.
    """
    # Get the reference image
    ref_image = pose.get_image()

    if pose.source == "photo":
        # Extract pose from the photo using OpenPose
        detector = pipe.get_pose_detector()
        # Run detection - returns a pose skeleton image
        pose_image = detector(ref_image)
    else:
        # Already a pose map, use directly
        pose_image = ref_image

    # Resize/crop to target dimensions
    if pose.resize_mode == "resize":
        pose_image = pose_image.resize((width, height), PILImageModule.Resampling.LANCZOS)
    else:
        # Center crop
        pose_image = _center_crop(pose_image, width, height)

    return pose_image.convert("RGB")


def _center_crop(image: Image, target_width: int, target_height: int) -> Image:
    """
    Center-crop an image to the target dimensions.

    If the image is smaller than target in any dimension, it will be
    resized up first to ensure the crop is possible.

    Args:
        image: Source PIL Image.
        target_width: Desired output width.
        target_height: Desired output height.

    Returns:
        Center-cropped PIL Image.
    """
    img_width, img_height = image.size

    # Calculate scale to ensure image covers target area
    scale = max(target_width / img_width, target_height / img_height)

    if scale > 1.0:
        # Need to upscale first
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        image = image.resize((new_width, new_height), PILImageModule.Resampling.LANCZOS)
        img_width, img_height = new_width, new_height

    # Center crop
    left = (img_width - target_width) // 2
    top = (img_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return image.crop((left, top, right, bottom))


# =============================================================================
# Pipeline Construction
# =============================================================================


def make_pipe(
    enable_pose: bool = True,
    lora_path: str | Path | None = None,
) -> SDXLPipelines:
    """
    Create and return SDXL pipelines for text-to-image and pose-controlled generation.

    This function loads the base SDXL model and optionally the ControlNet
    pose model. Uses CPU offloading to minimize VRAM usage - models are moved
    to GPU only when needed during inference.

    Args:
        enable_pose: Whether to load the ControlNet pose pipeline.
                     Set to False if you only need basic text-to-image.
        lora_path: Optional path to a LoRA weights directory (from training).
                   The LoRA will be loaded on top of the base model.

    Returns:
        SDXLPipelines container with .base and optionally .pose pipelines.

    Example:
        # Full pipeline with pose support
        pipe = make_pipe()

        # Lighter pipeline without pose support
        pipe = make_pipe(enable_pose=False)

        # With LoRA weights
        pipe = make_pipe(lora_path="output_lora_sdxl/lora_2025-12-07_143052")
    """
    # Load base SDXL pipeline
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pose_pipe = None

    if enable_pose:
        # Load ControlNet model for pose (keep on CPU until needed)
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_OPENPOSE_MODEL_ID,
            torch_dtype=torch.float16,
        )

        # Create ControlNet pipeline, reusing components from base
        # This shares VAE, text encoders, UNet, scheduler - only ControlNet is extra
        pose_pipe = StableDiffusionXLControlNetPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            controlnet=controlnet,
        )

    pipelines = SDXLPipelines(base=base_pipe, pose=pose_pipe)

    # Load LoRA if provided (must be done BEFORE enabling CPU offloading)
    # PeftModel wrapping conflicts with diffusers' hook system
    if lora_path is not None:
        load_lora(pipelines, lora_path)

    # Enable CPU offloading AFTER LoRA loading
    # This is critical for running SDXL on consumer GPUs (< 24GB VRAM)
    # Must be done after LoRA loading because PeftModel wrapping breaks existing hooks
    base_pipe.enable_model_cpu_offload()
    base_pipe.enable_attention_slicing()
    base_pipe.vae.enable_slicing()

    if pose_pipe is not None:
        pose_pipe.enable_model_cpu_offload()
        pose_pipe.enable_attention_slicing()
        pose_pipe.vae.enable_slicing()

    return pipelines


def load_lora(pipe: SDXLPipelines, lora_path: str | Path) -> None:
    """
    Load LoRA weights onto an existing pipeline.

    This loads PEFT LoRA adapters trained with ai.lora.sdxl or similar
    onto the UNet. The LoRA modifies the model's style/aesthetics based on
    training data.

    Note: This should be called BEFORE enabling CPU offloading on the pipeline,
    because PeftModel wrapping conflicts with diffusers' hook system. When using
    make_pipe(lora_path=...), this is handled automatically.

    Args:
        pipe: SDXLPipelines container from make_pipe().
        lora_path: Path to the LoRA weights directory (PEFT format).

    Example:
        pipe = make_pipe()
        load_lora(pipe, "output_lora_sdxl/lora_2025-12-07_143052")
        image = snap(pipe, "a portrait", lora_scale=0.8)
    """
    from peft import PeftModel

    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    print(f"Loading LoRA weights from: {lora_path}")

    # Load PEFT LoRA adapter onto the UNet
    pipe.base.unet = PeftModel.from_pretrained(pipe.base.unet, lora_path)

    # If pose pipeline exists, update its reference to the same UNet
    if pipe.pose is not None:
        pipe.pose.unet = pipe.base.unet

    print("LoRA loaded successfully")


# =============================================================================
# Image Generation
# =============================================================================


def snap(
    pipe: SDXLPipelines | StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 768,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 4.5,
    pose: Pose | None = None,
    lora_scale: float = 1.0,
) -> Image:
    """
    Generate an image from a text prompt, optionally conditioned on a pose.

    This is the main generation function. Without a pose argument, it performs
    standard SDXL text-to-image generation. With a pose, it uses ControlNet
    to match the generated image to the reference pose.

    Args:
        pipe: Pipeline container from make_pipe(), or legacy StableDiffusionXLPipeline.
        prompt: Text description of the desired image.
        negative_prompt: Things to avoid in the generation.
        seed: Random seed for reproducibility. 0 = random seed.
        width: Output image width (should be divisible by 8).
        height: Output image height (should be divisible by 8).
        steps: Number of denoising steps. More = higher quality but slower.
        guidance: Classifier-free guidance scale. Higher = closer to prompt.
        pose: Optional Pose configuration for pose-controlled generation.
              When None, performs standard text-to-image.
        lora_scale: Scale factor for LoRA influence (0.0-1.0). Only applies if
                    LoRA was loaded. 1.0 = full LoRA effect, 0.0 = base model only.

    Returns:
        Generated PIL Image.

    Example:
        # Basic text-to-image
        image = snap(pipe, "a woman standing in a garden")

        # Pose-controlled generation
        pose = Pose(image_path="reference.jpg", source="photo", strength=1.0)
        image = snap(pipe, "a woman standing in a garden", pose=pose)

        # With LoRA
        pipe = make_pipe(lora_path="my_lora")
        image = snap(pipe, "portrait in my style", lora_scale=0.8)

        # With all options
        image = snap(
            pipe,
            prompt="elegant woman in evening dress",
            negative_prompt="ugly, blurry",
            seed=42,
            width=1024,
            height=1024,
            steps=30,
            guidance=7.5,
            pose=Pose(image_path="ref.jpg", strength=0.8),
            lora_scale=0.7,
        )
    """
    # Handle legacy single-pipeline usage
    if isinstance(pipe, StableDiffusionXLPipeline):
        if pose is not None:
            raise ValueError(
                "Pose control requires SDXLPipelines from make_pipe(). "
                "Got a bare StableDiffusionXLPipeline instead."
            )
        actual_pipe = pipe
        is_pose_mode = False
    else:
        is_pose_mode = pose is not None
        if is_pose_mode:
            if pipe.pose is None:
                raise ValueError(
                    "Pose control requested but ControlNet pipeline not loaded. "
                    "Call make_pipe(enable_pose=True)."
                )
            actual_pipe = pipe.pose
        else:
            actual_pipe = pipe.base

    # Set up the generator with seed
    generator = torch.Generator(device="cuda").manual_seed(new_seed() if seed == 0 else seed)

    # Pass LoRA scale via cross_attention_kwargs instead of set_adapters
    # set_adapters fails with inference mode tensors
    cross_attention_kwargs = {"scale": lora_scale} if lora_scale != 1.0 else None

    with torch.inference_mode():
        if is_pose_mode:
            # Pose-controlled generation
            assert pose is not None
            assert isinstance(pipe, SDXLPipelines)

            conditioning_image = _prepare_conditioning_image(pose, width, height, pipe)

            result = actual_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioning_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=pose.strength,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            # Standard text-to-image
            result = actual_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        image = result.images[0]
        del result

    return image
