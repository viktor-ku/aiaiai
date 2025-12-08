"""
Flux.1 Text-to-Image Generation with Optional Pose Control

This module provides Flux.1 image generation with optional ControlNet-based pose copying.

Basic Usage (text-to-image):
    from ai.text2image.flux import make_pipe, snap

    pipe = make_pipe()
    image = snap(pipe, prompt="a woman in a red dress")
    image.save("output.png")

Pose-Copying Usage:
    from ai.text2image.flux import make_pipe, snap, Pose

    pipe = make_pipe(enable_pose=True)

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

import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from PIL import Image as PILImageModule

from utils.seed import new_seed

if TYPE_CHECKING:
    from PIL.Image import Image


# =============================================================================
# Constants
# =============================================================================

# Base Flux.1 model
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# ControlNet Union for Flux (supports multiple control types including pose)
CONTROLNET_UNION_MODEL_ID = "InstantX/FLUX.1-dev-Controlnet-Union"


# =============================================================================
# Pose Configuration Class
# =============================================================================


@dataclass
class Pose:
    """
    Configuration for pose-controlled image generation.

    This class encapsulates all settings needed to condition Flux generation
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
            raise ValueError(
                "Only one of 'image' or 'image_path' should be provided, not both"
            )
        if not 0.0 <= self.strength <= 2.0:
            raise ValueError(
                f"strength must be between 0.0 and 2.0, got {self.strength}"
            )

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
class FluxPipelines:
    """
    Container holding both base Flux and ControlNet Flux pipelines.

    This is returned by make_pipe() and passed to snap(). It allows snap()
    to choose the appropriate pipeline based on whether pose control is requested.

    Attributes:
        base: Standard Flux text-to-image pipeline.
        pose: Flux ControlNet pipeline for pose-controlled generation.
              May be None if ControlNet loading fails or is disabled.
        _pose_detector: OpenPose detector for extracting poses from photos.
                       Only loaded when needed.
    """

    base: FluxPipeline
    pose: FluxControlNetPipeline | None = None
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

            self._pose_detector = OpenposeDetector.from_pretrained(
                "lllyasviel/ControlNet"
            )
        return self._pose_detector


# =============================================================================
# Pose Preprocessing
# =============================================================================


def _prepare_conditioning_image(
    pose: Pose,
    width: int,
    height: int,
    pipe: FluxPipelines,
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
        pose_image = pose_image.resize(
            (width, height), PILImageModule.Resampling.LANCZOS
        )
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
    model: str = FLUX_MODEL_ID,
    enable_pose: bool = False,
) -> FluxPipelines:
    """
    Create and return Flux pipelines for text-to-image and pose-controlled generation.

    This function loads the base Flux model and optionally the ControlNet
    pose model. Uses CPU offloading to minimize VRAM usage - models are moved
    to GPU only when needed during inference.

    Args:
        model: Hugging Face model ID for the base Flux model.
        enable_pose: Whether to load the ControlNet pose pipeline.
                     Set to False if you only need basic text-to-image.

    Returns:
        FluxPipelines container with .base and optionally .pose pipelines.

    Example:
        # Full pipeline with pose support
        pipe = make_pipe(enable_pose=True)

        # Lighter pipeline without pose support
        pipe = make_pipe(enable_pose=False)
    """
    # Load base Flux pipeline
    base_pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )

    # Use sequential CPU offloading for memory efficiency
    base_pipe.enable_sequential_cpu_offload()

    pose_pipe = None

    if enable_pose:
        # Load ControlNet model for pose control
        controlnet = FluxControlNetModel.from_pretrained(
            CONTROLNET_UNION_MODEL_ID,
            torch_dtype=torch.bfloat16,
        )

        # Create ControlNet pipeline
        pose_pipe = FluxControlNetPipeline.from_pretrained(
            model,
            controlnet=controlnet,
            torch_dtype=torch.bfloat16,
        )
        # Enable CPU offloading for pose pipeline too
        pose_pipe.enable_sequential_cpu_offload()

    return FluxPipelines(base=base_pipe, pose=pose_pipe)


# =============================================================================
# Image Generation
# =============================================================================


def snap(
    pipe: FluxPipelines | FluxPipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 4.5,
    pose: Pose | None = None,
) -> Image:
    """
    Generate an image from a text prompt, optionally conditioned on a pose.

    This is the main generation function. Without a pose argument, it performs
    standard Flux text-to-image generation. With a pose, it uses ControlNet
    to match the generated image to the reference pose.

    Args:
        pipe: Pipeline container from make_pipe(), or legacy FluxPipeline.
        prompt: Text description of the desired image.
        negative_prompt: Things to avoid in the generation (note: Flux has limited negative prompt support).
        seed: Random seed for reproducibility. 0 = random seed.
        width: Output image width (should be divisible by 8).
        height: Output image height (should be divisible by 8).
        steps: Number of denoising steps. More = higher quality but slower.
        guidance: Classifier-free guidance scale. Higher = closer to prompt.
        pose: Optional Pose configuration for pose-controlled generation.
              When None, performs standard text-to-image.

    Returns:
        Generated PIL Image.

    Example:
        # Basic text-to-image
        image = snap(pipe, "a woman standing in a garden")

        # Pose-controlled generation
        pose = Pose(image_path="reference.jpg", source="photo", strength=1.0)
        image = snap(pipe, "a woman standing in a garden", pose=pose)

        # With all options
        image = snap(
            pipe,
            prompt="elegant woman in evening dress",
            seed=42,
            width=1024,
            height=1024,
            steps=30,
            guidance=7.5,
            pose=Pose(image_path="ref.jpg", strength=0.8),
        )
    """
    # Handle legacy single-pipeline usage
    if isinstance(pipe, FluxPipeline):
        if pose is not None:
            raise ValueError(
                "Pose control requires FluxPipelines from make_pipe(enable_pose=True). "
                "Got a bare FluxPipeline instead."
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
    generator = torch.Generator(device=actual_pipe._execution_device).manual_seed(
        new_seed() if seed == 0 else seed
    )

    image = None

    with torch.inference_mode():
        if is_pose_mode:
            # Pose-controlled generation
            assert pose is not None
            assert isinstance(pipe, FluxPipelines)

            conditioning_image = _prepare_conditioning_image(pose, width, height, pipe)

            # Flux ControlNet Union uses control_mode for different control types
            # 4 = OpenPose control mode
            result = actual_pipe(
                prompt=prompt,
                control_image=conditioning_image,
                control_mode=4,  # OpenPose mode
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=pose.strength,
                width=width,
                height=height,
                generator=generator,
                output_type="pil",
            )
        else:
            # Standard text-to-image
            result = actual_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
                output_type="pil",
            )

        image = result.images[0]
        del result

    return image
