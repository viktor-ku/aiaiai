"""
Text-to-Image Generation Module

This module provides utilities for generating images from text prompts
using various diffusion models.

Available models:
    - SDXL: High quality text-to-image with optional pose control
    - FLUX: Flux.1 diffusion model with optional pose control
    - FLUX2: FLUX variant

Example:
    from ai.text2image import sdxl

    # Basic text-to-image
    pipe = sdxl.make_pipe()
    image = sdxl.snap(pipe, "a beautiful sunset over mountains")
    image.save("sunset.png")

    # Pose-controlled generation (SDXL)
    from ai.text2image.sdxl import Pose
    pose = Pose(image_path="reference.jpg", source="photo", strength=1.0)
    image = sdxl.snap(pipe, "person dancing", pose=pose)

    # Pose-controlled generation (Flux)
    from ai.text2image import flux
    pipe = flux.make_pipe(enable_pose=True)
    pose = flux.Pose(image_path="reference.jpg", source="photo", strength=1.0)
    image = flux.snap(pipe, "person dancing", pose=pose)
"""

from ai.text2image import flux
from ai.text2image.flux import FluxPipelines
from ai.text2image.flux import Pose as FluxPose
from ai.text2image.sdxl import Pose as SDXLPose
from ai.text2image.sdxl import SDXLPipelines

# Backwards compatibility: Pose defaults to SDXLPose
Pose = SDXLPose

__all__ = ["Pose", "SDXLPose", "SDXLPipelines", "FluxPose", "FluxPipelines", "flux"]

