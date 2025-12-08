"""
LoRA (Low-Rank Adaptation) training modules.

This package contains LoRA training code for various model architectures.
For using trained LoRAs during inference, see ai.text2image modules.
"""

from ai.lora.sdxl import TrainingConfig, train_lora_sdxl

__all__ = ["TrainingConfig", "train_lora_sdxl"]

