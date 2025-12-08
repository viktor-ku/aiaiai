"""
SDXL LoRA Training

This module provides LoRA (Low-Rank Adaptation) training for SDXL models.
After training, use ai.text2image.sdxl to load and use the trained LoRA.

Training Usage:
    from ai.lora.sdxl import TrainingConfig, train_lora_sdxl

    # Configure training
    cfg = TrainingConfig(
        train_images_dir="data/images",
        captions_file="data/captions.jsonl",
        output_dir="output",
        num_epochs=10,
    )

    # Train
    train_lora_sdxl(cfg)

Inference Usage (after training):
    from ai.text2image.sdxl import make_pipe, snap

    pipe = make_pipe(lora_path="output/lora_2025-12-07_233755")
    image = snap(pipe, prompt="a woman in a red dress", lora_scale=0.8)

Dataset Format:
    - A folder with images, for example: data/images
    - A captions file in JSON lines format, for example: data/captions.jsonl

    Each line in captions.jsonl should look like:
      {"file_name": "0001.png", "text": "warm pastel film like portrait, soft light"}
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    # Base SDXL model on Hugging Face
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Your dataset
    train_images_dir: str = "data/images"
    captions_file: str = "data/captions.jsonl"

    # Where to save the LoRA weights
    output_dir: str = "output"

    # Image size - use 768 or 512 if you get OOM errors
    resolution: int = 768  # 1024 needs ~20GB VRAM, 768 needs ~12GB

    # Training settings
    batch_size: int = 1  # keep tiny for a single consumer GPU
    num_epochs: int = 10
    learning_rate: float = 1e-5  # lower LR for stability with fp16
    lora_rank: int = 8  # LoRA rank: capacity vs VRAM
    max_grad_norm: float = 1.0  # gradient clipping for stability

    # Training noise schedule
    num_train_timesteps: int = 1000  # typical DDPM schedule length

    # Hardware and precision
    device: str = "cuda"
    mixed_precision: str = "fp16"  # "fp16" or "bf16" or "no"

    # Randomness
    seed: int | None = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AestheticDataset(Dataset):
    """
    Simple image plus caption dataset backed by a json lines file.

    Each line in captions_file is:
      {"file_name": "...", "text": "caption ..."}

    Also accepts a regular JSON file (array of objects) as fallback if
    the .jsonl file doesn't exist.
    """

    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        resolution: int,
    ) -> None:
        self.images_dir = Path(images_dir)

        # Read all lines of captions into memory
        records: List[Tuple[str, str]] = []
        captions_path = Path(captions_file)

        # Check if .jsonl exists, otherwise try .json fallback
        if captions_path.exists():
            records = self._load_jsonl(captions_path)
        else:
            # Try .json fallback (same name but different extension)
            json_fallback = captions_path.with_suffix(".json")
            if json_fallback.exists():
                print(f"JSONL not found, using JSON fallback: {json_fallback}")
                records = self._load_json(json_fallback)
            else:
                raise FileNotFoundError(
                    f"Neither {captions_path} nor {json_fallback} found."
                )

        if not records:
            raise ValueError("Dataset is empty. Check captions file and folder.")

        # Validate images: filter out non-square images, track which need resize/crop
        self.resolution = resolution
        records = self._validate_and_filter_images(records)

        if not records:
            raise ValueError(
                "No valid square images found. All images must have 1:1 aspect ratio."
            )

        self.records = records

        # Transform: normalize to range minus one to plus one (resize handled in __getitem__)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _validate_and_filter_images(
        self, records: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Validate images and filter out non-square ones.

        - Images must have 1:1 aspect ratio (square)
        - Smaller images will be upscaled in __getitem__
        - Larger images will be cropped in __getitem__
        """
        valid_records: List[Tuple[str, str]] = []
        skipped = 0

        for file_name, text in records:
            image_path = self.images_dir / file_name

            if not image_path.is_file():
                print(f"Warning: Image not found, skipping: {image_path}")
                skipped += 1
                continue

            # Check dimensions without fully loading the image
            with Image.open(image_path) as img:
                width, height = img.size

            if width != height:
                print(
                    f"Warning: Non-square image ({width}x{height}), skipping: {file_name}"
                )
                skipped += 1
                continue

            valid_records.append((file_name, text))

        if skipped > 0:
            print(f"Filtered out {skipped} images, {len(valid_records)} remaining")

        return valid_records

    def _load_jsonl(self, path: Path) -> List[Tuple[str, str]]:
        """Load captions from JSON Lines format (one JSON object per line)."""
        records: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                file_name = obj["file_name"]
                text = obj["text"]
                records.append((file_name, text))
        return records

    def _load_json(self, path: Path) -> List[Tuple[str, str]]:
        """Load captions from a regular JSON file (array of objects)."""
        records: List[Tuple[str, str]] = []
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        # Expect an array of objects with file_name/filename and text/caption keys
        if isinstance(data, list):
            for obj in data:
                # Support both naming conventions
                file_name = obj.get("file_name") or obj.get("filename")
                text = obj.get("text") or obj.get("caption")
                if not file_name or not text:
                    raise ValueError(
                        f"Each entry needs file_name/filename and text/caption keys, got: {obj}"
                    )
                records.append((file_name, text))
        else:
            raise ValueError(
                f"Expected JSON array in {path}, got {type(data).__name__}"
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        file_name, text = self.records[idx]
        image_path = self.images_dir / file_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        # Image is already validated as square, adjust to target resolution
        if width < self.resolution:
            # Upscale smaller images (e.g., 1023x1023 -> 1024x1024)
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.BILINEAR
            )
        elif width > self.resolution:
            # Center crop larger images down to target resolution
            left = (width - self.resolution) // 2
            top = (height - self.resolution) // 2
            image = image.crop(
                (left, top, left + self.resolution, top + self.resolution)
            )
        # else: exactly target resolution, no change needed

        image_tensor = self.transform(image)
        return image_tensor, text


def make_dataloader(cfg: TrainingConfig) -> DataLoader:
    dataset = AestheticDataset(
        images_dir=cfg.train_images_dir,
        captions_file=cfg.captions_file,
        resolution=cfg.resolution,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# LoRA preparation for SDXL UNet (using PEFT)
# ---------------------------------------------------------------------------


def create_pipeline_and_lora(
    cfg: TrainingConfig,
) -> tuple[StableDiffusionXLPipeline, nn.Module]:
    """
    Load SDXL, freeze base weights, attach LoRA blocks using PEFT and return both
    the pipeline and the UNet with LoRA adapters.
    """
    torch_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=torch_dtype,
        variant="fp16",  # SDXL fp16 weights
        use_safetensors=True,
    )

    # Move UNet and VAE to GPU
    # Text encoders stay on CPU to save VRAM - moved to GPU only when encoding
    pipe.unet.to(cfg.device)
    pipe.vae.to(cfg.device)

    # Freeze base model weights
    pipe.vae.requires_grad_(False)
    if pipe.text_encoder is not None:
        pipe.text_encoder.requires_grad_(False)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    # Enable memory optimizations
    pipe.unet.enable_gradient_checkpointing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Configure LoRA for UNet attention layers
    # Target the key projection layers in attention blocks
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank,  # scaling factor, commonly set equal to rank
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Apply LoRA to UNet using PEFT
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Re-enable gradient checkpointing after PEFT wrapping
    pipe.unet.enable_gradient_checkpointing()

    # Only LoRA parameters are trainable
    pipe.unet.print_trainable_parameters()

    return pipe, pipe.unet


def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline, prompts: List[str], device: str
):
    """
    Use the pipeline helper to encode prompts for SDXL.

    This hides the details of the two text encoders and pooled embeddings.
    Text encoders are moved to GPU temporarily, then back to CPU to save VRAM.
    """
    # Move text encoders to GPU for encoding
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    # Returns: prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    # Move text encoders back to CPU to free VRAM for UNet
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    return prompt_embeds, pooled_prompt_embeds


def train_lora_sdxl(cfg: TrainingConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Prepare data
    print_section("Loading dataset")
    dataloader = make_dataloader(cfg)
    num_images = len(dataloader.dataset)
    print(f"    Loaded {num_images} images")

    # Load model
    print_section("Loading SDXL model")
    print(f"    Downloading/loading {cfg.model_id}...")
    pipe, unet = create_pipeline_and_lora(cfg)
    print("    Model loaded and LoRA layers attached")

    device = torch.device(cfg.device)
    dtype = next(pipe.unet.parameters()).dtype  # Get dtype from model params

    # Get trainable LoRA parameters
    lora_params = [p for p in unet.parameters() if p.requires_grad]

    # Simple AdamW over LoRA parameters
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.learning_rate,
        weight_decay=1e-2,
    )

    # GradScaler for mixed precision training (prevents NaN with fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

    # Scheduler inside the pipeline already knows its noise schedule
    noise_scheduler = pipe.scheduler
    num_train_timesteps = noise_scheduler.config.num_train_timesteps

    total_steps = cfg.num_epochs * len(dataloader)

    print_section("Starting training")
    print(
        f"    Total steps:  {total_steps} ({len(dataloader)} batches x {cfg.num_epochs} epochs)"
    )
    print(f"    LoRA params:  {sum(p.numel() for p in lora_params):,}")
    print("    Press Ctrl+C to stop early (weights will be saved)")
    print()

    pipe.unet.train()

    start_time = datetime.now()
    step_idx = 0
    interrupted = False
    completed_epochs = 0

    def save_checkpoint(is_interrupt: bool = False) -> Path:
        """Save LoRA weights, returns the output path."""
        nonlocal end_time
        end_time = datetime.now()
        elapsed_minutes = (end_time - start_time).total_seconds() / 60

        suffix = "_interrupted" if is_interrupt else ""
        date_str = end_time.strftime("%Y-%m-%d_%H%M%S")
        output_path = Path(cfg.output_dir) / f"lora_{date_str}{suffix}"
        os.makedirs(output_path, exist_ok=True)

        if is_interrupt:
            print(f"\n\n>>> Interrupted! Saving checkpoint...")
        else:
            print_section("Saving LoRA weights")

        # Save using PEFT's save method
        unet.save_pretrained(output_path)
        print_summary(
            output_path, elapsed_minutes, len(dataloader.dataset), completed_epochs
        )
        return output_path

    # Handle Ctrl+C gracefully
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    original_handler = signal.signal(signal.SIGINT, handle_interrupt)
    end_time = datetime.now()  # Initialize in case of early interrupt

    try:
        for epoch in range(cfg.num_epochs):
            if interrupted:
                break

            epoch_start = datetime.now()
            epoch_loss = 0.0

            for batch in dataloader:
                if interrupted:
                    break

                step_idx += 1

                images, texts = batch
                images = images.to(device=device, dtype=dtype)

                # Encode images to latents using the VAE
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                    # Temporarily cast VAE to fp32 for stable encoding
                    vae_dtype = pipe.vae.dtype
                    pipe.vae.to(torch.float32)
                    images_fp32 = images.float()

                    latents_dist = pipe.vae.encode(images_fp32).latent_dist
                    latents = latents_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    # Cast VAE back to fp16 and latents to training dtype
                    pipe.vae.to(vae_dtype)
                    latents = latents.to(dtype)

                # Sample random noise and a random timestep for each image
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    low=0,
                    high=min(num_train_timesteps, cfg.num_train_timesteps),
                    size=(latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )

                # Forward diffusion: add noise at the chosen timesteps
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text prompts for SDXL
                prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                    pipe, list(texts), device
                )

                # SDXL needs extra conditioning via time ids
                add_time_ids = pipe._get_add_time_ids(
                    original_size=(cfg.resolution, cfg.resolution),
                    crops_coords_top_left=(0, 0),
                    target_size=(cfg.resolution, cfg.resolution),
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
                )
                add_time_ids = add_time_ids.to(device)
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)

                # Predict the noise with UNet using autocast for mixed precision
                with torch.amp.autocast(
                    "cuda", enabled=(cfg.mixed_precision == "fp16")
                ):
                    model_pred = pipe.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids,
                        },
                    ).sample

                # Compute loss in fp32 (outside autocast for numerical stability)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    Warning: NaN/Inf loss at step {step_idx}, skipping...")
                    del (
                        latents,
                        noise,
                        noisy_latents,
                        prompt_embeds,
                        pooled_prompt_embeds,
                    )
                    del add_time_ids, model_pred, loss
                    torch.cuda.empty_cache()
                    continue

                # Backward pass with scaler (handles fp16 gradients)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                loss_value = loss.detach().item()
                epoch_loss += loss_value

                # Free memory
                del latents, noise, noisy_latents, prompt_embeds, pooled_prompt_embeds
                del add_time_ids, model_pred, loss
                torch.cuda.empty_cache()

                if step_idx % 10 == 0:
                    pct = (step_idx / total_steps) * 100
                    print(
                        f"    [{pct:5.1f}%] Step {step_idx:4d}/{total_steps}  Loss: {loss_value:.4f}"
                    )

            if not interrupted:
                completed_epochs = epoch + 1
                epoch_duration = (datetime.now() - epoch_start).total_seconds()
                avg_epoch_loss = epoch_loss / max(1, len(dataloader))
                print(
                    f"  ✓ Epoch {epoch + 1}/{cfg.num_epochs} done  |  Loss: {avg_epoch_loss:.4f}  |  Time: {epoch_duration:.1f}s"
                )

        # Save checkpoint (normal completion or after interrupt)
        save_checkpoint(is_interrupt=interrupted)

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------


def print_banner(text: str) -> None:
    """Print a prominent banner."""
    width = max(len(text) + 4, 50)
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n>>> {text}")


def print_config(cfg: TrainingConfig) -> None:
    """Print training configuration summary."""
    print_section("Configuration")
    print(f"    Model:        {cfg.model_id}")
    print(f"    Images:       {cfg.train_images_dir}")
    print(f"    Captions:     {cfg.captions_file}")
    print(f"    Output:       {cfg.output_dir}")
    print(f"    Resolution:   {cfg.resolution}x{cfg.resolution}")
    print(f"    Epochs:       {cfg.num_epochs}")
    print(f"    Batch size:   {cfg.batch_size}")
    print(f"    Learning rate:{cfg.learning_rate}")
    print(f"    LoRA rank:    {cfg.lora_rank}")
    print(f"    Precision:    {cfg.mixed_precision}")
    print(f"    Device:       {cfg.device}")


def print_summary(
    output_path: Path, elapsed_minutes: float, num_images: int, num_epochs: int
) -> None:
    """Print final training summary."""
    print()
    print("+" + "-" * 58 + "+")
    print("|" + " TRAINING COMPLETE ".center(58) + "|")
    print("+" + "-" * 58 + "+")
    print(f"|  Output:    {str(output_path):<44} |")
    print(f"|  Duration:  {elapsed_minutes:.1f} minutes{' ' * 37}|"[:60] + "|")
    print(
        f"|  Images:    {num_images} images x {num_epochs} epochs{' ' * 30}|"[:60] + "|"
    )
    print("+" + "-" * 58 + "+")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = TrainingConfig()

    print_banner("SDXL LoRA Training")
    print_config(cfg)

    train_lora_sdxl(cfg)
