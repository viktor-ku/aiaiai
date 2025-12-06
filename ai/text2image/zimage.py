import torch
from diffusers import ZImagePipeline
from PIL import Image
from utils.seed import new_seed


def make_pipe() -> ZImagePipeline:
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    return pipe


def snap(
    pipe: ZImagePipeline,
    prompt: str,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 9,  # Results in 8 DiT forwards for Z-Image-Turbo
) -> Image.Image:
    """Generate an image using Z-Image-Turbo.

    Note: guidance_scale is fixed at 0.0 for Turbo models.
    """
    generator = torch.Generator("cuda").manual_seed(new_seed() if seed == 0 else seed)

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Must be 0 for Turbo models
            generator=generator,
        )

    return out.images[0]
