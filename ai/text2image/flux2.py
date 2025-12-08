from diffusers import Flux2Pipeline
from PIL import Image
from utils.seed import new_seed
import torch


def make_pipe(model: str = "black-forest-labs/FLUX.2-dev") -> Flux2Pipeline:
    pipe = Flux2Pipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )

    pipe.enable_sequential_cpu_offload()

    return pipe


def snap(
    pipe: Flux2Pipeline,
    prompt: str,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 4.5,
) -> Image.Image:
    generator = torch.Generator(device=pipe._execution_device).manual_seed(
        new_seed() if seed == 0 else seed
    )

    out = None

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
        )

    return out.images[0]
