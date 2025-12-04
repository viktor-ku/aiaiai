import torch
from diffusers import FluxPipeline
import secrets
import random


def main():
    batch = random_hex8()

    prompt = "a cozy warm living room with a sleeping kitten on a blanket, soft lighting, cinematic, high detail"
    negative = "blurry, low quality, distorted, watermark, text"

    pipe = make_flux_pipeline()

    for i in range(10):
        image = generate_flux_image(pipe=pipe, prompt=prompt)
        image.save(f"output/ai-{batch}-{i}.{random_hex8()}.png")


def random_hex8() -> str:
    return secrets.token_hex(4)


def make_flux_pipeline() -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,  # Flux likes bf16
    )
    # You have 12 GB VRAM -> use CPU offload
    pipe.enable_sequential_cpu_offload()
    # For more speed but higher VRAM use:
    # pipe.enable_model_cpu_offload()
    # or if you somehow have 24GB+:
    # pipe.to("cuda")
    return pipe


def generate_flux_image(
    pipe: FluxPipeline,
    prompt: str,
    *,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    steps: int = 50,
    guidance: float = 3.5,
):
    # For dev variant, 50 steps & ~3–4 CFG is typical. :contentReference[oaicite:5]{index=5}
    generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

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


def new_seed() -> int:
    # 32-bit unsigned integer range
    return random.randint(0, 2**32 - 1)


if __name__ == "__main__":
    main()
