import torch
from diffusers import StableDiffusionXLPipeline
from utils.seed import new_seed


def make_pipe():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe.to("cuda")

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    return pipe


def snap(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 768,
    height: int = 1024,
    steps: int = 20,
    guidance: float = 4.5,
):
    generator = torch.Generator(device="cuda").manual_seed(
        new_seed() if seed == 0 else seed
    )

    image = None

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        )

        image = result.images[0]

        del result

    return image
