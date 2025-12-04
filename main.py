import torch
from diffusers import StableDiffusionXLPipeline
import secrets


def random_hex8() -> str:
    return secrets.token_hex(4)


def main():
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

    prompt = "a cozy warm living room with a sleeping kitten on a blanket, soft lighting, cinematic, high detail"
    negative = "blurry, low quality, distorted, watermark, text"

    batch = random_hex8()

    generator = torch.Generator(device="cuda").manual_seed(42)

    for i in range(10):
        image = None

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=30,
                guidance_scale=6.5,
                width=768,
                height=768,
                generator=generator,
            )

            image = result.images[0]

        image.save(f"output/ai-{batch}-{i}.{random_hex8()}.png")

        del result, image


if __name__ == "__main__":
    main()
