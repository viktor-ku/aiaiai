#!/usr/bin/env python3
"""
AI Image Upscaler CLI

Upscale images using Stable Diffusion ×2 or ×4 upscalers.

Usage:
    ai3-upscale input.png                    # ×4 upscale, output to input_4x.png
    ai3-upscale input.png -o output.png      # ×4 upscale to specific path
    ai3-upscale input.png -s 2               # ×2 upscale
    ai3-upscale input.png -p "detailed photo"  # With prompt guidance

Examples:
    # Basic upscaling
    ai3-upscale photo.jpg

    # Upscale with prompt guidance for better results
    ai3-upscale portrait.png -p "high resolution portrait, sharp details"

    # ×2 upscale (faster)
    ai3-upscale image.png -s 2 -o upscaled.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


HELP_TEXT = """\
ai3-upscale - AI Image Upscaler

Upscale images using Stable Diffusion ×2 or ×4 upscalers from Stability AI.

USAGE:
    ai3-upscale <image> [options]
    ai3-upscale help

ARGUMENTS:
    <image>             Input image path (required)

OPTIONS:
    -o, --output PATH   Output image path
                        Default: <input>_Nx.<ext> where N is scale factor

    -s, --scale N       Upscaling factor: 2 or 4 (default: 4)
                        ×4 is slower but higher quality
                        ×2 is faster, uses latent-space upscaling

    -p, --prompt TEXT   Text prompt to guide upscaling
                        Improves results by describing desired output
                        Example: "high resolution portrait, sharp details"

    -n, --negative-prompt TEXT
                        Things to avoid in the output
                        Default: "blurry, low quality, artifacts, noise"

    --steps N           Number of denoising steps (default: 20)
                        Higher = better quality but slower

    --guidance N        Guidance scale (default: 7.5)
                        Higher = follows prompt more closely

    --seed N            Random seed for reproducibility
                        Default: 0 (random)

EXAMPLES:
    # Basic ×4 upscaling
    ai3-upscale photo.jpg

    # ×2 upscaling (faster)
    ai3-upscale photo.jpg -s 2

    # Custom output path
    ai3-upscale photo.jpg -o upscaled.png

    # With prompt guidance for better quality
    ai3-upscale portrait.png -p "high resolution portrait photo, sharp details"

    # Full control
    ai3-upscale image.png -o big.png -s 4 -p "detailed" --steps 30 --seed 42

MODELS:
    ×4: stabilityai/stable-diffusion-x4-upscaler (prompt-guided)
    ×2: stabilityai/sd-x2-latent-upscaler (latent-space)
"""


class HelpfulArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that shows full help on error."""

    def error(self, message: str) -> None:
        print(HELP_TEXT, file=sys.stderr)
        print(f"\nError: {message}", file=sys.stderr)
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    # Handle "help" as first argument or no arguments
    if len(sys.argv) == 1:
        print(HELP_TEXT)
        print("Error: missing required argument: <image>", file=sys.stderr)
        sys.exit(2)

    if sys.argv[1].lower() in ("help", "-h", "--help"):
        print(HELP_TEXT)
        sys.exit(0)

    parser = HelpfulArgumentParser(
        prog="ai3-upscale",
        add_help=False,  # We handle help ourselves
    )

    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-s", "--scale", type=int, choices=[2, 4], default=4)
    parser.add_argument("-p", "--prompt", type=str, default="")
    parser.add_argument(
        "-n",
        "--negative-prompt",
        type=str,
        default="blurry, low quality, artifacts, noise",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate input
    if not args.image.exists():
        print(f"Error: Input image not found: {args.image}", file=sys.stderr)
        return 1

    # Determine output path
    if args.output is None:
        stem = args.image.stem
        suffix = args.image.suffix or ".png"
        output_path = args.image.parent / f"{stem}_{args.scale}x{suffix}"
    else:
        output_path = args.output

    # Print info
    print(f"Input:  {args.image}")
    print(f"Output: {output_path}")
    print(f"Scale:  ×{args.scale}")
    if args.prompt:
        print(f"Prompt: {args.prompt}")
    print()

    # Load pipeline (import here to keep CLI fast for --help)
    from ai.upscaling.sdxl import make_pipe, upscale

    print(f"Loading ×{args.scale} upscaler...")
    pipe = make_pipe(scale=args.scale)  # type: ignore
    print("Pipeline ready.\n")

    # Upscale
    print("Upscaling...")
    result = upscale(
        pipe,
        image=args.image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"\nSaved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

