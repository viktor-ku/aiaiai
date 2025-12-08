#!/usr/bin/env python3
"""
AI Image Upscaler Daemon

A FastAPI server that provides image upscaling via HTTP API.

Usage:
    python main.py                    # Load both, port 3000
    python main.py --port 8080        # Custom port
    python main.py --scale 4          # Only ×4 upscaler
    python main.py --scale 2          # Only ×2 upscaler
    python main.py --scale both       # Both (default)

API:
    POST /upscale      - Upscale an image
    GET  /ping         - Health check
    GET  /docs         - OpenAPI documentation
    GET  /openapi.json - OpenAPI schema
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.upscaling.sdxl import UpscalerPipeline, make_pipe, upscale

# =============================================================================
# Global State
# =============================================================================

pipelines: dict[int, UpscalerPipeline] = {}


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global pipelines

    scales_to_load: list[int] = app.state.scales

    for scale in scales_to_load:
        print(f"Loading ×{scale} upscaler...")
        pipelines[scale] = make_pipe(scale=scale)  # type: ignore
        print(f"  ✓ ×{scale} upscaler ready")

    available = ", ".join(f"×{s}" for s in sorted(pipelines.keys()))
    print(f"\nReady and listening on port {app.state.port}")
    print(f"Available scales: {available}")

    yield

    # Cleanup
    pipelines.clear()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="AI Image Upscaler",
    description="Upscale images using Stable Diffusion ×2 or ×4 upscalers",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post(
    "/upscale",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Upscaled image as PNG",
        }
    },
)
async def upscale_image(
    image: Annotated[UploadFile, File(description="Image file to upscale")],
    scale: Annotated[
        Literal[2, 4], Form(description="Upscaling factor: 2 or 4")
    ] = 4,
    prompt: Annotated[
        str, Form(description="Text prompt to guide upscaling")
    ] = "",
    negative_prompt: Annotated[
        str, Form(description="Things to avoid in output")
    ] = "blurry, low quality, artifacts, noise",
    seed: Annotated[
        int, Form(description="Random seed (0 = random)")
    ] = 0,
    steps: Annotated[
        int, Form(description="Number of denoising steps", ge=1, le=100)
    ] = 20,
    guidance: Annotated[
        float, Form(description="Guidance scale", ge=0.0, le=20.0)
    ] = 7.5,
) -> Response:
    """
    Upscale an image using Stable Diffusion upscalers.

    The ×4 upscaler is prompt-guided and produces higher quality results.
    The ×2 upscaler uses latent-space upscaling and is faster.
    """
    # Validate scale
    if scale not in (2, 4):
        raise HTTPException(status_code=400, detail="scale must be 2 or 4")

    if scale not in pipelines:
        available = ", ".join(str(s) for s in sorted(pipelines.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"scale {scale} not loaded. Available: {available}",
        )

    # Read and validate image
    try:
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Get pipeline
    pipe = pipelines[scale]

    # Log request
    print(f"Upscaling {input_image.size[0]}×{input_image.size[1]} → ×{scale}")

    # Upscale
    try:
        result = upscale(
            pipe,
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upscaling failed: {e}")

    # Convert result to PNG bytes
    output_buffer = io.BytesIO()
    result.save(output_buffer, format="PNG")
    output_bytes = output_buffer.getvalue()

    print(f"  ✓ Result: {result.size[0]}×{result.size[1]} ({len(output_bytes)} bytes)")

    return Response(content=output_bytes, media_type="image/png")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai3-upscale-daemon",
        description="AI Image Upscaler Daemon",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to listen on (default: 3000)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="both",
        choices=["2", "4", "both"],
        help="Which upscaler(s) to load: 2, 4, or both (default: both)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine which scales to load
    if args.scale == "both":
        scales = [2, 4]
    else:
        scales = [int(args.scale)]

    # Store config in app state for lifespan to access
    app.state.port = args.port
    app.state.scales = scales

    scale_str = "×2 and ×4" if args.scale == "both" else f"×{args.scale}"
    print(f"Starting AI Image Upscaler Daemon on port {args.port}...")
    print(f"Loading: {scale_str} upscaler(s)")
    print(f"OpenAPI docs will be available at http://localhost:{args.port}/docs\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",  # Reduce uvicorn noise, we log ourselves
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

