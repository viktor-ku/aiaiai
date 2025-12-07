---
name: sdxl-pose-copying
overview: Add pose-copying image generation to the existing SDXL text-to-image helper by introducing a ControlNet-based SDXL pipeline that can take either a plain photo or a pre-generated pose map as conditioning input.
todos:
  - id: define-pose-class
    content: Define a `Pose` configuration class holding image/image_path, source type, and strength to drive pose-controlled SDXL generation.
    status: pending
  - id: add-sdxl-controlnet
    content: Extend `ai/text2image/sdxl.py` so `make_pipe()` builds both base SDXL and a ControlNet pose pipeline, wrapped into a single returned pipe object.
    status: pending
    dependencies:
      - define-pose-class
  - id: add-pose-preprocessing
    content: Implement internal helpers that convert a `Pose` instance into a ControlNet-ready conditioning image (handling both photo and pre-made pose maps).
    status: pending
    dependencies:
      - add-sdxl-controlnet
  - id: extend-snap-with-pose
    content: "Extend `snap` in `ai/text2image/sdxl.py` to accept `pose: Pose | None` and branch between base or ControlNet SDXL pipelines accordingly, keeping existing behavior unchanged when `pose` is `None`."
    status: pending
    dependencies:
      - add-pose-preprocessing
  - id: demo-pose-script
    content: Optionally add a small CLI/demo script that constructs a `Pose` and calls `snap(..., pose=pose)` for pose-locked generation from a reference image.
    status: pending
    dependencies:
      - extend-snap-with-pose
---

# SDXL Pose Copying with ControlNet (Single `snap` API + `Pose` object)

### Goal

Add **pose-copying** image generation to your SDXL helper while keeping the public API centered on:

- `make_pipe()` – construct and return a pipe that internally knows about both base SDXL and pose-aware SDXL.
- `snap(pipe, prompt, ..., pose: Pose | None = None, **other_kwargs)` – one function that handles both plain and pose-controlled generations based on an optional `Pose` object.

### High-level Approach

- **Keep existing usage unchanged**: all current calls without a `pose` argument behave exactly as today.
- Introduce a small **`Pose` configuration class** that encapsulates the pose-control options (reference image, source type, strength, etc.).
- Internally extend `make_pipe()` and `snap()` to support ControlNet-based pose conditioning when a `Pose` instance is provided.

### Step 1: Define the `Pose` configuration class

- Add a `Pose` class (either in `ai/text2image/sdxl.py` or a closely related module) that holds:
- **Reference image** as either:
- `image: PIL.Image.Image | None`
- `image_path: str | None`
- **Source type**: `source: Literal["photo", "pose_map"] = "photo"` to distinguish raw person photos vs pre-made pose maps.
- **Pose strength**: `strength: float = 1.0` (mapped to ControlNet conditioning scale).
- Room for a couple of extra knobs later (e.g. `resize_mode` or `debug` flags) without changing the `snap` signature.
- Provide a simple `__init__` and optional helper methods (e.g. a method to resolve `image_path` to `PIL.Image` lazily).

### Step 2: Internal SDXL + ControlNet setup

- Ensure dependencies (planning only): `diffusers` with SDXL ControlNet support, plus `controlnet-aux` for pose extraction.
- In `ai/text2image/sdxl.py`:
- Import `ControlNetModel` and `StableDiffusionXLControlNetPipeline` from `diffusers`.
- Define constants for the base SDXL model and pose ControlNet checkpoint IDs.
- Update `make_pipe()` to:
- Build the existing base SDXL pipeline as it does now.
- Build an SDXL ControlNet pipeline for pose control.
- Wrap both in a small internal container object (e.g. with `.base` and `.pose` attributes) that is returned as `pipe`.

### Step 3: Pose preprocessing helpers (driven by `Pose`)

- Add an internal helper such as `_pose_to_conditioning_image(pose: Pose, width: int, height: int)` that:
- Resolves the reference image from either `pose.image` or `pose.image_path`.
- For `pose.source == "photo"`:
- Runs pose detection using `controlnet-aux` (OpenPose or similar) on the RGB photo to produce a pose map image.
- For `pose.source == "pose_map"`:
- Treats the resolved image as an existing pose/skeleton map.
- Normalizes the pose map (RGB, resized to `width`/`height`) for ControlNet input.

### Step 4: Extend `snap` to accept `pose: Pose | None`

- Update the `snap` function in `ai/text2image/sdxl.py` to have signature along the lines of:
- `snap(pipe, prompt: str, negative_prompt: str = "", seed: int = 0, width: int = 768, height: int = 1024, steps: int = 20, guidance: float = 4.5, pose: Pose | None = None)`.
- Inside `snap`:
- If `pose is None`:
- Use the base SDXL pipeline and keep behavior identical to the current implementation.
- If `pose is not None`:
- Use `_pose_to_conditioning_image(pose, width, height)` to get the conditioning pose image.
- Call the ControlNet SDXL pipeline with:
- `prompt`, `negative_prompt`, `height`, `width`, `num_inference_steps`, `guidance_scale`.
- `image` set to the pose conditioning image.
- `controlnet_conditioning_scale` set from `pose.strength`.
- Return the resulting `PIL.Image` just like in the non-pose path.

### Step 5: Backwards compatibility and ergonomics

- Ensure all existing positional parameters and defaults remain unchanged so current call sites work without modification.
- Make `pose` strictly optional with a default of `None`.
- Document the `Pose` class and `snap` usage patterns:
- Plain text-to-image (no `pose`).
- Pose-copy mode via `snap(pipe, prompt=..., pose=Pose(image=..., source="photo", strength=1.0))` or using `image_path` instead of `image`.

### Step 6: (Optional) simple usage script

- Optionally create a small CLI script (e.g. `chat/pose_copy.py`) that demonstrates:
- Loading `pipe = make_pipe()`.
- Building `pose = Pose(image_path="/path/to/ref.jpg", source="photo", strength=1.0)`.
- Calling `snap(pipe, prompt=..., pose=pose)` and saving the output.
- Use this to manually test pose-copy quality and tune default `strength`, guidance, and step count.