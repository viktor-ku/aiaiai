/**
 * AI Image Upscaler Client
 *
 * TypeScript client for the ai3-upscale daemon API.
 *
 * @example
 * ```ts
 * import { UpscaleClient } from './client';
 *
 * const client = new UpscaleClient('http://localhost:3000');
 *
 * // Health check
 * const status = await client.ping();
 *
 * // Get available upscale methods
 * const { capabilities } = await client.capabilities(); // e.g. [2, 4]
 *
 * // Upscale an image
 * const file = new File([imageBytes], 'image.png', { type: 'image/png' });
 * const upscaled = await client.upscale(file, { scale: 4, prompt: 'high quality' });
 * ```
 */

/** Upscaling options */
export interface UpscaleOptions {
  /** Upscaling factor: 2 or 4 (default: 4) */
  scale?: 2 | 4;
  /** Text prompt to guide upscaling */
  prompt?: string;
  /** Things to avoid in output (default: "blurry, low quality, artifacts, noise") */
  negativePrompt?: string;
  /** Random seed, 0 = random (default: 0) */
  seed?: number;
  /** Number of denoising steps, 1-100 (default: 20) */
  steps?: number;
  /** Guidance scale, 0-20 (default: 7.5) */
  guidance?: number;
}

/** Ping response */
export interface PingResponse {
  status: string;
}

/** Capabilities response */
export interface CapabilitiesResponse {
  capabilities: (2 | 4)[];
}

/** API error response */
export interface ApiError {
  detail: string;
}

/** Error thrown when API request fails */
export class UpscaleApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string
  ) {
    super(`API error ${status}: ${detail}`);
    this.name = "UpscaleApiError";
  }
}

/**
 * Client for the AI Image Upscaler daemon
 */
export class UpscaleClient {
  private readonly baseUrl: string;

  /**
   * Create a new client
   * @param baseUrl - Base URL of the daemon (e.g., "http://localhost:3000")
   */
  constructor(baseUrl: string = "http://localhost:3000") {
    // Remove trailing slash
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  /**
   * Health check - verify the daemon is running
   * @returns Ping response with status
   */
  async ping(): Promise<PingResponse> {
    const response = await fetch(`${this.baseUrl}/ping`);

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new UpscaleApiError(response.status, error.detail);
    }

    return response.json();
  }

  /**
   * Get available upscale capabilities
   * @returns Available upscale methods (2 and/or 4)
   */
  async capabilities(): Promise<CapabilitiesResponse> {
    const response = await fetch(`${this.baseUrl}/api/capabilities`);

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new UpscaleApiError(response.status, error.detail);
    }

    return response.json();
  }

  /**
   * Upscale an image
   * @param image - Image file or Blob to upscale
   * @param options - Upscaling options
   * @returns Upscaled image as Blob
   */
  async upscale(
    image: File | Blob,
    options: UpscaleOptions = {}
  ): Promise<Blob> {
    const formData = new FormData();

    // Add image
    if (image instanceof File) {
      formData.append("image", image);
    } else {
      formData.append("image", image, "image.png");
    }

    // Add optional parameters
    if (options.scale !== undefined) {
      formData.append("scale", options.scale.toString());
    }
    if (options.prompt !== undefined) {
      formData.append("prompt", options.prompt);
    }
    if (options.negativePrompt !== undefined) {
      formData.append("negative_prompt", options.negativePrompt);
    }
    if (options.seed !== undefined) {
      formData.append("seed", options.seed.toString());
    }
    if (options.steps !== undefined) {
      formData.append("steps", options.steps.toString());
    }
    if (options.guidance !== undefined) {
      formData.append("guidance", options.guidance.toString());
    }

    const response = await fetch(`${this.baseUrl}/upscale`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json();
      throw new UpscaleApiError(response.status, error.detail);
    }

    return response.blob();
  }

  /**
   * Upscale an image from a URL
   * @param url - URL of the image to upscale
   * @param options - Upscaling options
   * @returns Upscaled image as Blob
   */
  async upscaleFromUrl(
    url: string,
    options: UpscaleOptions = {}
  ): Promise<Blob> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch image from ${url}: ${response.status}`);
    }
    const blob = await response.blob();
    return this.upscale(blob, options);
  }

  /**
   * Upscale an image and return as data URL
   * @param image - Image file or Blob to upscale
   * @param options - Upscaling options
   * @returns Upscaled image as data URL string
   */
  async upscaleToDataUrl(
    image: File | Blob,
    options: UpscaleOptions = {}
  ): Promise<string> {
    const blob = await this.upscale(image, options);
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
}

export default UpscaleClient;

