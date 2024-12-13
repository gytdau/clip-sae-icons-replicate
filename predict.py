# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess
from typing import List, Optional
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from diffusers.models import UNet2DConditionModel
import tempfile
import uuid

from cog import BasePredictor, Input, Path

MODEL_CACHE = "weights_cache"
# PRIOR_URL = "https://weights.replicate.delivery/default/kandinsky-2-2/models--kandinsky-community--kandinsky-2-2-prior.tar"
DECODER_URL = "https://weights.replicate.delivery/default/kandinsky-2-2/models--kandinsky-community--kandinsky-2-2-decoder.tar"

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE


def download_weights(url, dest):
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    try:
        subprocess.check_call(["pget", "-xvf", url, dest], close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # Download the decoder model if not available locally
        decoder_path = os.path.join(
            MODEL_CACHE, "models--kandinsky-community--kandinsky-2-2-decoder"
        )
        if not os.path.exists(decoder_path):
            download_weights(DECODER_URL, decoder_path)

        device = torch.device("cuda:0")

        unet = (
            UNet2DConditionModel.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder",
                torch_dtype=torch.float16,
                subfolder="unet",
                cache_dir=MODEL_CACHE,
                local_files_only=True,
            )
            .half()
            .to(device)
        )
        self.decoder = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            unet=unet,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(device)

        self.base_embed = torch.load("base_embeds.pt")[0].to(device)
        self.zero_embed = torch.load("zero_embed.pt").to(device)

    def predict(
        self,
        embeddings: List[List[float]] = Input(
            description="Input embeddings, one per icon to generate. Each embedding should be a list of floats. Note that specifying multiple embeddings only guarantees deterministic behavior for the first embedding.",
        ),
        add_base_embedding: bool = Input(
            description="Whether to add the base embedding to the input embedding",
            default=True,
        ),
        width: int = Input(
            description="Width of output image",
            choices=[
                384,
                512,
                576,
                640,
                704,
                768,
                960,
                1024,
                1152,
                1280,
                1536,
                1792,
                2048,
            ],
            default=512,
        ),
        height: int = Input(
            description="Height of output image",
            choices=[
                384,
                512,
                576,
                640,
                704,
                768,
                960,
                1024,
                1152,
                1280,
                1536,
                1792,
                2048,
            ],
            default=512,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=500,
            default=12,
        ),
        seed: int = Input(
            description="Seed for randomness. By default, 1 is used.",
            default=1,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        output_paths = []

        # Add validation for embeddings
        if not embeddings:
            raise ValueError("No embeddings provided")

        # Convert input embedding to tensor with error handling
        try:
            embeds = torch.tensor(embeddings, device="cuda").reshape(
                len(embeddings), -1
            )
        except Exception as e:
            raise ValueError(f"Failed to process embeddings: {str(e)}")

        # Validate embedding dimensions
        expected_dim = self.base_embed.shape[-1]  # Get expected embedding dimension
        if embeds.shape[-1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_dim}, got {embeds.shape[-1]}"
            )

        # Add base embedding if requested
        if add_base_embedding:
            embeds = embeds + self.base_embed

        output = self.decoder(
            image_embeds=embeds,
            negative_image_embeds=[self.zero_embed] * len(embeds),
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
        )

        # Save each generated image
        for i, sample in enumerate(output.images):
            try:
                output_path = os.path.join(
                    tempfile.gettempdir(), f"generated_{uuid.uuid4()}.webp"
                )
                sample.save(output_path, "WEBP", quality=85)
                output_paths.append(Path(output_path))
            except Exception as e:
                print(f"Failed to save image {i}: {str(e)}")
                continue

        return output_paths
