# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import time
import subprocess
from typing import List
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel

from cog import BasePredictor, Input, Path

MODEL_CACHE = "weights_cache"
PRIOR_URL = "https://weights.replicate.delivery/default/kandinsky-2-2/models--kandinsky-community--kandinsky-2-2-prior.tar"
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

        # Download the prior model if not available locally
        prior_path = os.path.join(
            MODEL_CACHE, "models--kandinsky-community--kandinsky-2-2-prior"
        )
        if not os.path.exists(prior_path):
            download_weights(PRIOR_URL, prior_path)

        # Download the decoder model if not available locally
        decoder_path = os.path.join(
            MODEL_CACHE, "models--kandinsky-community--kandinsky-2-2-decoder"
        )
        if not os.path.exists(decoder_path):
            download_weights(DECODER_URL, decoder_path)

        device = torch.device("cuda:0")

        self.negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(
                "kandinsky-community/kandinsky-2-2-prior",
                torch_dtype=torch.float16,
                subfolder="image_encoder",
                cache_dir=MODEL_CACHE,
                local_files_only=True,
            )
            .half()
            .to(device)
        )
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
        self.prior = KandinskyV22PriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(device)
        self.decoder = KandinskyV22Pipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder",
            unet=unet,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(device)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A moss covered astronaut with a black background",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if hits memory limits.",
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
            description="Height of output image. Lower the setting if hits memory limits.",
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
            description="Number of denoising steps", ge=1, le=500, default=75
        ),
        num_inference_steps_prior: int = Input(
            description="Number of denoising steps for priors", ge=1, le=500, default=25
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Output image format",
            choices=["webp", "jpeg", "png"],
            default="webp",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        if negative_prompt is not None:
            negative_prior_prompt = negative_prompt + self.negative_prior_prompt
        else:
            negative_prior_prompt = self.negative_prior_prompt

        img_emb = self.prior(
            prompt=prompt,
            num_inference_steps=num_inference_steps_prior,
            num_images_per_prompt=num_outputs,
        )

        negative_emb = self.prior(
            prompt=negative_prior_prompt,
            num_inference_steps=num_inference_steps_prior,
            num_images_per_prompt=num_outputs,
        )
        output = self.decoder(
            image_embeds=img_emb.image_embeds,
            negative_image_embeds=negative_emb.image_embeds,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.{output_format}"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
