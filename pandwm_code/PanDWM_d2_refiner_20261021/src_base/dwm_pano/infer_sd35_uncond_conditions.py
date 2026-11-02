"""One-time precompute of the unconditional DiT condition pack.

Runs the SD3.5 text encoders on the empty prompt and builds the fixed pano
pseudo-camera conditions through the house `CrossviewTemporalSD.get_conditions`
for a single frame (B=1, T=1, V=2). The saved tensors are sample-independent
constants; the latent-refinement trainer expands them over batch and frames,
so no text encoder is ever loaded during decoder training.

Run (GPU node holding the text encoders):
    PYTHONPATH=src python src/dwm_pano/infer_sd35_uncond_conditions.py \
        --config configs/ctsd/pandwm/..._gs_decoder.json \
        --text-encoder-root /local/<user>/sd35_text_encoders \
        --output /local/<user>/sd35_uncond_conditions.pt
"""
import argparse
import json

import numpy as np
import torch
import transformers

import dwm.common
from dwm_pano.datasets.nuscenes_pano import (
    _make_virtual_pano_K, _make_virtual_pano_extrinsics,
)
from dwm_pano.pipelines.ctsd_pano import CrossviewTemporalSD


def create_parser():
    parser = argparse.ArgumentParser(
        description="Precompute the unconditional DiT condition pack.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="gs_decoder config json carrying the latent_refinement block.")
    parser.add_argument(
        "--text-encoder-root", type=str, required=True,
        help="Directory with SD3.5 text_encoder*/tokenizer* subfolders.")
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output .pt path for the condition pack.")
    parser.add_argument(
        "--latent-downsample", type=int, default=8,
        help="VAE spatial downsample factor for the latent shape.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for the text encoders.")
    return parser


def load_text_stack(root, device):
    tokenizers = [
        transformers.CLIPTokenizer.from_pretrained(root, subfolder="tokenizer"),
        transformers.CLIPTokenizer.from_pretrained(
            root, subfolder="tokenizer_2"),
        transformers.T5TokenizerFast.from_pretrained(
            root, subfolder="tokenizer_3"),
    ]
    text_encoders = [
        transformers.CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder",
            torch_dtype=torch.float16).to(device),
        transformers.CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder_2",
            torch_dtype=torch.float16).to(device),
        transformers.T5EncoderModel.from_pretrained(
            root, subfolder="text_encoder_3",
            torch_dtype=torch.float16).to(device),
    ]
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    return tokenizers, text_encoders


def build_stub_batch(half_height, half_width):
    intrinsics = _make_virtual_pano_K(half_width, half_height)
    extrinsics = np.stack(
        [_make_virtual_pano_extrinsics(0), _make_virtual_pano_extrinsics(1)])
    return {
        "vae_images": torch.zeros(1, 1, 2, 3, half_height, half_width),
        "clip_text": [[["", ""]]],
        "pts": torch.zeros(1, 1, 2),
        "camera_intrinsics": torch.from_numpy(
            np.stack([intrinsics, intrinsics])).float()[None, None],
        "camera_transforms": torch.from_numpy(extrinsics).float()[None, None],
        "ego_transforms": torch.eye(4)[None, None, None],
        "image_size": torch.tensor(
            [[half_width, half_height], [half_width, half_height]],
            dtype=torch.int64)[None, None],
    }


@torch.no_grad()
def main():
    args = create_parser().parse_args()
    device = torch.device(args.device)
    config = json.load(open(args.config))
    refiner_config = config["pipeline"]["latent_refinement"]
    training_config = config["pipeline"]["training_config"]

    tokenizers, text_encoders = load_text_stack(
        args.text_encoder_root, device)
    dit_model = dwm.common.create_instance_from_config(
        refiner_config["model"])

    half_height = training_config["half_height"]
    half_width = training_config["half_width"]
    batch = build_stub_batch(half_height, half_width)
    downsample_shape = (
        1, 1, 2, refiner_config["model"]["in_channels"],
        half_height // args.latent_downsample,
        half_width // args.latent_downsample)

    conditions = CrossviewTemporalSD.get_conditions(
        dit_model, text_encoders, tokenizers,
        refiner_config["common_config"],
        batch["vae_images"].shape, batch, device, torch.float16,
        text_condition_mask=[False],
        explicit_view_modeling_mask=torch.ones(1, dtype=torch.bool),
        latents_shape=downsample_shape)

    pack = {}
    for key, value in conditions.items():
        if torch.is_tensor(value):
            pack[key] = value.cpu()
            print(f"[uncond] {key} {tuple(value.shape)} {value.dtype}")
        else:
            print(f"[uncond] {key} skipped ({type(value).__name__})")
    torch.save(pack, args.output)
    print(f"[uncond] saved {args.output}")


if __name__ == "__main__":
    main()
