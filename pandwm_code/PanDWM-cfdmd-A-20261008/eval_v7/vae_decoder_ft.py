"""Fine-tune the SD3.5 VAE decoder on nuScenes training panoramas (2026-10-12).

Why: the VAE round trip of the real frames already caps sharpness at 314 (GT 329, -4.6%), so every student sits at or
below that ceiling.  The encoder stays frozen, so the latent space the DiT students were trained in is unchanged and
every saved student latent (latents.pt) can be decoded by the new decoder without retraining anything.
Losses, after the taming / SD autoencoder recipe (no KL term: it belongs to the frozen encoder):
  L1 + perceptual_weight * perceptual + d_weight * hinge-GAN (from --disc-start)
  perceptual = VGG16 relu1_2/2_2/3_3/4_3/5_3 features, ImageNet-normalised, channel-unit-normalised squared
               differences summed over channels and averaged over space, layers summed with uniform weights
               (LPIPS without its learned linear heads; the lpips package is not installed offline);
  GAN = taming NLayerDiscriminator (3 layers, BatchNorm) with hinge loss and taming's adaptive weight
        (|grad rec| / |grad adv| at decoder.conv_out, times --disc-weight).
Latent-noise augmentation (LV-RAE, off by default since run 1): half the crops get Gaussian noise of std
U(0, --latent-noise) on the scaled latent before decoding, so the decoder also learns to decode slightly off-manifold
(generated) latents.  It also teaches the decoder to denoise, which smooths, and sharpness is the goal here.
Run 1 (lr 1e-5, Adam betas 0.5/0.9, no warmup, latent noise 0.05) lost sharpness and PSNR in its first 250 steps
(batch sharpness ratio 0.95 -> ~0.75, PSNR 35.4 -> ~33.6 dB), so it was stopped.  Now: lr 2e-6 with a linear
warmup, betas 0.9/0.999, and every log line compares with a frozen copy of the original decoder on the same batch
(psnr_gain, sharp_gain), which removes the batch-to-batch noise of the raw numbers.
Run 2 (that recipe with perceptual weight 0.2 and the GAN from step 1500; frozen copy in vae_decoder_ft_run2.py)
gained ~1 dB PSNR over the original decoder on the same batches but lost ~20% sharpness within 50 steps (sharp_gain
~0.8 through step 1150): L1 with a weak perceptual term regresses ambiguous detail to its mean (the
perception-distortion trade-off; Stability's sd-vae-ft-mse, MSE + 0.1 LPIPS, is likewise "smoother").  Run 3 goes
back to the autoencoder's own balance, L1 + 1.0 perceptual + adaptive GAN, and has the GAN on from the first decoder
update: the discriminator trains alone for --d-head-start steps against the frozen original decoder
(--freeze-decoder-until), then both train together.  An EMA copy of the decoder (--ema) is logged and saved too.
Training data: random 256 x --crop-w crops of the 256 x 2304 half panoramas of single train-split frames
(PanoNuScenesDataset with sequence_length 1 and only the RGB key, same slicing and resize as the evaluation).
  python vae_decoder_ft.py --train-config <cfdmd config> --vae <sd3.5 dir> --out <dir> [--steps 6000 ...]
Writes <out>/decoder_<step>.pt and decoder_ema_<step>.pt (decoder state_dicts, fp32), <out>/train_log.jsonl and
DECFT_DONE at the end.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import dwm.common


def frames_dataset(train_config):
    cfg = json.loads(Path(train_config).read_text())
    td = copy.deepcopy(cfg["training_dataset"])
    base = td["base_dataset"]
    base["sequence_length"] = 1
    for key in ("enable_3dbox", "enable_hdmap", "enable_camera_transforms", "enable_ego_transforms",
                "enable_description"):
        base[key] = False
    td["transform_list"] = [t for t in td["transform_list"] if t["old_key"] == "images"]
    return dwm.common.create_instance_from_config(td)


class Crops(torch.utils.data.Dataset):
    """Both halves of one frame, each randomly cropped to crop_w (8-pixel aligned): (V, 3, 256, crop_w) in [0, 1]."""

    def __init__(self, base, crop_w):
        self.base, self.crop_w = base, crop_w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x = self.base[index]["vae_images"][0]
        out = []
        for v in range(x.shape[0]):
            offset = int(torch.randint(0, (x.shape[-1] - self.crop_w) // 8 + 1, ())) * 8
            out.append(x[v, :, :, offset:offset + self.crop_w])
        return torch.stack(out)


class Perceptual(nn.Module):
    TAPS = (3, 8, 15, 22, 29)

    def __init__(self, weights_path):
        super().__init__()
        vgg = torchvision.models.vgg16()
        vgg.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.features = vgg.features[:self.TAPS[-1] + 1].eval()
        self.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        """x, y in [-1, 1]."""
        h = (torch.cat([x, y]) * 0.5 + 0.5 - self.mean) / self.std
        loss = h.new_zeros(())
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.TAPS:
                a, b = h.chunk(2)
                a = a / (a.norm(dim=1, keepdim=True) + 1e-10)
                b = b / (b.norm(dim=1, keepdim=True) + 1e-10)
                loss = loss + (a - b).square().sum(dim=1).mean()
        return loss


def patch_discriminator(ndf=64, n_layers=3):
    layers = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
    mult = 1
    for n in range(1, n_layers):
        prev, mult = mult, min(2 ** n, 8)
        layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * mult),
                   nn.LeakyReLU(0.2, True)]
    prev, mult = mult, min(2 ** n_layers, 8)
    layers += [nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1, bias=False), nn.BatchNorm2d(ndf * mult),
               nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * mult, 1, 4, 1, 1)]
    disc = nn.Sequential(*layers)
    for m in disc.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
    return disc


def laplacian_var(x):
    """Mean Laplacian variance of the grey image in [0, 255]; x in [-1, 1], (N, 3, H, W)."""
    g = ((x.clamp(-1, 1) * 0.5 + 0.5) * 255 * torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
         ).sum(1, keepdim=True)
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device).view(1, 1, 3, 3)
    return F.conv2d(g, k).flatten(1).var(dim=1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-config", required=True)
    p.add_argument("--vae", required=True, help="stable-diffusion-3.5-medium directory (subfolder vae)")
    p.add_argument("--vgg", default="/path/to/pandwm_root/.cache/torch/hub/checkpoints/vgg16-397923af.pth")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=4, help="frames per step (two half-panorama crops each)")
    p.add_argument("--crop-w", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--warmup", type=int, default=300, help="decoder lr warmup, counted in decoder updates")
    p.add_argument("--perceptual-weight", type=float, default=1.0)
    p.add_argument("--freeze-decoder-until", type=int, default=500, help="no decoder update up to this step")
    p.add_argument("--disc-start", type=int, default=500, help="adversarial term on the decoder after this step")
    p.add_argument("--d-head-start", type=int, default=500, help="discriminator updates start this much earlier")
    p.add_argument("--disc-weight", type=float, default=0.5)
    p.add_argument("--disc-lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999, help="decoder EMA decay (0: off)")
    p.add_argument("--latent-noise", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Workers are forked before CUDA is touched.
    data = Crops(frames_dataset(args.train_config), args.crop_w)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True,
                                         num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                                         prefetch_factor=4 if args.num_workers > 0 else None)
    batches = iter(loader)
    print(f"DECFT frames={len(data)} batch={args.batch}x2 crops of 256x{args.crop_w} steps={args.steps}", flush=True)

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)
    vae.train()
    vae.encoder.eval()
    shift, scale = vae.config.shift_factor or 0.0, vae.config.scaling_factor
    ref_decoder = copy.deepcopy(vae.decoder).eval().requires_grad_(False)     # the original decoder, for logging
    ema = copy.deepcopy(vae.decoder).eval().requires_grad_(False) if args.ema > 0 else None
    perc = Perceptual(args.vgg).to(device)
    disc = patch_discriminator().to(device)
    dec_params = [q for q in vae.decoder.parameters() if q.requires_grad]
    opt = torch.optim.Adam(dec_params, lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / max(args.warmup, 1)))
    dopt = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, betas=(0.5, 0.9))
    print(f"DECFT decoder params={sum(q.numel() for q in dec_params) / 1e6:.1f}M", flush=True)

    log = open(args.out / "train_log.jsonl", "a")
    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            x = next(batches)
        except StopIteration:
            batches = iter(loader)
            x = next(batches)
        x = x.to(device, non_blocking=True).flatten(0, 1) * 2 - 1           # (N, 3, 256, crop_w) in [-1, 1]
        with torch.no_grad():
            z = (vae.encode(x).latent_dist.mode() - shift) * scale
            if args.latent_noise > 0:
                pick = (torch.rand(z.shape[0], 1, 1, 1, device=device) < 0.5).float()
                sigma = torch.rand(z.shape[0], 1, 1, 1, device=device) * args.latent_noise * pick
                z = z + sigma * torch.randn_like(z)
        train_dec = step > args.freeze_decoder_until
        with torch.set_grad_enabled(train_dec):
            xr = vae.decode(z / scale + shift).sample
            l1 = (xr - x).abs().mean()
            lp = perc(xr, x)
            rec = l1 + args.perceptual_weight * lp
        use_gan = train_dec and step > args.disc_start
        g_adv = torch.zeros((), device=device)
        d_weight = 0.0
        if use_gan:
            g_adv = -disc(xr).mean()
            last = vae.decoder.conv_out.weight
            g_rec = torch.autograd.grad(rec, last, retain_graph=True)[0]
            g_gan = torch.autograd.grad(g_adv, last, retain_graph=True)[0]
            d_weight = float((g_rec.norm() / (g_gan.norm() + 1e-4)).clamp(0.0, 1e4)) * args.disc_weight
        if train_dec:
            loss = rec + d_weight * g_adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for q_ema, q in zip(ema.parameters(), vae.decoder.parameters()):
                        q_ema.lerp_(q, 1.0 - args.ema)
        d_loss = torch.zeros((), device=device)
        if step > args.disc_start - args.d_head_start:
            d_real, d_fake = disc(x), disc(xr.detach())
            d_loss = 0.5 * (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean())
            dopt.zero_grad(set_to_none=True)
            d_loss.backward()
            dopt.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                xref = ref_decoder(z / scale + shift)
                mse = ((xr.clamp(-1, 1) - x) * 0.5).square().mean()
                mse_ref = ((xref.clamp(-1, 1) - x) * 0.5).square().mean()
                lv, lv_ref, lv_gt = laplacian_var(xr), laplacian_var(xref), laplacian_var(x)
                row = {"step": step, "l1": round(float(l1), 5), "perc": round(float(lp), 5),
                       "g_adv": round(float(g_adv), 4), "d_loss": round(float(d_loss), 4),
                       "d_weight": round(d_weight, 4), "psnr": round(float(-10 * torch.log10(mse)), 3),
                       "psnr_gain": round(float(10 * torch.log10(mse_ref / mse)), 3),
                       "sharp_ratio": round(float(lv / lv_gt), 4), "sharp_ref": round(float(lv_ref / lv_gt), 4),
                       "sharp_gain": round(float(lv / lv_ref), 4), "lr": opt.param_groups[0]["lr"],
                       "s_per_step": round((time.monotonic() - t0) / step, 3)}
                if ema is not None:
                    xema = ema(z / scale + shift)
                    mse_ema = ((xema.clamp(-1, 1) - x) * 0.5).square().mean()
                    row["psnr_gain_ema"] = round(float(10 * torch.log10(mse_ref / mse_ema)), 3)
                    row["sharp_gain_ema"] = round(float(laplacian_var(xema) / lv_ref), 4)
            print("DECFT " + json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            torch.save(vae.decoder.state_dict(), args.out / f"decoder_{step}.pt")
            if ema is not None:
                torch.save(ema.state_dict(), args.out / f"decoder_ema_{step}.pt")
            print(f"DECFT_SAVED step={step}", flush=True)
    print("DECFT_DONE", flush=True)


if __name__ == "__main__":
    main()
