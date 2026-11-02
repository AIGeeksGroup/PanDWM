# Third-party notices

This repository contains code from the following projects. Their licenses apply to the corresponding files.

| Component | Location in this repository | License |
|---|---|---|
| [OpenDWM](https://github.com/SenseTime-FVG/OpenDWM) (`dwm` package, configs, training framework) | `dwm/` inside each code tree, `pandwm_code/PanDWM-gs-aa-ft/` | MIT, Copyright (c) 2024 SenseTime Research (`LICENSE`) |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | `pandwm_code/*/vendor/lpips/`, `pandwm_code/PanDWM_long_horizon_20260928/lpips/` | BSD-2-Clause (`LPIPS-LICENSE`) |
| [Causal Forcing](https://github.com/thu-ml/Causal-Forcing) reference files | `dwm_pano/pipelines/causal_forcing_official_reference/` | Apache-2.0 |

External dependencies that are not included and must be obtained separately:

| Component | Use | License |
|---|---|---|
| [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | VAE and text encoders | Stability AI Community License |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) code and weights | Range relabelling | see the repository and model cards |
| [TATS](https://github.com/songweige/TATS) `tats/fvd` and I3D weights | FVD | MIT |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Gaussian rasterization | Apache-2.0 |
| [nuScenes](https://www.nuscenes.org/) | Data | nuScenes terms of use |
