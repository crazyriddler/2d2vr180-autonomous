# One photo → the best possible Gaussian splat (research and plan, Oct 2026)

## What the owner's tests showed (RTX 4080 16 GB, 32 GB RAM)

| Approach | Result |
|---|---|
| Single-view depth (MoGe-2, Apple SHARP) | Sharp from the front, but the 3D is a relief: no real volume, scale guessed |
| Multi-view diffusion (Stable Virtual Camera) | People deformed |
| Wan 2.2 video → all frames → trained splat | Better people, but generated frames disagree → blur, broken from above/below |
| Wan / Qwen many views → sharp fusion | Sharp per view, but seams and inconsistent layers between views |
| **Photo + a few views 45° left / 45° right / above → multi-view (VGGT) → trained splat** | **Best: real volume, correct scale** — camera motion gives parallax, triangulation gives metric 3D |

So the base is: **a few strongly separated, consistent views + multi-view reconstruction + training**. Everything
below improves one link of that chain.

## The chain and the state of the art for each link

1. **View generation** (consistency is everything)
   - Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA (Apache-2.0): trained on Gaussian-splat renders → rigid camera
     moves; sharp ~1 MP stills. Best fit for 16 GB with GGUF + block streaming. *In use.*
   - No LoRA guarantees zero non-rigid change (pose, gaze, expression). Industry practice (VidSplat, ReconX,
     "Pseudo-View Enhancement via Confidence Fusion", 2026) is to **measure consistency and down-weight what does
     not fit**, not to trust the generator. → best-of-N selection (rc17) and **per-pixel confidence maps** (next).
2. **Poses + depth**
   - VGGT-1B (CC BY-NC) *in use*. **Depth Anything 3** (ByteDance, Nov 2025): +35.7 % pose accuracy and +23.6 %
     geometry vs VGGT on its benchmark; `DA3NESTED-GIANT-LARGE-1.1` adds metric scale (CC BY-NC 4.0, 1.4 B);
     `DA3-BASE`/`DA3-SMALL` Apache-2.0. Heavy optional deps (xformers, e3nn, pycolmap) – inference path to be
     vendored minimally. MapAnything (Apache-2.0 variant) and π³ (weights CC BY-NC) are alternatives.
3. **Sparse-view training** (4 views is "few-shot" territory)
   - Monocular depth priors as regularisers (FSGS, D²GS, HBSplat, 2024-26): keep each view's rendered depth close to
     its (scale-aligned) MoGe-2 depth → no floaters, correct surfaces between views.
   - Dense initialisation from the aligned depth maps (instead of sparse VGGT points) → sharp from step 0.
   - **Confidence-weighted photometric loss** for generated views: pixels that disagree with the photo's
     reprojection get low weight (VidSplat / ReconX "3D confidence-aware optimisation").
   - The photo itself keeps full weight and higher sampling probability (already).

## Plan (releases)

| Release | Content |
|---|---|
| rc18 | Simplified interface: one "What to make" choice with the recommended flow as default; everything else under *Advanced* |
| rc19 | Training: per-pixel confidence maps for generated views + MoGe-2 depth prior + dense sharp initialisation |
| rc20 | Depth Anything 3 as pose/depth engine (Nested Giant-Large 1.1, metric), VGGT kept as fallback |
| next | View consistency: Qwen conditioned on the photo *and* the previously accepted view; symmetric low view option; automatic retry of views whose confidence is too low |

All inference stays local; licences are recorded per model (VGGT/DA3-Giant non-commercial, used under the
personal/research profile).

## Status (updated as releases ship)

| Release | Done |
|---|---|
| rc18 | Simplified Create page with presets |
| rc19 | Confidence maps for generated views, MoGe-2 depth priors, dense sharp initialisation |
| rc20 | Depth Anything 3 Nested Giant-Large 1.1 camera engine (VGGT fallback) |
| rc21 | Photo + 3 views trained at photo detail (1280/1600 px) with SH degree 1; inconsistent views dropped |
| rc22 | Post-training cleanup (unseen / transparent / oversized / isolated splats); AI views overview sheet |

### Evaluated and deferred

- **DA3 Gaussian head as initialisation.** The Nested model predicts per-pixel Gaussians. It needs `e3nn` for SH
  rotation, and its metric scaling path for Gaussians must be verified on a GPU. Its advantage over the aligned
  MoGe-2 dense init with only 4 widely spaced views is uncertain. Revisit if training from the current init
  shows blur at view boundaries.
- **Intermediate angles (±20°).** The Multiple-Angles LoRA only knows 45° steps; plain instructions without it
  changed pose and gaze (stereo test, rc16). The only continuous-degree LoRA found
  (`berkerdooo/qwen-image-edit-2511-camera-angle-lora`, Apache-2.0) is trained on 512 px Objaverse objects on white
  backgrounds and says real photos with backgrounds need background removal; not suitable for people in scenes.
