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

### Owner feedback to keep (October 2026)

- **Sharp per-view MoGe-2 depth made the 3D worse** than the smoother joint multi-view depth (the earlier
  option that fused the images). Each generated view's own monocular depth disagrees with the others', and
  that pulls the splat apart. Since rc31, training uses the engine's joint depth (DA3 / VGGT) for the
  initial points and the consistency check; MoGe-2 priors are off by default (`depth_prior`).
  MoGe-2 is still used for single-view jobs, for the subject mask, and for the photo's own background in
  subject mode (one view, nothing to disagree with).
- **Studio portraits:** Qwen turns the person ~45° but keeps a plain backdrop as it was. DA3 then measured
  only 5-8° of camera motion and training tore the person apart (rc30 test). Since rc31, subject mode
  finds the cameras from the subject only, and generated views supervise only the subject.
  rc31 test: the subject was found in the photo but not in the AI views (strict criteria), and mixing
  subject-only and whole images gave wrong cameras (79° for a 30° view). Since rc32, AI views use relaxed
  criteria, and candidates without a mask are excluded.
- **rc32 test: still ghosting** (double profile, layered lantern) even with plausible cameras. The owner
  judged that the generated views look fine and the splat builder is the weak link. Per-scene training
  from 4 slightly inconsistent views is the cause. Since rc33, the splat is DA3's feed-forward Gaussian
  head (Nested Giant-Large, already installed), followed by a short polish with 10× lower position
  learning rate. The raw feed-forward splat is exported alongside for comparison.

- **rc33 test: DA3's feed-forward splat did not help either** (raw: blurry, semi-transparent, double face;
  polished: sharper but the same ghosts). Conclusion: no reconstruction method can make one rigid 3D out of
  views that are not 3D-consistent with each other. Qwen draws every angle independently. The owner asked
  for a radical change of strategy.
- **rc36 feedback: back to Qwen views + DA3.** The owner did not like FlashWorld's result. Of everything
  so far, the Qwen views assembled by the multi-view engine gave the most real 3D. Two observations:
  - The polish/training was "so aggressive it erases every detail". The **raw** DA3 splat looked better
    than the polished turntable and the VR view, though ghostly.
  - The raw DA3 splat is still blurry. DA3 works at ~504 px, so each Gaussian covers 2-3 photo pixels.

  rc37 answers each point:
  - **Photo layer.** The photo's part is one Gaussian per photo pixel at training resolution (1280 px in
    Auto), with the photo's own colours. Its depth is DA3's joint depth, sampled bilinearly, with pixels on
    depth jumps left out.
  - **Duplicate removal.** A generated view's Gaussian on, or in front of, a surface the photo shows is
    dropped. Between generated views, the earlier view's copy wins. That removes the ghost copies.
  - **Colour-only polish.** Positions, sizes and rotations stay fixed, with no densification or pruning.
    Colours and opacities are optimised, so a copy the other views contradict fades out.
  - **Per-view image alignment.** Each generated view gets a smooth displacement field of at most 6 % of
    the half-width, with 24 control points along the long side. It is applied to the render before the
    loss, so local misalignments of AI views are absorbed instead of averaged into blur.
  - Qwen is the default engine again. FlashWorld can still be chosen as the generative engine.
- **rc37 feedback: "the raw DA3 splat is the splat".** The owner asked to drop all processing and use the raw
  DA3 splat as the VR output, for every multi-view job (photo + AI views, 360°, multi-photo). Since rc38 the
  worker writes the feed-forward splat as the scene: no training, no cleanup. The photo layer and duplicate
  removal stay, as part of building it. Without DA3, or when DA3 cannot take the views, the assembly asked
  for before is used. The colour polish is still available internally (`ff_polish`), off by default.
- **New strategy (rc34+): generate the 3D directly.** FlashWorld (ICLR 2026 oral, code Apache-2.0, weights
  CC BY-NC-SA 4.0) is fine-tuned from Wan2.2-TI2V-5B. At every denoising step it decodes 3D Gaussians and
  renders them back into the model, so every view comes from one 3D. Inputs: one image, a text prompt and
  a camera path (24 frames). Output: a Gaussian PLY. It needs ~9 GB VRAM with offloading. Downloads:
  model.ckpt 20.9 GB, plus the Wan2.2-TI2V-5B VAE (2.8 GB) and UMT5-XXL text encoder (11.4 GB). Its
  renderer uses the gsplat API available in 1.5.3, so it runs in the recon3d runtime (torch 2.4.1 cu124).

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
| rc23–24 | Automatic retry of weak AI views; rebuild with hand-picked candidates |
| rc25 | Turntable preview video of every multi-view result |
| rc26 | Per-view colour correction (appearance) for generated views; OBJ fused from the trained splat |
| rc27 | Joint camera refinement (SE(3)) of generated views during training |
| rc28 | Quality: 8-step Lightning LoRA for Qwen views; camera refinement for real multi-photo sets |
| rc29 | AbsGS densification (absolute screen-space gradients) |
| rc30 | Fix: DA3 Giant load; VGGT fallback |
| rc34 | FlashWorld: the 3D generated directly (no separate views); Wan TI2V parts; runs in recon3d |
| rc33 | DA3 feed-forward Gaussians (GS head) + gentle polish instead of training from scratch; raw FF splat exported |
| rc32 | Subject masks for AI views (relaxed criteria), maskless candidates excluded, stronger angle check |
| rc31 | Subject mode (cameras from the subject, background from the photo); joint engine depth instead of per-view MoGe-2 priors; turntable around the subject |

### Evaluated and deferred

- **DA3 Gaussian head as initialisation** (done in rc33; SH rotation replaced by DC-only). The Nested model predicts per-pixel Gaussians. It needs `e3nn` for SH
  rotation, and its metric scaling path for Gaussians must be verified on a GPU. Its advantage over the aligned
  MoGe-2 dense init with only 4 widely spaced views is uncertain. Revisit if training from the current init
  shows blur at view boundaries.
- **Intermediate angles (±20°).** The Multiple-Angles LoRA only knows 45° steps; plain instructions without it
  changed pose and gaze (stereo test, rc16). The only continuous-degree LoRA found
  (`berkerdooo/qwen-image-edit-2511-camera-angle-lora`, Apache-2.0) is trained on 512 px Objaverse objects on white
  backgrounds and says real photos with backgrounds need background removal; not suitable for people in scenes.
- **Feed-forward single-view / sparse-view 3DGS (2026: AnySplat, AnchorSplat, SparseSplat, PhGS,
  multi-layer single-image GS).** Re-checked in October 2026. These are faster, but none beats
  "consistent generated views + pose engine + per-scene training" on detail for people in real scenes, and
  most of them have no released Windows-ready weights. PhGS's idea (post-hoc pruning and refinement of a
  feed-forward splat) is already covered by the rc22 cleanup and the training stage.
- **Difix3D+ (NVIDIA, CVPR 2025) — generative repair of novel views.** Single-step SD-Turbo fine-tune
  (`nvidia/difix`, `nvidia/difix_ref` with a reference image) that cleans splat renders. Difix3D+ uses it to
  progressively add repaired pseudo-views during training. Code and weights are under the NVIDIA License
  (non-commercial), and Stability's SD-Turbo community licence also applies. Repo commit `c76edc59`
  (October 2026). It pins `diffusers==0.25.1` and ships a custom pipeline and multi-view UNet (~2.5 k lines);
  the gen runtime has diffusers 0.37.1, so it needs porting or its own runtime. Its 576×1024 working
  resolution is below our 1280–1600 px training views. Planned as an opt-in Advanced step once it can be
  smoke-tested on the target GPU: render ±20° pseudo-views → Difix-ref with the photo → fine-tune.
