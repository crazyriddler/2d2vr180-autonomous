# From rc19 to the most precise splat from one photo (research, October 2026)

Target machine: RTX 4080 16 GB, 32 GB RAM, Windows 11. Everything runs locally; licences do not limit the
choice (personal use), but every model must run on this machine.

## 1. What rc19 does in "Real 3D from one photo" (the baseline the owner judged best)

1. **Views.** Qwen-Image-Edit-2511 (GGUF Q5, 4-step Lightning LoRA) with the fal Multiple-Angles LoRA (strength
   0.9) draws 3 views: 45° left, 45° right and a 30° high angle, each about 1 MP. Auto makes 2 candidates per angle,
   Quality 3; the most rigid one (a pure camera move of the photo, measured with VGGT) is kept.
2. **Cameras and depth.** VGGT-1B poses the photo and the 3 views together, and gives depth and confidence.
3. **Scale.** MoGe-2 ViT-L measures metric depth on the photo; VGGT's scene is rescaled to it.
4. **Depth priors.** MoGe-2 depth of every view, at training resolution, aligned to VGGT's depth.
5. **Initialisation.** Dense points from those aligned depth maps, coloured by their own pixels.
6. **Confidence maps.** Pixels of a generated view that contradict the photo's reprojection get low weight
   (floor 0.1). Pixels the photo does not see keep full weight.
7. **Training.** gsplat 3DGS, 9000 steps in Auto, all images at most 1024 px, SH degree 3. The photo is sampled
   4× more often than any view. A depth prior (relative L1, weight 0.1, fading out by 60 % of the steps) and
   standard densification are used.

## 2. Why it looks the way it does

| Owner's observation | Cause in rc19 |
|---|---|
| Best geometry so far | Dense, sharp initialisation from MoGe-2, a depth prior while the views are still being fitted, and a photo-dominated loss: the shape follows the photo's own depth and the views only add what the photo cannot see. |
| Texture "not fused well", noisy | **SH degree 3 with 4 views.** 48 colour coefficients per splat let each splat show a different colour to each of the 4 cameras: the views' small inconsistencies are memorised as view-dependent colour, which appears as noise and flicker between the cameras. **1024 px cap** on the photo too: fine detail of a 12–48 MP photo is thrown away before training. |
| No back of a person, ever | No camera ever looks at it. The 3 views stay within ±45°/+30°, so the back has no supervision and no initial points. Nothing removes it — it never existed. |
| Later cleanup lost detail | rc22+ pruned splats by opacity, size and isolation, and rc31+ masked the backgrounds of the views. Both removed true detail together with floaters. Not to be repeated. |

## 3. The state of the art for each link, and what fits 16 GB on Windows

### 3.1 Views of other angles (consistency decides everything)

- **Qwen-Image-Edit-2511 + Multiple-Angles LoRA** — in use. Trained on Gaussian-splat renders, so it moves
  the camera rigidly. It knows 8 azimuths at 45° steps × 4 elevations (−30, 0, 30, 60). 90°, 135° and 180° are
  supported, but untested in rc19.
- **Multi-reference conditioning.** `QwenImageEditPlusPipeline` takes several input images. Conditioning a view
  on the photo **and the nearest accepted view** (e.g. 135° on the photo + 90°) keeps clothes, hair and
  proportions consistent around the subject. This is the cheapest consistency gain, with no new model.
- **Qwen-Image-2.1** (Alibaba, September 2026; research licence). 7B unified generation and editing, up to 10
  reference images, native 2048 px. A community Multiple-Angles LoRA for it appeared on 2026-10-07 (12
  azimuths at 30° steps, 4 elevations). Smaller than the 20B 2511, so it fits 16 GB better, and at a higher
  resolution. To verify on the CI resolver (huggingface.co is not reachable from the dev container), then
  evaluate against 2511.
- **Video models (Wan, SEVA, Lyra, GEN3C, Voyager).** Lyra 1/2 (NVIDIA, single image → 3DGS by video
  self-distillation) and GEN3C / HunyuanWorld-Voyager build on 7B–14B video models. They need 40–80 GB, which
  is not realistic on 16 GB. Wan 2.2 5B is already integrated for other presets.

### 3.2 Complete subject (back, sides): 3D generators

| Model | Output | Fits RTX 4080 / Windows? | Note |
|---|---|---|---|
| SAM 3D Objects (Meta, Nov 2025) | Gaussians + layout in the photo's frame | No: 32 GB, Linux, gated | Ideal in principle |
| TRELLIS.2-4B (Microsoft, Dec 2025, MIT) | Mesh + PBR textures, 512³–1536³ | Yes via the `TRELLIS.2-stableprojectorz` Windows fork (Torch 2.8, CUDA 12.8, "optimised for 8 GB") | Best open shape quality |
| TRELLIS-image-large (v1, MIT) | Gaussians + mesh | 16 GB; Windows community installers | Lower detail than v2 |
| LHM (Alibaba, ICCV 2025) | Animatable human Gaussians | Not verified | Humans only |

**Plan:** an optional "complete subject" component. TRELLIS.2 builds the subject from the photo (mask). Its
mesh is aligned to the photo's MoGe-2 depth of the subject (scale, rotation, translation). It is then turned
into Gaussians:
- the back and sides keep TRELLIS's texture;
- the front takes the photo's own pixels (exact detail).

This gives a real back. The Qwen 180° view gives a plausible back too, but only one image of it; TRELLIS gives
consistent geometry all around.

### 3.3 Cameras, depth and scale

- **VGGT-1B** — in use in rc19. Depth Anything 3 (rc20–rc39) was not judged better by the owner, so it stays
  out. π³ and MapAnything are alternatives with similar accuracy; no change for now.
- **MoGe-2 ViT-L** — the most accurate open metric monocular geometry (2025–26). It accepts a known
  horizontal field of view (`fov_x`). Feeding it the photo's real focal length from **EXIF**
  (`FocalLengthIn35mmFilm`, or `FocalLength` + sensor width) removes the largest single source of scale and
  shape error: a wrong FOV stretches or flattens the whole scene in depth.
- **HunyuanWorld-Mirror** (Tencent, Oct 2025) takes camera, intrinsics and depth priors and outputs Gaussians.
  Its licence excludes the EU (the owner is in Spain); not used.

### 3.4 Training settings (few views, 3DGS)

- **SH degree 1** for photo + few views. View-dependent colour beyond degree 1 cannot be learned from 4–6
  views; it only memorises inconsistencies.
- **Train the photo at its own detail:** 1600 px in Auto and 2048 px in Quality. Generated views stay at their
  native ~1 MP (Qwen's output size), because enlarging them adds nothing.
- Steps: 9000 (Auto) is enough at 1024 px. With the photo at 1600–2048 px and more views, 12000 (Auto) and
  20000 (Quality).
- Keep: the depth prior, the dense initialisation, the confidence maps, 4× photo sampling, standard
  densification. **No** post-training pruning.

### 3.5 Texture after training ("properly calculated textures")

- **Best-view colour baking.** After the geometry is trained, every visible splat takes its colour from the view
  that sees it most frontally and at the highest resolution. The photo wins wherever it sees the splat;
  colours are not averaged across inconsistent views. This is what photogrammetry texturing does for meshes.
  It changes colours only, so no geometry or detail is removed.
- **Difix3D+ (NVIDIA)** — repairs renders of novel views with the photo as reference, then fine-tunes on them.
  Non-commercial licence; diffusers 0.25 pin, needs its own runtime. Later, opt-in.

## 4. Roadmap (each step a release, built on rc19)

| Release | Change | Expected effect |
|---|---|---|
| rc41 | SH degree 1; photo trained at 1600/2048 px with the views at native size; more steps; EXIF focal length → MoGe-2 FOV | Less texture noise, the photo's real detail, more accurate scale and depth |
| rc42 | Back and sides from Qwen: 90°, 135°, 180°, each conditioned on the photo **and** the previous accepted view | A back of the person exists; views agree with each other |
| rc43 | Best-view colour baking after training | Texture as sharp as the best view of each surface, no averaging noise |
| rc44 | Optional TRELLIS.2 "complete subject" (Windows fork), aligned to MoGe-2, front from the photo | Full, consistent 3D subject with a real back |
| later | Qwen-Image-2.1 + its angles LoRA (after resolver check); Difix3D+ repair pass | Higher-resolution, more consistent views |
| after the image work | **Video → 4D Gaussian Splatting** with a player for 4D splats; the current VR180 video path is removed and rebuilt on top of 4DGS (owner, October 2026: today a video only yields one frame) | Moving scenes in 3D/VR |

## Sources

- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects (setup: 32 GB, Linux, gated)
- TRELLIS.2: https://github.com/microsoft/TRELLIS.2; Windows fork https://github.com/IgorAherne/TRELLIS.2-stableprojectorz
- TRELLIS v1: https://github.com/microsoft/TRELLIS
- Qwen-Image / Edit-2511: https://github.com/QwenLM/Qwen-Image
- Qwen-Image-2.1 announcements (Sept 2026): aiweekly.co, eesel.ai review, digg.com summaries
- Lyra: https://github.com/nv-tlabs/lyra (arXiv 2509.19296)
- HunyuanWorld-Mirror: https://huggingface.co/tencent/HunyuanWorld-Mirror (arXiv 2510.10726)
- LHM: ICCV 2025, https://openaccess.thecvf.com/content/ICCV2025/html/Qiu_LHM_Large_Animatable_Human_Reconstruction_Model_for_Single_Image_to_ICCV_2025_paper.html
