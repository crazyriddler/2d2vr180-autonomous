"""2D2VR180 adaptation of FlashWorld's app.GenerationSystem (imlixinyang/FlashWorld, Apache-2.0).

Differences from upstream: model files come from local directories (no hub access); the transformer is built
on the meta device and its weights come straight from FlashWorld's checkpoint, memory-mapped, quantised to
FP8 layer by layer like upstream does (the 21 GB checkpoint never sits in RAM as a whole, and the Wan2.2
transformer weights are not needed); the text encoder runs in a separate step and the prompt embedding is
passed to generate(). The denoising / 3D decoding code below is upstream's, unchanged."""

from contextlib import contextmanager

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from .fw_utils import create_raymaps, normalize_cameras, sample_from_dense_cameras
from .models import AutoencoderKLWan, WANDecoderPixelAligned3DGSReconstructionModel, WanTransformer3DModel

mode = "Local"


@contextmanager
def onload_model(model, device, onload=False):
    """Move a model to the GPU for the duration of the block (upstream's helper)."""
    if onload and str(device) != "cpu":
        model.to(device)
        try:
            yield model
        finally:
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        yield model


class MyFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        return torch.argmin((timestep - schedule_timesteps.to(timestep.device)).abs(), dim=0).item()


def _fix_causal_padding(module):
    """Upstream: drop the causal time padding of the VAE's 3-D convolutions (single-frame use)."""
    from .models.autoencoder_kl_wan import WanCausalConv3d

    with torch.no_grad():
        for _, m in module.named_modules():
            if isinstance(m, WanCausalConv3d):
                time_pad = m._padding[4]
                m.padding = (0, m._padding[2], m._padding[0])
                m._padding = (0, 0, 0, 0, 0, 0)
                m.weight = torch.nn.Parameter(m.weight[:, :, time_pad:].clone())


class GenerationSystem(nn.Module):
    def __init__(self, vae_dir, transformer_config, scheduler_dir, ckpt_path=None, device="cuda",
                 offload_vae=False, offload_transformer_during_vae=False, fp8=True, log=print,
                 state_dict=None):
        super().__init__()
        self.device = torch.device(device)
        self.offload_t5 = False
        self.offload_vae = offload_vae
        self.offload_transformer_during_vae = offload_transformer_during_vae

        self.latent_dim = 48
        self.temporal_downsample_factor = 4
        self.spatial_downsample_factor = 16
        self.feat_dim = 1024
        self.latent_patch_size = 2
        self.denoising_steps = [0, 250, 500, 750]

        log("FlashWorld: VAE")
        self.vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=torch.float).eval()
        _fix_causal_padding(self.vae)
        self.vae.requires_grad_(False)
        self.register_buffer("latents_mean", torch.tensor(self.vae.config.latents_mean).float().view(
            1, self.vae.config.z_dim, 1, 1, 1).to(self.device))
        self.register_buffer("latents_std", torch.tensor(self.vae.config.latents_std).float().view(
            1, self.vae.config.z_dim, 1, 1, 1).to(self.device))

        # the transformer's shape (incl. upstream's extra input/output channels) without allocating it
        with torch.device("meta"):
            self.transformer = WanTransformer3DModel.from_config(transformer_config).requires_grad_(False)
            t = self.transformer
            t.patch_embedding.weight = nn.Parameter(F.pad(t.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, 6 + self.latent_dim)))
            weight = t.proj_out.weight.reshape(self.latent_patch_size ** 2, self.latent_dim, t.proj_out.weight.shape[1])
            bias = t.proj_out.bias.reshape(self.latent_patch_size ** 2, self.latent_dim)
            extra_w = torch.empty(self.latent_patch_size ** 2, self.feat_dim, t.proj_out.weight.shape[1])
            extra_b = torch.empty(self.latent_patch_size ** 2, self.feat_dim)
            t.proj_out.weight = nn.Parameter(torch.cat([weight, extra_w], dim=1).flatten(0, 1))
            t.proj_out.bias = nn.Parameter(torch.cat([bias, extra_b], dim=1).flatten(0, 1))
            self.use_feedback = True
            t.patch_embedding.weight = nn.Parameter(F.pad(t.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, self.feat_dim + self.latent_dim)))

        log("FlashWorld: 3D Gaussian decoder")
        self.recon_decoder = WANDecoderPixelAligned3DGSReconstructionModel(
            self.vae, self.feat_dim, use_render_checkpointing=True, use_network_checkpointing=False).requires_grad_(False)
        self.scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_dir, shift=3)
        self.register_buffer("timesteps", self.scheduler.timesteps.clone().to(self.device))
        self.transformer.disable_gradient_checkpointing()
        self.transformer.gradient_checkpointing = False

        if state_dict is None:
            log(f"FlashWorld: weights {ckpt_path} (memory-mapped)")
            state_dict = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        self.transformer.load_state_dict(state_dict["transformer"], assign=True)
        # buffers computed at construction (RoPE tables) are not in the checkpoint: build them for real
        from .models.transformer_wan import WanRotaryPosEmbed

        cfg = self.transformer.config
        self.transformer.rope = WanRotaryPosEmbed(cfg.attention_head_dim, cfg.patch_size, cfg.rope_max_seq_len)
        meta = [n for n, b in list(self.transformer.named_parameters()) + list(self.transformer.named_buffers())
                if b.is_meta]
        if meta:
            raise RuntimeError(f"FlashWorld checkpoint does not cover {meta[:5]}")
        self.recon_decoder.load_state_dict(state_dict["recon_decoder"])
        del state_dict

        if fp8:
            log("FlashWorld: quantising the transformer to FP8")
            from .quant import FluxFp8GeMMProcessor

            FluxFp8GeMMProcessor(self.transformer)

        del self.vae.post_quant_conv, self.vae.decoder
        self.vae.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
        self.recon_decoder.to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
        self.transformer.to(self.device if not self.offload_transformer_during_vae else "cpu")

    def latent_scale_fn(self, x):
        return (x - self.latents_mean) / self.latents_std

    def latent_unscale_fn(self, x):
        return x * self.latents_std + self.latents_mean

    def add_feedback_for_transformer(self):
        self.use_feedback = True
        self.transformer.patch_embedding.weight = nn.Parameter(F.pad(self.transformer.patch_embedding.weight, (0, 0, 0, 0, 0, 0, 0, self.feat_dim + self.latent_dim)))
    
    def encode_text(self, texts):
        max_sequence_length = 512

        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if getattr(self, "offload_t5", False):
            text_input_ids = text_inputs.input_ids.to("cpu")
            mask = text_inputs.attention_mask.to("cpu")
        else:
            text_input_ids = text_inputs.input_ids.to(self.device)
            mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        if getattr(self, "offload_t5", False):
            with torch.no_grad():
                text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state.to(self.device)
        else:
            text_embeds = self.text_encoder(text_input_ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds.float()

    def forward_generator(self, noisy_latents, raymaps, condition_latents, t, text_embeds, cameras, render_cameras, image_height, image_width, need_3d_mode=True):

        with onload_model(self.transformer, self.device, onload=self.offload_transformer_during_vae):
            out = self.transformer(
                hidden_states=torch.cat([noisy_latents, raymaps, condition_latents], dim=1),
                timestep=t,
                encoder_hidden_states=text_embeds,
                return_dict=False,
            )[0]

        v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
               
        sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
        latents_pred_2d = noisy_latents - sigma * v_pred

        if need_3d_mode:
            scene_params = self.recon_decoder(
                                einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                einops.rearrange(self.latent_unscale_fn(latents_pred_2d.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                cameras.to(self.device if not self.offload_vae else "cpu").float()
                            ).flatten(1, -2).to(self.device).float()

            images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

            latents_pred_3d = einops.rearrange(self.latent_scale_fn(self.vae.encode(
                            einops.rearrange(images_pred, 'B T C H W -> (B T) C H W', T=images_pred.shape[1]).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                        ).latent_dist.sample().to(self.device)).squeeze(2), '(B T) C H W -> B C T H W', T=images_pred.shape[1]).to(noisy_latents.dtype)

        return {
            '2d': latents_pred_2d,
            '3d': latents_pred_3d if need_3d_mode else None,
            'rgb_3d': images_pred if need_3d_mode else None,
            'scene': scene_params if need_3d_mode else None,
            'feat': feats
        }

    @torch.no_grad()
    def generate(self, cameras, n_frame, image=None, text="", image_index=0, image_height=480, image_width=704, video_path=None, video_fps=15):  

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=self.device.type):
            batch_size = 1
            
            cameras = cameras.to(self.device).unsqueeze(0)

            if cameras.shape[1] != n_frame:
                cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, n_frame, device=self.device)).unsqueeze(0)

            if video_path is not None:
                render_cameras = sample_from_dense_cameras(cameras.squeeze(0), torch.linspace(0, 1, (n_frame - 1) * video_fps + 1, device=self.device)).unsqueeze(0)
            else:
                render_cameras = None
            
            cameras, ref_w2c, T_norm = normalize_cameras(cameras, return_meta=True, n_frame=None)

            render_cameras = normalize_cameras(render_cameras, ref_w2c=ref_w2c, T_norm=T_norm, n_frame=None) if render_cameras is not None else None

            # 2D2VR180: the prompt "[Static] <text>" is encoded beforehand (text_embeds.py) and passed in
            text_embeds = text.to(self.device).float() if torch.is_tensor(text) else self.encode_text(["[Static] " + text])
            # neg_text_embeds = self.encode_text([""]).repeat(batch_size, 1, 1)

            masks = torch.zeros(batch_size, n_frame, device=self.device)

            condition_latents = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            if image is not None:
                image = image.to(self.device)

                latent = self.latent_scale_fn(self.vae.encode(
                        image.unsqueeze(0).unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16)
                    ).latent_dist.sample().to(self.device)).squeeze(2)

                masks[:, image_index] = 1
                condition_latents[:, :, image_index] = latent

            raymaps = create_raymaps(cameras, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor)
            raymaps = einops.rearrange(raymaps, 'B T H W C -> B C T H W', T=n_frame)
            
            noise = torch.randn(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            noisy_latents = noise 

            if self.use_feedback:
                prev_latents_pred = torch.zeros(batch_size, self.latent_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

                prev_feats = torch.zeros(batch_size, self.feat_dim, n_frame, image_height // self.spatial_downsample_factor, image_width // self.spatial_downsample_factor, device=self.device)

            for i in range(len(self.denoising_steps)):
                t_ids = torch.full((noisy_latents.shape[0],), self.denoising_steps[i], device=self.device)

                t = self.timesteps[t_ids]

                if self.use_feedback:
                    _condition_latents = torch.cat([condition_latents, prev_feats, prev_latents_pred], dim=1)
                else:
                    _condition_latents = condition_latents

                if i < len(self.denoising_steps) - 1:
                    out = self.forward_generator(noisy_latents, raymaps, _condition_latents, t, text_embeds, cameras, cameras, image_height, image_width, need_3d_mode=True)

                    latents_pred = out["3d"]

                    if self.use_feedback:
                        prev_latents_pred = latents_pred
                        prev_feats = out['feat']
                   
                    noisy_latents = self.scheduler.scale_noise(latents_pred, self.timesteps[torch.full((noisy_latents.shape[0],), self.denoising_steps[i + 1], device=self.device)], torch.randn_like(noise))
                    
                else:
                    with onload_model(self.transformer, self.device, onload=self.offload_transformer_during_vae):
                        out = self.transformer(
                            hidden_states=torch.cat([noisy_latents, raymaps, _condition_latents], dim=1),
                            timestep=t,
                            encoder_hidden_states=text_embeds,
                            return_dict=False,
                        )[0]

                    v_pred, feats = out.split([self.latent_dim, self.feat_dim], dim=1)
                        
                    sigma = torch.stack([self.scheduler.sigmas[self.scheduler.index_for_timestep(_t)] for _t in t.unbind(0)], dim=0).to(self.device)
                    latents_pred = noisy_latents - sigma * v_pred

                    scene_params = self.recon_decoder(
                                        einops.rearrange(feats, 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        einops.rearrange(self.latent_unscale_fn(latents_pred.detach()), 'B C T H W -> (B T) C H W').unsqueeze(2).to(self.device if not self.offload_vae else "cpu").to(torch.bfloat16), 
                                        cameras.to(self.device if not self.offload_vae else "cpu").float()
                                    ).flatten(1, -2).to(self.device).float()

            if video_path is not None:
                interpolated_images_pred, _ = self.recon_decoder.render(scene_params.unbind(0), render_cameras, image_height, image_width, bg_mode="white")

                interpolated_images_pred = einops.rearrange(interpolated_images_pred[0].clamp(-1, 1).add(1).div(2), 'T C H W -> T H W C')

                interpolated_images_pred = [torch.cat([img], dim=1).detach().cpu().mul(255).numpy().astype(np.uint8) for i, img in enumerate(interpolated_images_pred.unbind(0))]

                import imageio

                imageio.mimwrite(video_path, interpolated_images_pred, fps=video_fps, quality=8, macro_block_size=1) 

        scene_params = scene_params[0]

        return scene_params, ref_w2c, T_norm
