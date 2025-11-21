import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
import torch.nn.functional as F
from basic_module import Transformer
import utils
import math
from pretrained_models.dpv2.depth_anything_v2.dpt import DepthAnythingV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleGallery(StableDiffusionPipeline):
    def freeze(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.extractor.requires_grad_(False)

    def load_image(self,image_path, size=None, mode="RGB"):
        img = Image.open(image_path).convert(mode)
        if size is None:
            width, height = img.size
            new_width = (width // 64) * 64
            new_height = (height // 64) * 64
            size = (new_width, new_height)
        img = img.resize(size, Image.BICUBIC)
        return ToTensor()(img).unsqueeze(0)


    @torch.no_grad()
    def image2latent(self, image):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        image = image.to(device=device, dtype=dtype) * 2.0 - 1.0
        latent = self.vae.encode(image)["latent_dist"].mean
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def latent2image(self, latent):
        dtype = next(self.vae.parameters()).dtype
        device = self._execution_device
        latent = latent.to(device=device, dtype=dtype)
        latent = latent / self.vae.config.scaling_factor
        image = self.vae.decode(latent)[0]
        return (image * 0.5 + 0.5).clamp(0, 1)

    def init(self, enable_gradient_checkpoint):
        self.freeze()
        self.enable_vae_slicing()
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)
        self.extractor.to(self.accelerator.device, dtype=weight_dtype)
        # self.extractor = self.accelerator.prepare(self.extractor)
        if enable_gradient_checkpoint:
            self.extractor.enable_gradient_checkpointing()
          
    def weighted_features(self, latent, total_elements=15):
        def adaptive_decay(x, total_elements):
            return 1 / (1 + torch.exp(5 * (x / total_elements - 0.7)))
        indices = torch.arange(total_elements, device=latent[0].device)
        weights = adaptive_decay(indices, total_elements)
        weights = weights / weights.sum()
        weighted_feature = torch.stack(latent).mul(weights[:, None, None, None, None]).sum(dim=0)

        return weighted_feature

    def forward_process(self, latent, text_embeds, steps=15):
        pred_images = []
        pred_latents = []
        Unetcache = utils.UnetDataCache()
        unet_feature = []
        print("latent:", latent.shape)
        self.scheduler.set_timesteps(steps)
        timesteps = reversed(self.scheduler.timesteps)
        cur_latent = latent.clone()
        with torch.no_grad():
            for i in tqdm(range(0, steps), desc="DDIM Inversion"):
                t = timesteps[i]
                hooks = utils.register_unet_feature_extraction(
                    self.unet, Unetcache
                )
                noise_pred = self.unet(
                    cur_latent,
                    t.to(device),
                    text_embeds
                ).sample

                _, unet_feature_2 = Unetcache.get_features()
                unet_feature.append(unet_feature_2)
                current_t = max(0, t.item() - (1000 // steps))
                next_t = t
                alpha_t = self.scheduler.alphas_cumprod[torch.tensor(current_t, dtype=torch.long)]
                alpha_t_next = self.scheduler.alphas_cumprod[torch.as_tensor(next_t, dtype=torch.long).clone().detach()]
                cur_latent = (cur_latent - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred
                pred_latents.append(cur_latent)
                if i % 3 == 0:
                    pred_images.append(self.latent2image(cur_latent))

                for hook in hooks:
                    hook.remove()

        unet_up_block_feature2 = self.weighted_features(unet_feature, steps)
        return pred_images, pred_latents, unet_up_block_feature2

  
   def style_transfer(
        self,
        content_dict,
        style_dict,
        controller,
        cluster_matches,
        mixed_precision="fp16",
        num_optimize_steps=50,
        enable_gradient_checkpoint=False,
        lr=0.05,
        iters=1,
        c_ratio=0.26,
    ):
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision, gradient_accumulation_steps=1
        )
        self.init(enable_gradient_checkpoint)

        content_latent = content_dict["feature"].to(device).half()
        content_mask   = content_dict["mask"].to(device).half()
        latents = content_dict["C15"].clone().to(device)
        latents = self.remove_style(latents)

        null_text_embeds = self.encode_prompt("", self.device, 1, False)[0]
        text_embeds = null_text_embeds.repeat(content_latent.shape[0], 1, 1).to(device)

        self.attncache = utils.AttnDataCache()
        self.controller = controller
        utils.register_attn_control(
            self.extractor,
            self.controller,
            self.attncache
        )
        self.extractor = self.accelerator.prepare(self.extractor)
        gpu_style_dict = []
        for sd in style_dict:
            gpu_style_dict.append({
                "feature": sd["feature"].to(device),
                "mask":    sd["mask"].to(device)
            })

        self.scheduler.set_timesteps(num_optimize_steps)
        timesteps = self.scheduler.timesteps

        latents = latents.detach().float()
        optimizer = torch.optim.Adam([latents.requires_grad_()], lr=lr)
        optimizer = self.accelerator.prepare(optimizer)

        pbar = tqdm(timesteps, desc="Optimize")

        for i, t in enumerate(pbar):
            with torch.no_grad():
                qc_list, kc_list, vc_list, c_out_list = self.extract_feature(
                    content_latent,
                    t,
                    text_embeds
                )
                style_attn_dict = []
                for sd in gpu_style_dict:
                    style_feat = sd["feature"]
                    style_mask = sd["mask"]
                    q, k, v, s_out_list = self.extract_feature(
                        style_feat,
                        t,
                        text_embeds
                    )
                    style_attn_dict.append({
                        "q": q,
                        "k": k,
                        "v": v,
                        "mask": style_mask,
                        "feature": style_feat
                    })

            for j in range(iters):
                optimizer.zero_grad(set_to_none=True)
                style_loss = 0.0
                q_list, k_list, v_list, self_out_list = self.extract_feature(
                    latents,
                    t,
                    text_embeds
                )

                content_loss = utils.content_loss(q_list, qc_list)
                style_loss = self.get_style_loss(
                       cluster_matches,
                       content_mask,
                       q_list,
                       qc_list,
                       self_out_list,
                       style_attn_dict,
                )
                # for k in range(len(style_attn_dict)):
                #     ks_list = style_attn_dict[k]['k']
                #     vs_list = style_attn_dict[k]['v']
                #     style_loss += utils.style_loss(q_list, ks_list, vs_list, self_out_list)

                loss = style_loss + c_ratio * content_loss

                self.accelerator.backward(loss)
                optimizer.step()

                pbar.set_postfix(loss=float(loss.detach().item()), time=float(t.item()))

        image = self.latent2image(latents)
        self.maybe_free_model_hooks()
        return image


    @torch.no_grad()
    def get_depth(
        self,
        image_path,
        output_dir: str = "./outputs/depth_visual",
        save_colormap=True
    ):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl' 

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(f'pretrained_models/dpv2/ckpts/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        model = model.to(self.device).eval()

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(image_path).convert("RGB"))[:, :, ::-1] 
       
        depth = model.infer_image(img)  

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(output_dir, base + "_depth.png")

            d_min, d_max = float(depth.min()), float(depth.max())
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
            d8 = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
            vis = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO) if save_colormap else d8
            cv2.imwrite(save_path, np.ascontiguousarray(vis))
            print(f"[Saved] depth image: {save_path}")

        return depth






