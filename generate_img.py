
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import torch 
import json
import numpy as np
from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler
from utils_model import load_model_from_config 
import torch.nn as nn
from PIL import Image
from ldm.models.autoencoder import AutoencoderKL
from utils_model import ConvLoRA


device = torch.device("cuda")

ldm_ckpt = "stable-diffusion-2/768-v-ema.ckpt"


ddconfig = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}
lossconfig = {"target": "torch.nn.Identity"}

vae = AutoencoderKL(embed_dim=4, ddconfig=ddconfig, lossconfig=lossconfig)

checkpoint = torch.load(ldm_ckpt, map_location="cpu")
vae_state_dict = {k.replace("first_stage_model.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("first_stage_model.")}
vae.load_state_dict(vae_state_dict, strict=True) # 加载提取的 VAE 权重
vae.to(device)

student = vae.decoder
getattr(getattr(student.up, '3').block, '1').conv1 = ConvLoRA(getattr(getattr(student.up, '3').block, '1').conv1)
getattr(getattr(student.up, '3').block, '1').conv2 = ConvLoRA(getattr(getattr(student.up, '3').block, '1').conv2)
getattr(getattr(student.up, '3').block, '2').conv1 = ConvLoRA(getattr(getattr(student.up, '3').block, '2').conv1)
getattr(getattr(student.up, '3').block, '2').conv2 = ConvLoRA(getattr(getattr(student.up, '3').block, '2').conv2)


class CustomVAE(nn.Module):
    def __init__(self):
        super(CustomVAE, self).__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = student

    def forward(self, src_mean, wm):
        z = self.post_quant_conv(src_mean)
        imgs_rec = self.decoder(z, wm)
        return imgs_rec

my_decoder = CustomVAE()


prompts = [
    "a high-tech laboratory with neon lights",
    "a serene landscape of mountains reflected in a crystal clear lake",
    "a futuristic city with flying cars and tall skyscrapers",
    "a cozy wooden cabin in a snowy forest at twilight"
]

with torch.no_grad():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    generator = torch.Generator(device="cuda").manual_seed(42)

    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base', 
        safety_checker=None, 
        requires_safety_checker=False
    ).to(device)

    print(f"Starting generation for {len(prompts)} prompts...")
    
    total_time_ms = 0
    for i, prompt in enumerate(prompts):
        start_time = time.time()

        # Dual Fingerprinting: 32-bit + 32-bit
        watermark1 = torch.zeros((1, 32), dtype=torch.float).random_(0, 2).to(device)
        watermark2 = torch.zeros((1, 32), dtype=torch.float).random_(0, 2).to(device)
        
        output = pipe(
            prompt, 
            num_images_per_prompt=1, 
            generator=generator, 
            height=512, 
            width=512, 
            if_wm=True, 
            wm=[watermark1, watermark2], 
            my_decoder=my_decoder
        )
        
        img = output.images[0]
        img.save(f"output_{i}.png")
        
        elapsed_time_ms = (time.time() - start_time) * 1000
        total_time_ms += elapsed_time_ms
        print(f"Prompt {i} done. Time: {elapsed_time_ms:.2f}ms")

    average_time = total_time_ms / len(prompts)
    print(f"\nAverage time per image: {average_time:.2f} ms")
