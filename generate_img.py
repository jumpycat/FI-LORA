
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


filename = 'captions_val2017.json'
with open(filename, 'r') as file:
    data = json.load(file)
annotations = data['annotations']

first_captions = {}
for annotation in annotations:
    image_id = annotation['image_id']
    if image_id not in first_captions:
        first_captions[image_id] = annotation['caption']

captions_list = list(first_captions.values())


from collections import Counter
counter = Counter(captions_list)
suffix_tracker = {}
updated_captions = []

for name in captions_list:
    if counter[name] > 1:
        if name not in suffix_tracker:
            suffix_tracker[name] = 1
        suffix_tracker[name] += 1
        new_name = f"{name}_{suffix_tracker[name]}"
        updated_captions.append(new_name)
    else:
        updated_captions.append(name)

captions_list = updated_captions
print(len(set(captions_list)))


with torch.no_grad():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    generator = torch.Generator(device="cuda").manual_seed(42)

    # Load the Stable Diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base', 
        safety_checker=None, 
        requires_safety_checker=False
    ).to(device)
    import time
    total_time_ms = 0  # To accumulate total elapsed time

    # Run the loop 10 times
    for group in captions_list[:10]:
        start_time = time.time()  # Start time in seconds

    #    Assuming `grouped_captions` is predefined
        watermark1 = torch.zeros((1, 32), dtype=torch.float).random_(0, 2).to(device)
        watermark2 = torch.zeros((1, 32), dtype=torch.float).random_(0, 2).to(device)
        
        # Generate images
        imgs = pipe(
            group, 
            num_images_per_prompt=1, 
            generator=generator, 
            height=512, 
            width=512, 
            if_wm=True, 
            wm=[watermark1, watermark2], 
            my_decoder=my_decoder
        ).images
        

    end_time = time.time()  # End time in seconds
    elapsed_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Add to the total time
    total_time_ms += elapsed_time_ms


# Calculate the average time
average_time_ms = total_time_ms / 10
print(f"\nAverage time per iteration: {average_time_ms:.2f} ms")