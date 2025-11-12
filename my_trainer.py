import argparse
import os
import sys
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn.utils import clip_grad_norm_
import time
import utils
import utils_img
import utils_model
from torch import optim
from torch import Tensor
import random
from datetime import timedelta
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from copy import copy
import sys
import os

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.datasets import ImageFolder
from ldm.models.autoencoder import AutoencoderKL
from torch.utils.data import Dataset
from PIL import Image
import math
from typing import Optional, List
from utils_model import ConvLoRA, WMEbeddingNet


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def format_time(seconds):
    delta = timedelta(seconds=seconds)
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{days}d {hours}h {minutes}m {seconds}s'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default='CoCo/coco_train/train_2017')

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str, default="v2-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_stu_config", type=str, default="v2-inference_stu.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="768-v-ema.ckpt", help="Path to the checkpoint file for the LDM model") 

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=512, help="Resize images to this size")

    aa("--lambda_i_lpips", type=int, default=1, help="Weight of the image loss in the total loss")
    aa("--lambda_i_mse", type=int, default=1, help="Weight of the image loss in the total loss")
    aa("--lambda_w1", type=int, default=5, help="Weight of the watermark loss in the total loss")
    aa("--lambda_w2", type=int, default=1, help="Weight of the watermark loss in the total loss")

    aa("--lr", type=float, default=3e-4)
    aa("--steps", type=int, default=350000, help="Number of steps to train the model for")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_model_freq", type=int, default=10000, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--seed", type=int, default=42)
    aa("--bits", type=int, default=32)
    aa("--warm_steps", type=int, default=10000)
    aa("--temp", type=float, default=1)
    aa("--resume", type=str, default='checkpoint_300000.pth')
    aa("--cum_times", type=int, default=1)
    aa("--attack",type=str, default='')
    return parser 

def main(params):
    # Create directories if not exist.
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%y%m%d%H%M%S")

    exp_path = os.path.join('runs',dt_string)
    model_save_dir = os.path.join('runs',dt_string,'models')
    sample_dir = os.path.join('runs',dt_string,'samples')
    os.makedirs(exp_path,exist_ok=True)
    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(sample_dir,exist_ok=True)

    cmd = ' '.join(sys.argv)
    with open(os.path.join(exp_path,'args.txt'), 'a') as file:
        for arg in vars(params):
            file.write(f"{arg}: {getattr(params, arg)}\n")
        file.write(f"{cmd}\n")

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    # Loads the data
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.RandomCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan])

    class CustomImageNpyDataset(Dataset):
        def __init__(self, img_dir, npy_dir, transform=None):
            """
            Args:
                img_dir (str): 图片文件夹路径
                npy_dir (str): npy文件夹路径
                transform (callable, optional): 用于处理图像的变换
            """
            self.img_dir = img_dir
            self.npy_dir = npy_dir
            self.transform = transform
            self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        def __len__(self):
            return len(self.img_names)

        def __getitem__(self, idx):
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            npy_name = img_name.replace('.jpg', '.npy')
            npy_path = os.path.join(self.npy_dir, npy_name)

            # 读取图像
            image = Image.open(img_path).convert('RGB')

            # 读取npy文件
            npy_data = np.load(npy_path)

            # 应用图像变换
            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(npy_data)

    # from torchvision.datasets import ImageFolder

    train_dataset = CustomImageNpyDataset('dataset_z/imgs', 'dataset_z/z', transform=vqgan_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=16, drop_last=True)

    # 定义 ddconfig 配置
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
    lossconfig = {
        "target": "torch.nn.Identity"
    }

    # 初始化 AutoencoderKL 模型
    vae = AutoencoderKL(
        embed_dim=4,
        ddconfig=ddconfig,
        lossconfig=lossconfig
    )

    checkpoint = torch.load(params.ldm_ckpt, map_location="cpu")
    vae_state_dict = {k.replace("first_stage_model.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("first_stage_model.")} 
    vae.load_state_dict(vae_state_dict, strict=True) 
    vae.to(device)


    vae_decoder = vae.decoder
    # getattr(getattr(vae_decoder.up, '3').block, '0').conv1 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '0').conv1)
    # getattr(getattr(vae_decoder.up, '3').block, '0').conv2 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '0').conv2)
    getattr(getattr(vae_decoder.up, '3').block, '1').conv1 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '1').conv1)
    getattr(getattr(vae_decoder.up, '3').block, '1').conv2 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '1').conv2)
    getattr(getattr(vae_decoder.up, '3').block, '2').conv1 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '2').conv1)
    getattr(getattr(vae_decoder.up, '3').block, '2').conv2 = ConvLoRA(getattr(getattr(vae_decoder.up, '3').block, '2').conv2)
    # vae_decoder.conv_out = ConvLoRA(vae_decoder.conv_out)

    # Loads hidden decoder
    from efficientnet_pytorch import EfficientNet
    wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
    feature = wm_decoder._fc.in_features
    wm_decoder._fc = nn.Linear(in_features=feature, out_features=params.bits * 2,bias=True)
    wm_decoder.to(device)



    # Create losses
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    loss_w = lambda decoded, keys, temp: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    my_mse = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)

    optimizer = optim.Adam(list(vae_decoder.parameters()) + list(wm_decoder.parameters()), lr=params.lr)

    train_iter = iter(train_loader)
    Minus112ZeroOne = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5])


    try:
        checkpoint = torch.load(params.resume)
        vae_decoder.load_state_dict(checkpoint["ldm_decoder"])
        wm_decoder.load_state_dict(checkpoint["wm_decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        from pathlib import Path
        import re
        file_name = Path(params.resume).name
        start_step = int(re.search(r'\d+', file_name).group()) + 1
        print('all load')
    except:
        print('fail load')
        start_step = 1

    from tqdm import tqdm
    start_time = time.time()
    with tqdm(total = params.steps, initial=start_step) as pbar:
        for total_steps in range(start_step, params.steps + 1):
                
            watermark1 = torch.zeros((params.batch_size, params.bits), dtype=torch.float).random_(0, 2).to(device)
            watermark2 = torch.zeros((params.batch_size, params.bits), dtype=torch.float).random_(0, 2).to(device)

            try:
                imgs, imgs_z = next(train_iter)
                imgs = imgs.to(device)
                imgs_z = imgs_z.to(device)

            except:
                train_iter = iter(train_loader)
                imgs, imgs_z = next(train_iter)
                imgs = imgs.to(device)
                imgs_z = imgs_z.to(device)

            # encode images
            # with torch.no_grad():
            #     imgs_z = vae.encode(imgs) # b c h w -> b z h/f w/f
            #     imgs_z = imgs_z.mode()
            #     imgs_z = vae.post_quant_conv(imgs_z).detach()

            imgs_marked = vae_decoder(imgs_z, [watermark1,watermark2])
            imgs_marked_aug = Minus112ZeroOne(imgs_marked)
            decoded = wm_decoder(imgs_marked_aug) # b c h w -> b 
            # decodedB = wm_decoderB(imgs_marked_aug) # b c h w -> b 


            if total_steps < params.warm_steps:
                lambda_i_lpips = params.lambda_i_lpips * 1
                lambda_i_mse = params.lambda_i_mse * 1
                lambda_w1 = params.lambda_w1 * 1
                lambda_w2 = params.lambda_w2 * 1
                temp = params.temp * 10
                cum_times = params.cum_times * 1

            elif params.warm_steps <= total_steps <= 2 * params.warm_steps:
                lambda_i_lpips = params.lambda_i_lpips * 1
                lambda_i_mse = params.lambda_i_mse * 1
                lambda_w1 = params.lambda_w1 * 1
                lambda_w2 = params.lambda_w2 * 1
                temp = params.temp * 1
                cum_times = params.cum_times * 1

            elif total_steps > 2 * params.warm_steps:
                lambda_i_lpips = params.lambda_i_lpips * 10
                lambda_i_mse = params.lambda_i_mse * 10
                lambda_w1 = params.lambda_w1 * 0.1
                lambda_w2 = params.lambda_w2 * 0.1
                temp = params.temp * 1
                cum_times = params.cum_times * 1
            else:
                lambda_i_lpips = params.lambda_i_lpips * 10
                lambda_i_mse = params.lambda_i_mse * 1
                lambda_w1 = params.lambda_w1 * 1
                lambda_w2 = params.lambda_w2 * 1
                temp = params.temp * 1
                cum_times = params.cum_times * 1
            
            # compute loss
            lossw = lambda_w1 * loss_w(decoded[:,:32], watermark1, temp) + lambda_w2 * loss_w(decoded[:,32:], watermark2, temp)
            lossi_lpips = loss_fn_vgg.forward(imgs, imgs_marked).mean()
            lossi_mse = my_mse(imgs, imgs_marked)
            loss = lossw + lambda_i_lpips * lossi_lpips + lambda_i_mse * lossi_mse

            # optim step
            loss.backward()
            if total_steps % cum_times == 0:
                optimizer.step()
                optimizer.zero_grad()

            if total_steps % params.log_freq == 0:
                last_time = time.time() - start_time

                wm_predictedA = (decoded[:,:32] > 0.0).float()
                bitwise_acc1 = 100 * (1.0 - torch.mean(torch.abs(watermark1 - wm_predictedA)))

                wm_predictedB = (decoded[:,32:] > 0.0).float()
                bitwise_acc2 = 100 * (1.0 - torch.mean(torch.abs(watermark2 - wm_predictedB)))

                log = f"{dt_string} {total_steps:06} {lambda_i_lpips} LPIPS: {lossi_lpips.item():.5f} {lambda_i_mse} MSE: {lossi_mse.item():.5f} W1/W2 {lambda_w1} {lambda_w2} WM: {lossw.item():.5f} Acc1: {bitwise_acc1.item():.2f} Acc2: {bitwise_acc2.item():.2f} Time:[{format_time(last_time)}]"
                # print(log)
                with open(os.path.join(exp_path,'logs.txt'), 'a', encoding='utf-8') as f:
                    f.write(log)
                    f.write('\n')
                tqdm.write(log)

            if total_steps % params.save_model_freq == 0:
                save_dict = {
                    'ldm_decoder': vae_decoder.state_dict(),
                    'wm_decoder': wm_decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'params': params,
                }

                # Save checkpoint
                torch.save(save_dict, os.path.join(model_save_dir, f"checkpoint_{total_steps:06d}.pth"))
            pbar.update(1)


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)