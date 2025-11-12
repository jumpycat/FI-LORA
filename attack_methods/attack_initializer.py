from .Gaussian_blur import Gaussian_blur
from .Gaussian_noise import Gaussian_noise
from .Jpeg_compression import Jpeg
from .DiffJPEG_master.DiffJPEG import DiffJPEG
from .Combination import Combination_attack
from .Crop import Crop
from kornia import augmentation as K
from kornia.augmentation import RandomGaussianBlur, RandomGaussianNoise,RandomBrightness, RandomCrop, RandomRotation
import random
import torch
from torchvision import transforms

import sys
sys.path.append("..")


unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x + 1.0) / 2.0
normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5


def clamp_transform(x, min_val=0, max_val=1):
    return torch.clamp(x, min_val, max_val)

def apply_with_prob(p, transform):
    def apply_transform(x):
        if random.random() > p:
            return x
        return transform(x)
    return apply_transform

def random_crop(p):
    def apply_transform(x):
        if random.random() > p:
            return x
        return CropAndPad(x)
    return apply_transform

def CropAndPad(x):
    # size = random.choice([0.6, 0.7, 0.8, 0.9])
    size = random.choice([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    c = transforms.RandomCrop((int(512*size), int(512*size)))
    x, (i, j, h, w) = c(x,aug=True)

    p = transforms.Pad((j, i, 512-j-w, 512-i-h))  #left, top, right and bottom borders respectively.
    x = p(x)
    return x


class IdentityTransform:
    def __call__(self, img):
        return img
    
def attack_initializer(device):
    attack_prob = 0.8
    resolution = 512

    # ejgbnha, RandomGaussianBlur, apply_diffjpeg, RandomErasing
    # define custom lambda function
    def apply_diffjpeg(x):
        quality = random.choice([20, 30, 40, 50, 60, 70, 80, 90])  # randomly select quality parameter
        return DiffJPEG(height=resolution, width=resolution, differentiable=True, quality=quality).to(device)(x)

    # gbcnhja gbcnhja
    attack = transforms.RandomChoice([
        # aug_list,
        K.RandomGaussianBlur(kernel_size=random.choice([(3,3), (5,5), (7,7),(9,9), (11,11)]), sigma=(1.5, 3.5), p = attack_prob, keepdim=True),
        K.RandomBrightness((0.2, 1.8), p = attack_prob, keepdim=True), #0.7
        # K.RandomContrast((0.2, 1.8), p = attack_prob, keepdim=True), #0.7
        K.RandomGaussianNoise(mean=0., std = 0.2, p = attack_prob, keepdim=True),
        # K.RandomErasing(p = attack_prob if 'e' in args else 0, keepdim=True),
        K.RandomHorizontalFlip(p = attack_prob, keepdim=True),
        transforms.Lambda(apply_with_prob(attack_prob, apply_diffjpeg)),  # add conditional transformation
        transforms.Lambda(random_crop(attack_prob)),  # add conditional transformation
        # K.RandomRotation(degrees=90, p = attack_prob, keepdim=True),
        # K.RandomResizedCrop(size=(512,512), p = attack_prob if 'r' in args else 0, keepdim=True),
        IdentityTransform(),
    ], p=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.28])

    return attack
