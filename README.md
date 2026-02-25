# Scalable Dual Fingerprinting for Hierarchical Attribution of Text-to-Image Models

This repository contains the official code for the ICCV 2025 paper:
**"Scalable Dual Fingerprinting for Hierarchical Attribution of Text-to-Image Models"** *(Jianwei Fei, Yunshu Dai, Peipeng Yu, Zhe Kong, Jiantao Zhou, Zhihua Xia, ICCV 2025)*

## 📦 Repository Overview
This project proposes a **dual fingerprinting** mechanism for hierarchical attribution of AI-generated content. By injecting multi-level watermarks into the Latent Diffusion Model (LDM) decoding process, it enables both model-level and user-level tracking without significantly degrading image quality.

## 📋 Usage

### 1. Environment & Preparation
- **SD2 Checkpoint**: Download `768-v-ema.ckpt` from [Stability AI](https://huggingface.co/stabilityai/stable-diffusion-2-base) and place it in the `stable-diffusion-2/` directory.
- **Dataset**: Prepare your training data in `dataset_z/` including:
  - `imgs/`: Source images.
  - `z/`: Pre-extracted latent representations (`.npy`).

### 2. Training
Train the hierarchical watermark decoder and the LoRA-enhanced LDM decoder:

```bash
python my_trainer.py
```

**Key Training Parameters:**
- `--lambda_w1` / `--lambda_w2`: Weights for the dual watermarks.
- `--lambda_i_lpips` / `--lambda_i_mse`: Weights for image reconstruction quality.
- `--warm_steps`: Step threshold for the dynamic loss scheduling (Default: 10000).
- `--resume`: Path to a previous checkpoint to continue training.

### 3. Image Generation

#### Step 1: Modify Diffusers Library
To support the custom `my_decoder` and hierarchical watermarks, you must modify the `StableDiffusionPipeline` in your environment (typically `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`):

1. **Update `__call__` arguments**:
   ```python
   def __call__(self, prompt=None, ..., if_wm: bool = False, wm = None, my_decoder = None):
   ```
2. **Inject custom decoding logic**:
   ```python
   # Replace standard VAE decode with:
   if if_wm and my_decoder is not None:
       image = my_decoder(latents / self.vae.config.scaling_factor, wm)
   else:
       image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
   ```

#### Step 2: Run Generation
Edit the `prompts` list in `generate_img.py` and run:

```bash
python generate_img.py
```

### 4. Watermark Extraction & Verification
Extract and verify the dual watermarks from the generated images:

```bash
python extract_wm.py
```

**Important Notes:**
- The `--seed` parameter during extraction must exactly match the seed used during generation to generate the correct ground truth watermarks for comparison.
- The script outputs the bitwise accuracy for both Model-level (W1) and User-level (W2) fingerprints.

## 🛠 Model Architecture
The framework utilizes **ConvLoRA** layers injected into the `up.3.block.1` and `up.3.block.2` of the LDM VAE decoder. This allows for high-fidelity image generation while maintaining robust dual-level watermark extraction using an EfficientNet-based extractor.

## 📦 Pretrained Checkpoints
We provide **pretrained models** for different use cases. 
The model checkpoints can be downloaded from Google Drive:
👉 **[Download checkpoints (ckpt)](https://drive.google.com/drive/folders/1ztJPBe9MYXiLTF1sfPjCOIxKDdc1nIjW)**

## 📄 Citation
If you find our work useful, please consider citing:
```bibtex
@inproceedings{fei2025scalable,
  title={Scalable Dual Fingerprinting for Hierarchical Attribution of Text-to-Image Models},
  author={Fei, Jianwei and Dai, Yunshu and Yu, Peipeng and Kong, Zhe and Zhou, Jiantao and Xia, Zhihua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15025--15034},
  year={2025}
}
```

## 📧 Contact
For any questions or collaboration inquiries, please contact: fei_jianwei@163.com
