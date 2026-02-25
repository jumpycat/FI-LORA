import argparse
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import os

def get_parser():
    parser = argparse.ArgumentParser(description="Extract and verify dual fingerprints")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the watermarked image")
    parser.add_argument("--resume", type=str, required=True, help="Path to the trained checkpoint (.pth)")
    parser.add_argument("--bits", type=int, default=32, help="Length of a single watermark (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used during generation for GT watermark")
    parser.add_argument("--img_size", type=int, default=512, help="Image size for extraction")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load the EfficientNet-based Watermark Decoder
    print("Loading watermark decoder...")
    wm_decoder = EfficientNet.from_pretrained('efficientnet-b0')
    feature = wm_decoder._fc.in_features
    # The decoder outputs concatenated W1 and W2, so out_features = bits * 2
    wm_decoder._fc = nn.Linear(in_features=feature, out_features=args.bits * 2, bias=True)
    
    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"Checkpoint not found at {args.resume}")
        
    checkpoint = torch.load(args.resume, map_location='cpu')
    wm_decoder.load_state_dict(checkpoint["wm_decoder"])
    wm_decoder.to(device)
    wm_decoder.eval()

    # 2. Re-generate Ground Truth Watermarks based on the Seed
    torch.manual_seed(args.seed)
    gt_watermark1 = torch.zeros((1, args.bits), dtype=torch.float).random_(0, 2).to(device)
    gt_watermark2 = torch.zeros((1, args.bits), dtype=torch.float).random_(0, 2).to(device)

    # 3. Load and Preprocess Image
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
    
    img = Image.open(args.img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Extract and Compare
    with torch.no_grad():
        decoded = wm_decoder(img_tensor)
        
    # Predictions > 0.0 are mapped to 1.0, otherwise 0.0
    wm_predictedA = (decoded[:, :args.bits] > 0.0).float()
    wm_predictedB = (decoded[:, args.bits:] > 0.0).float()

    # Calculate bitwise accuracy
    bitwise_acc1 = 100 * (1.0 - torch.mean(torch.abs(gt_watermark1 - wm_predictedA)))
    bitwise_acc2 = 100 * (1.0 - torch.mean(torch.abs(gt_watermark2 - wm_predictedB)))

    print("-" * 40)
    print(f"Extraction Results for: {args.img_path}")
    print(f"Watermark 1 (Model-level) Accuracy: {bitwise_acc1.item():.2f}%")
    print(f"Watermark 2 (User-level)  Accuracy: {bitwise_acc2.item():.2f}%")
    print(f"Average Dual Accuracy:      {(bitwise_acc1.item() + bitwise_acc2.item()) / 2:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()
