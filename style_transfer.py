"""
Neural Style Transfer (Gatys et al., 2015) using VGG19.
Turns a content photo into the style of a reference poster image.

Usage:
    python style_transfer.py --content content/photo.jpg --style styles/poster1.jpg --output outputs/result.jpg
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMG_SIZE = 512 if device.type == "cuda" else 256  # smaller on CPU for speed

loader = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x


# ---------------------------------------------------------------------------
# Total variation loss (smooths out pixel-level noise)
# ---------------------------------------------------------------------------

def total_variation_loss(img):
    diff_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    diff_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return diff_h + diff_w


# ---------------------------------------------------------------------------
# Normalization (VGG expects ImageNet-normalized input)
# ---------------------------------------------------------------------------

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# ---------------------------------------------------------------------------
# Build the model: VGG19 layers interleaved with loss modules
# ---------------------------------------------------------------------------

CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def build_model_and_losses(content_img, style_img):
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in STYLE_LAYERS:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        if i >= 5:
            break

    return model, style_losses, content_losses


# ---------------------------------------------------------------------------
# Run the optimization (Adam — stable, forgiving; content_img → styled image)
# ---------------------------------------------------------------------------

def run_style_transfer(content_img, style_img, num_steps=1000,
                        style_weight=1e5, content_weight=1, tv_weight=1e-6,
                        lr=0.02):
    model, style_losses, content_losses = build_model_and_losses(content_img, style_img)

    input_img = content_img.clone()
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.Adam([input_img], lr=lr)

    print("Optimizing...")
    for step in range(1, num_steps + 1):
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        tv_score = total_variation_loss(input_img)

        loss = (style_weight * style_score
                + content_weight * content_score
                + tv_weight * tv_score)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:4d}  style loss: {style_score.item():.4f}  "
                  f"content loss: {content_score.item():.4f}  "
                  f"tv loss: {tv_score.item():.4f}")

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--style-weight", type=float, default=1e5)
    parser.add_argument("--content-weight", type=float, default=1)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    content_img = load_image(args.content)
    style_img = load_image(args.style)

    assert content_img.size() == style_img.size(), (
        "Content and style images must be the same size after resize."
    )

    output = run_style_transfer(
        content_img, style_img,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        lr=args.lr,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(output, args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()