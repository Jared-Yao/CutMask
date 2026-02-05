# -*- coding: utf-8 -*-
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from model1 import ProjectionNet 



# 1) Load model weights (.tch = state_dict)

def load_projectionnet(weight_path: str, head_layer: int = 2, device: str = "cuda"):
    sd = torch.load(weight_path, map_location="cpu")
    num_classes = sd["out.weight"].shape[0]  # Infer number of classes from out.weight
    head_layers = [512] * head_layer + [128]
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model



# 2) Grad-CAM implementation

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Forward hook: save feature maps
        def fwd_hook(module, inp, out):
            self.activations = out

        # Backward hook: save gradients
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    @torch.enable_grad()
    def __call__(self, x: torch.Tensor, target_class: int | None = None):
        """
        x: (1,3,H,W) input already normalized
        target_class: which class to visualize; if None, use the predicted class
        """
        self.model.zero_grad(set_to_none=True)

        embeds, logits, cost_sspcab = self.model(x)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)

        # activations: (1,C,h,w), gradients: (1,C,h,w)
        A = self.activations
        G = self.gradients
        # Channel weights: spatial average of gradients
        w = G.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (w * A).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)

        # Normalize to 0~1
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, target_class



# 3) Overlay heatmap on the original image

def overlay_heatmap_on_image(img_bgr: np.ndarray, cam_01: np.ndarray, alpha: float = 0.45):
    """
    img_bgr: original image in OpenCV BGR format
    cam_01: heatmap normalized to 0~1
    """
    H, W = img_bgr.shape[:2]
    cam = cv2.resize(cam_01, (W, H))
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(heat, alpha, img_bgr, 1 - alpha, 0)
    return out


def main(
    weight_path: str,
    img_path: str,
    out_path: str = "gradcam.png",
    size: int = 256,
    head_layer: int = 2,
    device: str = "cuda",
    target_class: int | None = None
):
    # Input preprocessing
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    model = load_projectionnet(weight_path, head_layer=head_layer, device=device)

    # It is recommended to set target_layer as the last convolutional feature block in the model
    cam_gen = GradCAM(model, target_layer=model.sspcab_block)

    # Load image: PIL->Tensor (for model) + OpenCV BGR (for overlay)
    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(device)

    cam_01, cls = cam_gen(x, target_class=target_class)

    img_bgr = cv2.cvtColor(np.array(pil.resize((size, size))), cv2.COLOR_RGB2BGR)
    vis = overlay_heatmap_on_image(img_bgr, cam_01, alpha=0.45)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"[OK] saved to {out_path}, target_class={cls}")


if __name__ == "__main__":
    # Replace with your model weights and image path
    main(
        weight_path="models/model-bottle-xxxx.tch",
        img_path="Data/bottle/test/good/000.png",
        out_path="eval/gradcam/gradcam_bottle_000.png",
        size=256,
        head_layer=2,
        device="cuda",
        target_class=None
    )
