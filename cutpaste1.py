import random
import math
from torchvision import transforms
import torch
from PIL import Image
import os
import cv2
import numpy as np
from PIL import ImageFilter

#from fenli import auto_crop
def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
#     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]
    

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img
    
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)
        
        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)

        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(img, augmented)

class CutPasteMaskNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewhere else.
    Args:
        area_ratio (list): [min, max] area ratio for the cutout region
        aspect_ratio (float): min aspect ratio; sampled between aspect_ratio and 1/aspect_ratio
    """
    def __init__(self, area_ratio=[0.02, 0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteMaskNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
     
        # 读取先验蒙版
        imagelist = os.listdir('maskpackge1/')
        rootdir = "maskpackge1/"
        mask_list = int(random.uniform(0, len(imagelist)))
        mask = Image.open(rootdir + imagelist[mask_list])  # 建议为灰度/带alpha的PNG
     

        # NOTE: PIL.Image.size = (width, height)
        h = img.size[1]
        w = img.size[0]

        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h

        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1 / self.aspect_ratio)))
        aspect = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        # 剪切的宽和高
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))

        # from-location（当前仅自样本采样）
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)  # RGB

        # 将先验mask缩放到patch尺寸，并用它把patch镂空到RGBA（带alpha）
        mask = mask.resize((cut_w, cut_h), Image.LANCZOS)  # 建议mask为L或RGBA
        # 统一拿到一个“L”通道作为alpha
        if mask.mode == "RGBA":
            alpha = mask.split()[3]
        elif mask.mode == "LA":
            alpha = mask.split()[1]
        else:
            alpha = mask.convert("L")
        m2 = Image.new('RGBA', (cut_w, cut_h))
        m2.paste(patch.convert("RGBA"), (0, 0), mask=alpha)  # 用先验mask作为alpha得到带透明边的patch
        patch = m2  # RGBA patch，带alpha

        if self.colorJitter:
            # 对RGBA的RGB部分做抖动
            rgb = patch.convert("RGB")
            rgb = self.colorJitter(rgb)
            patch = Image.merge("RGBA", (*rgb.split(), patch.split()[3]))

        # 目标位置
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]

        # 使用patch自身的alpha作为粘贴mask，避免矩形边 
        augmented = img.convert("RGBA").copy()
        augmented.paste(patch, insert_box, mask=patch.split()[3])  # 用自身 alpha
        augmented = augmented.convert("RGB")
     

        return super().__call__(img, augmented)


class CutPasteMaskscar(CutPaste):
    """在裁剪块上用先验蒙版做镂空，再旋转并按 alpha 贴回原图"""

    def __init__(self, width=(2, 16), height=(10, 25), rotation=(-45, 45), mask_dir="maskpackge1", **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.rotation = rotation
        self.mask_dir = mask_dir
        # 仅保留图片文件
        self._mask_files = [f for f in os.listdir(mask_dir)
                            if f.lower().endswith((".png", ".jpg", ".bmp", ".webp"))]
        if not self._mask_files:
            raise RuntimeError(f"No mask images found in '{mask_dir}'")

    def __call__(self, img):
        # 保证输入为 RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size  # 注意 PIL 返回的是 (width, height)

        # 随机裁剪尺寸并裁剪
        cut_w = max(1, min(int(random.uniform(*self.width)), w - 1))
        cut_h = max(1, min(int(random.uniform(*self.height)), h - 1))

        x0 = 0 if w - cut_w <= 0 else random.randrange(0, w - cut_w + 1)
        y0 = 0 if h - cut_h <= 0 else random.randrange(0, h - cut_h + 1)
        box = (x0, y0, x0 + cut_w, y0 + cut_h)
        patch_rgb = img.crop(box)  # 目标裁剪块（RGB）

        # 读取一张蒙版，转 L 作为 alpha，并缩放到裁剪块大小
        mask_name = random.choice(self._mask_files)
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask_img = Image.open(mask_path)
        alpha = mask_img.convert("L").resize((cut_w, cut_h), Image.LANCZOS)

        # 组合成 RGBA
        patch_rgba = Image.new("RGBA", (cut_w, cut_h), (0, 0, 0, 0))
        patch_rgba.paste(patch_rgb, (0, 0), mask=alpha)

        # 只对 RGB 做颜色抖动，保留 alpha，不会把透明区搞黑
        if getattr(self, "colorJitter", None) is not None:
            rgb_only = patch_rgba.convert("RGB")
            rgb_only = self.colorJitter(rgb_only)
            patch_rgba = Image.merge("RGBA", (*rgb_only.split(), alpha))

        # 旋转，保留 alpha
        rot_deg = random.uniform(*self.rotation)
        patch_rgba = patch_rgba.rotate(rot_deg, expand=True, resample=Image.BICUBIC)

        patch_w, patch_h = patch_rgba.size

        # 如果旋转后过大，则缩小以适配原图
        if patch_w >= w or patch_h >= h:
            scale = min((w - 1) / patch_w, (h - 1) / patch_h)
            new_w = max(1, int(patch_w * scale))
            new_h = max(1, int(patch_h * scale))
            patch_rgba = patch_rgba.resize((new_w, new_h), Image.LANCZOS)
            patch_w, patch_h = patch_rgba.size

        # 随机粘贴位置
        to_x = 0 if w - patch_w <= 0 else random.randrange(0, w - patch_w + 1)
        to_y = 0 if h - patch_h <= 0 else random.randrange(0, h - patch_h + 1)

        # 使用自身 alpha 作为粘贴 mask
        augmented = img.convert("RGBA")
        augmented.paste(patch_rgba, (to_x, to_y), mask=patch_rgba.split()[3])
        augmented = augmented.convert("RGB")

        return super().__call__(img, augmented)

class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.scar(img)

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar

class CutPasteMask3Way(object):
    def __init__(self, **kwags):
        self.masknormal = CutPasteMaskNormal(**kwags)
        self.maskscar = CutPasteMaskscar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_masknormal = self.masknormal(img)
        _, cutpaste_maskscar = self.maskscar(img)
        
        return org,cutpaste_masknormal, cutpaste_maskscar






