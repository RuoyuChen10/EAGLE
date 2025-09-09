import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

from PIL import Image
from torchvision.utils import save_image
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from einops import rearrange
import math

from qwen_vl_utils import process_vision_info

class LLaVACAM(object):
    def __init__(self, model, processor, 
                 target_layer, num_samples=50, noise_std=0.1, mode="qwen", device = "cuda"):
        self.model = model  # 要进行LLaVA-CAM处理的模型
        self.device = device
        self.target_layer = target_layer  # 要进行特征可视化的目标层
        self.feature_maps = None  # 存储特征图
        self.gradients = None  # 存储梯度

        # 为目标层添加钩子，以保存输出和梯度
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_backward_hook(self.save_gradients)
        
        self.num_samples = num_samples  # 平滑次数
        self.noise_std = noise_std  # 噪声标准差
        
        self.mode = mode
        
        self.processor = processor
        self.generated_ids = None
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None

    def save_feature_maps(self, module, input, output):
        """保存特征图"""
        # output.requires_grad = True
        self.feature_maps = output
        output.retain_grad()
        # self.feature_maps.requires_grad = True
        num_token = self.feature_maps.shape[1]
        h = int(np.sqrt(num_token))
        # self.feature_maps = self.feature_maps[0,1:,:].reshape((1,h,h,-1))

    def save_gradients(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
        # print(self.gradients)
        
    def qwen_vl_tokens_from_size(self, H, W):
        
        patch_size = self.model.config.vision_config.patch_size
        spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        n_h = math.ceil(H / patch_size)
        n_w = math.ceil(W / patch_size)
        Hm = n_h // spatial_merge_size      # 丢弃不满 s 的行
        Wm = n_w // spatial_merge_size      # 丢弃不满 s 的列
        
        return Hm, Wm
    
    def generate_cam(self, image, qu):
        """生成CAM热力图"""
        # 将模型设置为评估模式
        self.model.eval()
        # 清空所有梯度
        self.model.zero_grad()
        
        # 正向传播
        if self.mode == "qwen":
            info = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": qu},
                    ],},
                ]
            # Preparation for inference
            text = self.processor.apply_chat_template(
                info, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(info)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,    # 这里可以多个
                padding=True,
                return_tensors="pt",
            )
            self.generated_ids = self.generated_ids[:max(self.target_token_position)]   #bug
            inputs['input_ids'] = self.generated_ids
            inputs['attention_mask'] = torch.ones_like(self.generated_ids)
            inputs = inputs.to(self.model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
            
            vision_mask = (inputs["input_ids"] == self.model.config.image_token_id).to(self.model.device)
            
            W, H = image_inputs[0].size
            Hm, Wm = self.qwen_vl_tokens_from_size(H, W)
            
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=False,
            )
            
            all_logits = outputs.logits
            returned_logits = all_logits[:, self.target_token_position - 1] # The reason for the minus 1 is that the generated content is in the previous position
            self.selected_interpretation_token_word_id = torch.tensor(self.selected_interpretation_token_word_id).to(self.model.device)
            indices = self.selected_interpretation_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
                
            returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
            returned_logits = returned_logits.squeeze(-1)  # [1, N]
            
            # 对目标类进行反向传播
            target_logits = torch.sum(returned_logits[0])
        
        elif self.mode == "internvl":
            info = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": qu},
                    ],},
                ]
            # Preparation for inference
            inputs = self.processor.apply_chat_template(info, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
            self.generated_ids = self.generated_ids[:max(self.target_token_position)]   #bug
            inputs['input_ids'] = self.generated_ids
            inputs['attention_mask'] = torch.ones_like(self.generated_ids)
            inputs = inputs.to(self.model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
            
            vision_mask = (inputs["input_ids"] == self.model.config.image_token_id).to(self.model.device)

            if isinstance(image, str):
                img = cv2.imread(image.replace(" ", "_"))
                H, W = img.shape[:2]
            elif isinstance(image, Image.Image):
                W, H = image.size

            Hm, Wm = 16, 16
            
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=False,
            )
            
            all_logits = outputs.logits
            returned_logits = all_logits[:, self.target_token_position - 1] # The reason for the minus 1 is that the generated content is in the previous position
            self.selected_interpretation_token_word_id = torch.tensor(self.selected_interpretation_token_word_id).to(self.model.device)
            indices = self.selected_interpretation_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
                
            returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
            returned_logits = returned_logits.squeeze(-1)  # [1, N]
            
            # 对目标类进行反向传播
            target_logits = torch.sum(returned_logits[0])

        target_logits.retain_grad()
        target_logits.backward(retain_graph=True)

        # num_token = self.feature_maps.shape[1]
        # h = int(np.sqrt(num_token))
        # self.feature_maps = self.feature_maps[0:,1:,:].detach().reshape((1,h,h,-1))
        # self.feature_maps = rearrange(self.feature_maps[0:,34:610,:].detach(),'b (h w) c -> b c h w ',w=24,h=24)
        
        self.image_feature_maps = []
        self.image_gradients = []
        vision_mask = vision_mask.to(self.feature_maps.device)
        for b in range(self.feature_maps.shape[0]):
            feats = self.feature_maps[b][vision_mask[b]]  # [N_img_b, C]
            grads = self.gradients[b][vision_mask[b]]
            
            if self.mode == "internvl":
                feats = feats[-256:]
                grads = grads[-256:]
                
            self.image_feature_maps.append(feats)
            self.image_gradients.append(grads)
        self.image_feature_maps = torch.stack(self.image_feature_maps)
        self.image_gradients = torch.stack(self.image_gradients)
        
        # 假设 self.feature_maps: [B, L, C]
        B, L, C = self.image_feature_maps.shape # torch.Size([1, 391, 2048])
        h, w = Hm, Wm
        self.image_feature_maps = rearrange(self.image_feature_maps.detach(), 'b (h w) c -> b h w c', h=h, w=w)
        
        # 获取平均梯度和特征图
        self.image_gradients = rearrange(self.image_gradients.detach(), 'b (h w) c -> b h w c', h=h, w=w)

        self.image_gradients = nn.ReLU()(self.image_gradients)
        pooled_gradients = torch.mean(self.image_gradients, dim=[0, 1, 2])
        activation = self.image_feature_maps.squeeze(0)
        
        weighted_act = activation * pooled_gradients

        # 创建热力图
        # activation = activation.permute(0,2,1)
        heatmap = torch.mean(weighted_act, dim=-1, dtype=torch.float32).squeeze().cpu().numpy().astype(np.float32)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # 特征筛选
        threshold = 0.5 # 可以调整这个值
        # heatmap[heatmap < threshold] = 0  # 将低于阈值的值设为0

        if isinstance(image, str):
            image_cv = cv2.imread(image)
            h, w = image_cv.shape[:2]
        elif isinstance(image, Image.Image):
            w, h = image.size   # PIL 的 size 顺序是 (width, height)
        elif isinstance(image, (np.ndarray,)):
            h, w = image.shape[:2]
        else:
            raise TypeError(f"不支持的输入类型: {type(image)}")
        heatmap = cv2.resize(heatmap, (w, h))
        # heatmap = cv2.resize(heatmap, (image.size(3), image.size(2)))
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap

    def unprocess_image(self, image):
        """反预处理图像，将其转回原始图像"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (((image.transpose(1, 2, 0) * std) + mean) * 255).astype(np.uint8)
        return image
    
    def add_noise(self, image, noise_std):
        """向输入张量添加高斯噪声"""
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.clone().convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        # 转为 numpy 数组 (H, W, C)，范围 [0, 255]
        img_arr = np.array(pil_img).astype(np.float32)

        # 生成高斯噪声
        noise = np.random.normal(0, noise_std * 255, img_arr.shape)

        # 添加噪声并裁剪
        noisy_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        # 转回 PIL
        noisy_pil = Image.fromarray(noisy_arr)

        return noisy_pil

    def generate_smooth_cam(self, image, qu):
        base_cam = self.generate_cam(image, qu)  # 基础的Grad-CAM结果
        
        # 初始化平滑后的热力图和叠加图像
        smooth_cam = np.zeros_like(base_cam)

        for _ in range(self.num_samples):
            # 添加噪声到图像
            noisy_image = self.add_noise(image, self.noise_std)
            noisy_cam = self.generate_cam(noisy_image, qu)

            # 对每次扰动的结果累加
            smooth_cam += noisy_cam

        # 取平均得到最终的SmoothGrad-CAM热力图
        smooth_cam /= self.num_samples
        smooth_cam = np.maximum(smooth_cam, 0)
        smooth_cam /= np.max(smooth_cam)
        # smooth_cam = cv2.resize(smooth_cam, (image.size(3), image.size(2)))
        # smooth_cam = np.uint8(255 * smooth_cam)
        # smooth_cam = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET)

        # 将最终的热力图叠加到原始图像上
        # original_image = self.unprocess_image(image.squeeze().cpu().numpy())
        # superimposed_img = smooth_cam * 0.4 + original_image
        # superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return smooth_cam