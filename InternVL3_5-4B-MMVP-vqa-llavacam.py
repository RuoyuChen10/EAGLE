import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

import cv2
import json

from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig, AutoModelForImageTextToText

import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from baselines.llavacam import LLaVACAM
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

import json
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/MMVP/MMVP Images',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/InternVL3_5-4B-MMVP-VQA.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/InternVL3_5-4B-MMVP-VQA/LLaVACAM',
                        help='output directory to save results')
    args = parser.parse_args()
    return args


def main(args):
    # text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
    
    # Load InternVL
    model_name = "OpenGVLab/InternVL3_5-4B-HF"
    # default: Load the model on the available device(s)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True).eval()

    # default processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    # for name, module in model.named_modules():
    #     print(name, '->', module.__class__.__name__)

    explainer = LLaVACAM(model, processor, model.model.language_model.layers[32].post_attention_layernorm, mode="internvl")
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_vis_root_path = os.path.join(save_dir, "visualization")
    mkdir(save_vis_root_path)
    
    # visualization_root_path = os.path.join(save_dir, "vis")
    # mkdir(visualization_root_path)
    
    for content in tqdm(contents):
        if os.path.exists(
            os.path.join(save_npy_root_path, content["image_filename"].replace(".jpg", ".npy"))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_filename"])
        text_prompt = content["question"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        
        # Preparation for inference
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        
        selected_interpretation_token_id = content["selected_interpretation_token_id"]
        selected_interpretation_token_word_id = content["selected_interpretation_token_word_id"]
        
        explainer.generated_ids = torch.tensor(content["generated_ids"], dtype=torch.long).to(model.device).detach()
        explainer.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        explainer.selected_interpretation_token_word_id = selected_interpretation_token_word_id
    
        image = cv2.imread(image_path)
    
        # Sub-region division
        heatmap = explainer.generate_smooth_cam(image_path, text_prompt)
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_filename"].replace(".jpg", ".npy")),
            np.array(heatmap)
        )
        
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_image = cv2.imread(image_path)
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_vis_root_path, content["image_filename"]), superimposed_img)
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)