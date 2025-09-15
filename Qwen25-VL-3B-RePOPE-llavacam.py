import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

import cv2
import json

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from baselines.llavacam import LLaVACAM
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

prompt_template = """You are asked a visual question answering task. 
First, answer strictly with "Yes" or "No". 
Then, provide a short explanation if necessary.

Question: {}
Answer:"""

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco2014/val2014',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/Qwen2.5-VL-3B-RePOPE-FP.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/Qwen2.5-VL-3B-RePOPE/LLaVACAM',
                        help='output directory to save results')
    args = parser.parse_args()
    return args


def main(args):
    # Load Qwen2.5-VL
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model.eval()
    
    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer

    explainer = LLaVACAM(model, processor, model.model.layers[32].post_attention_layernorm)
    
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
            os.path.join(save_npy_root_path, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"])))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_name"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt_template.format(content["question"])},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Data proccessing
        inputs = processor(
            text=[text],
            images=image_inputs,    # 这里可以多个
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])

        selected_interpretation_token_id = content["selected_interpretation_token_id"]
        selected_interpretation_token_word_id = content["counter_word_id"]
        
        explainer.generated_ids = torch.tensor(content["generated_ids"], dtype=torch.long).to(model.device).detach()
        explainer.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        explainer.selected_interpretation_token_word_id = selected_interpretation_token_word_id
    
        image = cv2.imread(image_path)
    
        text_prompt = prompt_template.format(content["question"])
        # Sub-region division
        heatmap = explainer.generate_smooth_cam(image_path, text_prompt)
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"]))),
            np.array(heatmap)
        )
        
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_image = cv2.imread(image_path)
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_vis_root_path, content["image_name"].replace(".jpg", "_{}.jpg".format(content["id"]))), superimposed_img)
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)