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
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

import json
import cv2
import numpy as np

from baselines.IGOS_pp.utils import *
from baselines.IGOS_pp.methods_helper import *
from baselines.IGOS_pp.IGOS_pp import *

prompt_template = """You are asked a visual question answering task. 
First, answer strictly with "Yes" or "No". 
Then, provide a short explanation if necessary.

Question: {}
Answer:"""

def parse_args():
    parser = argparse.ArgumentParser(description='LLaVACAM Explanation for Qwen2.5-VL-3B Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco2014/val2014',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/InternVL3_5-4B-RePOPE-FP.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/InternVL3_5-4B-RePOPE/IGOS_PP',
                        help='output directory to save results')
    args = parser.parse_args()
    return args


def main(args):
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

    explainer = gen_explanations_internvl
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_vis_root_path = os.path.join(save_dir, "visualization")
    mkdir(save_vis_root_path)
    
    # save_json_root_path = os.path.join(save_dir, "json")
    # mkdir(save_json_root_path)
    
    # end = args.end
    # if end == -1:
    #     end = None
    # select_contents = contents[args.begin : end]
    
    for content in tqdm(contents):
        if os.path.exists(
            os.path.join(save_npy_root_path, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"])))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_name"])
        
        image = Image.open(image_path).convert("RGB")
    
        # Sub-region division
        heatmap, superimposed_img = explainer(model, processor, image, prompt_template.format(content["question"]), tokenizer, positions=[0], select_word_id = content["counter_word_id"])
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"]))),
            np.array(heatmap)
        )
        
        cv2.imwrite(os.path.join(save_vis_root_path, content["image_name"].replace(".jpg", "_{}.jpg".format(content["id"]))), superimposed_img)
        
if __name__ == "__main__":
    args = parse_args()
    
    main(args)