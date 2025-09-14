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
from PIL import Image

import numpy as np
# from interpretation.submodular_vision import MLLMSubModularExplanationVision

from baselines.tam_for_internvl import TAM
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
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
                        type=str, default='./baseline_results/InternVL3_5-4B-RePOPE/TAM',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def tam_demo_for_internvl(model, processor, image_path, prompt_text, token_id, save_path, counter_word_id):
    # Prepare input message with image/video and prompt
    if isinstance(image_path, list):
        messages = [{"role": "user", "content": [{"type": "video", "video": image_path}, {"type": "text", "text": prompt_text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]

    # Preparation for inference
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    # Generate model output with hidden states for visualization
    outputs = model.generate(
        **inputs,
        do_sample=False,      # Disable sampling and use greedy search instead
        num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
        max_new_tokens=128,
        output_hidden_states=True, # ---> TAM needs hidden states
        return_dict_in_generate=True
    )

    generated_ids = outputs.sequences

    target_token_position = np.array([1]) + len(inputs['input_ids'][0])
    generated_ids = generated_ids[:max(target_token_position)]
    generated_ids[0][-1] = counter_word_id[0]

    # === TAM code part ====

    # Compute logits from last hidden states with vocab classifier for TAM
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # Define special token IDs to separate image/prompt/answer tokens
    # See TAM in tam.py about its usage. See ids from the specific model.
    special_ids = {'img_id': [151671],                         # InternVL 的 image_token_id（单一）
                    'prompt_id': [151653, [151645, 198, 151644, 77091]], 
                    'answer_id': [[198, 151644, 77091, 198], -1]}

    # get shape of vision output
    # if isinstance(image_path, list):
    #     vision_shape = (inputs['video_grid_thw'][0, 0], inputs['video_grid_thw'][0, 1] // 2, inputs['video_grid_thw'][0, 2] // 2)
    # else:
    vision_shape = (16, 16)

    # get img or video inputs for next vis
    # vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]] if isinstance(image_path, list) else image_inputs
    # vis_inputs = inputs['pixel_values'][0].detach().to(torch.float32).cpu().numpy()
    # Image.open(img)
    vis_inputs = Image.open(image_path)

    # === TAM Visualization ===
    # Call TAM() to generate token activation map for each generation round
    # Arguments:
    # - token ids (inputs and generations)
    # - shape of vision token
    # - logits for each round
    # - special token identifiers for localization
    # - image / video inputs for visualization
    # - processor for decoding
    # - output image path to save the visualization
    # - round index (0 here)
    # - raw_vis_records: list to collect intermediate visualization data
    # - eval only, False to vis
    # return TAM vision map for eval, saving multimodal TAM in the function
    raw_map_records = []
    # for i in range(len(logits)):
    img_map = TAM(
        generated_ids[0].cpu().tolist(),
        vision_shape,
        logits,
        special_ids,
        vis_inputs,
        processor,
        save_path,
        token_id,
        raw_map_records,
        True)
    return img_map

def main(args):
    prompt_template = """You are asked a visual question answering task. 
    First, answer strictly with "Yes" or "No". 
    Then, provide a short explanation if necessary.

    Question: {}
    Answer:"""
    
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
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    # save_json_root_path = os.path.join(save_dir, "json")
    # mkdir(save_json_root_path)
    
    visualization_root_path = os.path.join(save_dir, "vis")
    mkdir(visualization_root_path)
    
    select_contents = contents
    
    for content in tqdm(select_contents):
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
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

        selected_interpretation_token_id = content["selected_interpretation_token_id"]
        counter_word_id = content["counter_word_id"]

        image = cv2.imread(image_path)

        text_prompt = prompt_template.format(content["question"])
        # try:
        heatmap = tam_demo_for_internvl(model, processor, image_path, text_prompt, token_id=selected_interpretation_token_id[0], save_path=os.path.join(visualization_root_path, content["image_name"].replace(".jpg", "_{}.jpg".format(content["id"]))), counter_word_id=counter_word_id)
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"]))),
            np.array(heatmap)
        )
        # except:
        #     print("Error in TAM for ", image_path)
        #     continue
        
    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)