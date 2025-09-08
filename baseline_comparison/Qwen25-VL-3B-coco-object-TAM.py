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
# from interpretation.submodular_vision import MLLMSubModularExplanationVision

from baselines.tam import TAM
from utils import SubRegionDivision, mkdir

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Submodular Explanation for Grounding DINO Model')
    # general
    parser.add_argument('--Datasets',
                        type=str,
                        default='datasets/coco/val2017',
                        help='Datasets.')
    parser.add_argument('--eval-list',
                        type=str,
                        default='datasets/coco_single_target_once_qwen25vl-7B.json',
                        help='Datasets.')
    parser.add_argument('--save-dir', 
                        type=str, default='./baseline_results/Qwen2.5-VL-7B-coco-object/TAM',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

def tam_demo_for_qwen25_vl(model, processor, image_path, prompt_text, token_id, save_path):
    # Prepare input message with image/video and prompt
    if isinstance(image_path, list):
        messages = [{"role": "user", "content": [{"type": "video", "video": image_path}, {"type": "text", "text": prompt_text}]}]
    else:
        messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]

    # Process input text and visual info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

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

    # === TAM code part ====

    # Compute logits from last hidden states with vocab classifier for TAM
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # Define special token IDs to separate image/prompt/answer tokens
    # See TAM in tam.py about its usage. See ids from the specific model.
    special_ids = {'img_id': [151652, 151653],
                   'prompt_id': [151653, [151645, 198, 151644, 77091]], 
                   'answer_id': [[198, 151644, 77091, 198], -1]}

    # get shape of vision output
    if isinstance(image_path, list):
        vision_shape = (inputs['video_grid_thw'][0, 0], inputs['video_grid_thw'][0, 1] // 2, inputs['video_grid_thw'][0, 2] // 2)
    else:
        vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)

    # get img or video inputs for next vis
    vis_inputs = [[video_inputs[0][i] for i in range(0, len(video_inputs[0]))]] if isinstance(image_path, list) else image_inputs

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
    text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
    
    # Load Qwen2.5-VL
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model.eval()
    
    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer = processor.tokenizer
    
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
            os.path.join(save_npy_root_path, content["image_path"].replace(".jpg", ".npy"))
        ):
            continue
        
        image_path = os.path.join(args.Datasets, content["image_path"])
        
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

        selected_interpretation_token_id = content["target_generated_index"]
        # selected_interpretation_token_word_id = [content["target_generated_id"]]

        image = cv2.imread(image_path)

        try:
            heatmap = tam_demo_for_qwen25_vl(model, processor, image_path, text_prompt, token_id=selected_interpretation_token_id, save_path=os.path.join(visualization_root_path, content["image_path"]))
            
            # Save npy file
            np.save(
                os.path.join(save_npy_root_path, content["image_path"].replace(".jpg", ".npy")),
                np.array(heatmap)
            )
        except:
            print("Error in TAM for ", image_path)
            continue
        
    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)