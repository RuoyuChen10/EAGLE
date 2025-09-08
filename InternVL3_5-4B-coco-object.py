import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

import math
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image

import torch
from torch import nn
import torchvision.transforms.functional as TF
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig, AutoModelForImageTextToText

import argparse
import json
import cv2
import numpy as np
from interpretation.submodular_vision import MLLMSubModularExplanationVision
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
                        default='datasets/coco_single_target_once_internvl-4B.json',
                        help='Datasets.')
    parser.add_argument('--superpixel-algorithm',
                        type=str,
                        default="slico",
                        choices=["slico", "seeds"],
                        help="")
    parser.add_argument('--lambda1', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--lambda2', 
                        type=float, default=1.,
                        help='')
    parser.add_argument('--division-number', 
                        type=int, default=64,
                        help='')
    parser.add_argument('--begin', 
                        type=int, default=0,
                        help='')
    parser.add_argument('--end', 
                        type=int, default=-1,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./interpretation_results/InternVL3_5-4B-coco-object/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class InternVLAdaptor(torch.nn.Module):
    def __init__(self, 
                 model,
                 processor,
                 device = "cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        
        self.processor = processor
        
        self.text_prompt = None
        self.generated_ids = None
        
        # The position of the token that needs to be explained in the newly generated content (include all tokens)
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None
        
    def forward(self, image):
        """_summary_

        Args:
            image: PIL format
        """
        if isinstance(image, torch.Tensor):
            if image.shape[-1] == 3:
                image_tensor = image[..., [2, 1, 0]]  # BGR → RGB
                image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.clamp(0, 255).byte()
                image = TF.to_pil_image(image_tensor)
                
        info = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": self.text_prompt},
                ],
            },
        ]
        
        # Preparation for inference
        inputs = self.processor.apply_chat_template(info, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)
        
        self.generated_ids = self.generated_ids[:max(self.target_token_position)]   #bug
        inputs['input_ids'] = self.generated_ids
        inputs = inputs.to(self.model.device)    # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
        
        # Forward calculation to get all logits (including the logits of the input part)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=True,
            )
            all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        if self.generated_ids != None:
            returned_logits = all_logits[:, self.target_token_position - 1] # The reason for the minus 1 is that the generated content is in the previous position
            returned_logits = self.softmax(returned_logits)
            
            if self.selected_interpretation_token_word_id != None:
                self.selected_interpretation_token_word_id = torch.tensor(self.selected_interpretation_token_word_id).to(self.model.device)
                indices = self.selected_interpretation_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
                
                returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
                
                returned_logits = returned_logits.squeeze(-1)  # [1, N]
        return returned_logits[0]   # size [N]

def main(args):
    text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
    
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
    
    # Encapsulation Qwen
    InternVL = InternVLAdaptor(
        model = model,
        processor = processor
    )
    
    # Submodular
    smdl = MLLMSubModularExplanationVision(
        InternVL,
        lambda1=args.lambda1,
        lambda2=args.lambda2
    )
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
        
    mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, "{}-{}-{}-division-number-{}".format(args.superpixel_algorithm, args.lambda1, args.lambda2, args.division_number))
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_contents = contents[args.begin : end]
    
    for content in tqdm(select_contents):
        if os.path.exists(
            os.path.join(save_json_root_path, content["image_path"].replace(".jpg", ".json"))
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
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        
        selected_interpretation_token_id = [content["target_generated_index"]]
        selected_interpretation_token_word_id = [content["target_generated_id"]]

        
        InternVL.generated_ids = torch.tensor([content["generated_ids"]], dtype=torch.long).to(model.device).detach()
        InternVL.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        InternVL.selected_interpretation_token_word_id = selected_interpretation_token_word_id
        InternVL.text_prompt = text_prompt
        
        image = cv2.imread(image_path)
        
        # Sub-region division
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        V_set = SubRegionDivision(image, mode=args.superpixel_algorithm, region_size = region_size)
        
        S_set, saved_json_file = smdl(image, V_set)
        saved_json_file["selected_interpretation_token_id"] = selected_interpretation_token_id
        saved_json_file["selected_interpretation_token_word_id"] = selected_interpretation_token_word_id
        saved_json_file["select_category"] = content["select_category"]
        saved_json_file["words"] = content["target_generated_token"]
         
        saved_json_file["location"] = content["location"]
        saved_json_file["segmentation"] = content["segmentation"]
        
        saved_json_file["output_word_id"] = content["output_word_id"]
        saved_json_file["output_words"] = processor.batch_decode(
            content["output_word_id"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, content["image_path"].replace(".jpg", ".npy")),
            np.array(S_set)
        )
        
        # Save json file
        with open(
            os.path.join(save_json_root_path, content["image_path"].replace(".jpg", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
    
if __name__ == "__main__":
    args = parse_args()
    
    main(args)