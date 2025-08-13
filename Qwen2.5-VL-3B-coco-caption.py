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
                        default='datasets/Qwen2.5-VL-3B-coco-caption.json',
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
                        type=str, default='./interpretation_results/Qwen2.5-VL-3B-coco-caption/',
                        help='output directory to save results')
    args = parser.parse_args()
    return args

class QwenVLAdaptor(torch.nn.Module):
    def __init__(self, 
                 model,
                 processor,
                 device = "cuda"):
        super().__init__()
        self.model = model
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        
        self.processor = processor
        
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
    
    # Load Qwen2.5-VL
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    model.eval()
    
    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    tokenizer = processor.tokenizer
    
    # Encapsulation Qwen
    Qwen = QwenVLAdaptor(
        model = model,
        processor = processor
    )

    # Submodular
    smdl = MLLMSubModularExplanationVision(
        Qwen,
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
    
    # visualization_root_path = os.path.join(save_dir, "vis")
    # mkdir(visualization_root_path)
    
    end = args.end
    if end == -1:
        end = None
    select_contents = contents[args.begin : end]
    
    for content in tqdm(select_contents):
        if os.path.exists(
            os.path.join(save_json_root_path, content["image_path"].replace(".jpg", ".json"))
        ):
            # id += 1
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

        selected_interpretation_token_id = content["selected_interpretation_token_id"]
        selected_interpretation_token_word_id = content["selected_interpretation_token_word_id"]
        
        Qwen.generated_ids = torch.tensor(content["generated_ids"], dtype=torch.long).to(model.device).detach()
        Qwen.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        Qwen.selected_interpretation_token_word_id = selected_interpretation_token_word_id
    
        image = cv2.imread(image_path)
    
        # Sub-region division
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        V_set = SubRegionDivision(image, mode=args.superpixel_algorithm, region_size = region_size)
        
        S_set, saved_json_file = smdl(image, V_set)
        saved_json_file["selected_interpretation_token_id"] = selected_interpretation_token_id
        saved_json_file["selected_interpretation_token_word_id"] = selected_interpretation_token_word_id
        saved_json_file["words"] = content["words"]
        
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