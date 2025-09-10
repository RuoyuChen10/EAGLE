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
from utils import SubRegionDivision, mkdir
from PIL import Image

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
    parser.add_argument('--division-number', 
                        type=int, default=64,
                        help='')
    parser.add_argument('--eval-dir', 
                        type=str, default='./baseline_results/Qwen2.5-VL-3B-coco-caption/LLaVACAM')
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
    
def perturbed(image, mask, rate = 0.5, mode = "insertion"):
    mask_flatten = mask.flatten()
    number = int(len(mask_flatten) * rate)
    
    if mode == "insertion":
        new_mask = np.zeros_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 1

        
    elif mode == "deletion":
        new_mask = np.ones_like(mask_flatten)
        index = np.argsort(-mask_flatten)
        new_mask[index[:number]] = 0
    
    new_mask = new_mask.reshape((mask.shape[0], mask.shape[1], 1))
    
    perturbed_image = image * new_mask
    return perturbed_image.astype(np.uint8)

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
    
    # Encapsulation Qwen
    Qwen = QwenVLAdaptor(
        model = model,
        processor = processor
    )
    
    save_json_root_path = os.path.join(args.eval_dir, "json")
    mkdir(save_json_root_path)
    
    npy_dir = os.path.join(args.eval_dir, "npy")
    
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
    
    for content in tqdm(contents):
        
        if "coco" in args.eval_list:
            image_path = os.path.join(args.Datasets, content["image_path"])
            save_json_path = os.path.join(save_json_root_path, content["image_path"].replace(".jpg", ".json"))
            text_prompt = "Describe the image in one factual English sentence of no more than 20 words. Do not include information that is not clearly visible."
            saliency_map = np.load(
                os.path.join(npy_dir, content["image_path"].replace(".jpg", ".npy"))
            )   # (375, 500)
        elif "MMVP" in args.eval_list:
            image_path = os.path.join(args.Datasets, content["image_filename"])
            save_json_path = os.path.join(save_json_root_path, content["image_filename"].replace(".jpg", ".json"))
            text_prompt = content["question"]
            saliency_map = np.load(
                os.path.join(npy_dir, content["image_filename"].replace(".jpg", ".npy"))
            )
        
        if os.path.exists(
            save_json_path
        ):
            continue
        
        image = cv2.imread(image_path)  # (375, 500, 3)
        
        if saliency_map.shape != image.shape[:2]:
            H, W = image.shape[:2]
            # 注意 cv2.resize 的输入是 (W, H)
            saliency_map = cv2.resize(saliency_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        json_file = {}
        json_file["insertion_score"] = []
        json_file["deletion_score"] = []
        json_file["insertion_word_score"] = []
        json_file["deletion_word_score"] = []
        json_file["region_area"] = []
        
        if "target" in args.eval_list:
            json_file["select_category"] = content["select_category"]
            json_file["location"] = content["location"]
            json_file["segmentation"] = content["segmentation"]
            
        if "target" in args.eval_list:
            selected_interpretation_token_id = [content["target_generated_index"]]
            selected_interpretation_token_word_id = [content["target_generated_id"]]
            Qwen.generated_ids = torch.tensor([content["generated_ids"]], dtype=torch.long).to(model.device).detach()
        else:
            selected_interpretation_token_id = content["selected_interpretation_token_id"]
            selected_interpretation_token_word_id = content["selected_interpretation_token_word_id"]
            Qwen.generated_ids = torch.tensor(content["generated_ids"], dtype=torch.long).to(model.device).detach()
        
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
        
        Qwen.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        Qwen.selected_interpretation_token_word_id = selected_interpretation_token_word_id
    
        for i in range(1, args.division_number+1):
            perturbed_rate = i / args.division_number
            json_file["region_area"].append(perturbed_rate)
            
            # insertion
            insertion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "insertion")
            insertion_image = Image.fromarray(cv2.cvtColor(insertion_image, cv2.COLOR_BGR2RGB))
            
            # deletion
            deletion_image = perturbed(image, saliency_map, rate = perturbed_rate, mode = "deletion")
            deletion_image = Image.fromarray(cv2.cvtColor(deletion_image, cv2.COLOR_BGR2RGB))
            
            with torch.no_grad():
                insertion_scores = Qwen(insertion_image)
                json_file["insertion_score"].append(insertion_scores.mean().item())
                json_file["insertion_word_score"].append(insertion_scores.detach().to(torch.float32).cpu().numpy().tolist())
                
                deletion_scores = Qwen(deletion_image)
                json_file["deletion_score"].append(deletion_scores.mean().item())
                json_file["deletion_word_score"].append(deletion_scores.detach().to(torch.float32).cpu().numpy().tolist())

        # Save json file
        with open(save_json_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

            
if __name__ == "__main__":
    args = parse_args()
    
    main(args)