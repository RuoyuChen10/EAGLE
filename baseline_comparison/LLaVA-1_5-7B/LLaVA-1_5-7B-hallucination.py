import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

import cv2
import json

from transformers import AutoProcessor, LlavaForConditionalGeneration

import argparse
import torch
from torch import nn
import torchvision.transforms.functional as TF

import numpy as np
from utils import SubRegionDivision, mkdir
from PIL import Image

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
                        default='datasets/LLaVA-1_5-7B-RePOPE-FP.json',
                        help='Datasets.')
    parser.add_argument('--division-number', 
                        type=int, default=64,
                        help='')
    parser.add_argument('--eval-dir', 
                        type=str, default='./baseline_results/LLaVA-1_5-7B-RePOPE/LLaVACAM')
    args = parser.parse_args()
    return args

class LLaVAAdaptor(torch.nn.Module):
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
    # Load InternVL
    model_name = "llava-hf/llava-1.5-7b-hf"
    # default: Load the model on the available device(s)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, 
        dtype=torch.float16, 
        device_map="auto",
        # low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    # default processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    # Encapsulation Qwen
    LLaVA = LLaVAAdaptor(
        model = model,
        processor = processor
    )
    
    save_json_root_path = os.path.join(args.eval_dir, "json")
    mkdir(save_json_root_path)
    
    npy_dir = os.path.join(args.eval_dir, "npy")
    
    
    with open(args.eval_list, "r") as f:
        contents = json.load(f)
    
    for content in tqdm(contents):
        image_path = os.path.join(args.Datasets, content["image_name"])
        save_json_path = os.path.join(save_json_root_path, content["image_name"].replace(".jpg", "_{}.json".format(content["id"])))
        text_prompt = prompt_template.format(content["question"])
        saliency_map = np.load(
            os.path.join(npy_dir, content["image_name"].replace(".jpg", "_{}.npy".format(content["id"])))
        )   # (375, 500)

        if os.path.exists(
            save_json_path
        ):
            continue

        image = cv2.imread(image_path)  # (375, 500, 3)
        
        if "IGOS_PP" in args.eval_dir:  # IGOS++ output is different than others
            saliency_map = 1.0 - saliency_map
        if "TAM" in args.eval_dir:
            saliency_map = 1.0 - saliency_map
        
        if saliency_map.shape != image.shape[:2]:
            H, W = image.shape[:2]
            # 注意 cv2.resize 的输入是 (W, H)
            saliency_map = cv2.resize(saliency_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        json_file = {}
        json_file["question"] = content["question"]
        json_file["label"] = content["label"]
        json_file["id"] = content["id"]
        json_file["counter_word_id"] = content["counter_word_id"]
        json_file["words"] = content["words"]
        json_file["insertion_score"] = []
        json_file["deletion_score"] = []
        json_file["region_area"] = []
        json_file["revised_answering"] = []
        
        selected_interpretation_token_id = content["selected_interpretation_token_id"]
        selected_interpretation_token_word_id = content["selected_interpretation_token_word_id"]
        InternVL.generated_ids = torch.tensor(content["generated_ids"], dtype=torch.long).to(model.device).detach()
        
        json_file["selected_interpretation_token_id"] = selected_interpretation_token_id
        json_file["selected_interpretation_token_word_id"] = selected_interpretation_token_word_id
        
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
        
        LLaVA.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
        LLaVA.selected_interpretation_token_word_id = selected_interpretation_token_word_id
        LLaVA.text_prompt = text_prompt
    
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
                insertion_scores = InternVL(insertion_image)
                json_file["insertion_score"].append(insertion_scores.mean().item())
                
                deletion_scores = InternVL(deletion_image)
                json_file["deletion_score"].append(deletion_scores.mean().item())
                
                messages_new = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": insertion_image,
                            },
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
                # Preparation for inference
                inputs = processor.apply_chat_template(messages_new, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                
                generated_ids_revised = model.generate(
                    **inputs, 
                    do_sample=False,      # Disable sampling and use greedy search instead
                    num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
                    max_new_tokens=128)
                generated_ids_trimmed_revised = [   # 去掉图像和prompt的文本
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids_revised)
                ]
                
                
                output_text = processor.batch_decode(
                    generated_ids_trimmed_revised, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                json_file["revised_answering"].append(output_text[0])

        # Save json file
        with open(save_json_path, "w") as f:
            f.write(json.dumps(json_file, ensure_ascii=False, indent=4, separators=(',', ':')))

            
if __name__ == "__main__":
    args = parse_args()
    
    main(args)