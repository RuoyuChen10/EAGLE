# Generated from tutorial/efficient.ipynb through the first visualization section.
# Run from the project root with: conda activate qwen && python efficientv2.py

import os
import argparse

# # 🔥 Efficient Explanation

# %% Cell 2
# Set the huggingface mirror and cache path
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from torch import nn
import torchvision.transforms.functional as TF

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
import textwrap
from tqdm import tqdm

from interpretation.efficient_attribution_v2 import EfficientMLLMSubModularExplanationVisionV2

from utils import SubRegionDivision
from visualization.visualization import visualization_mllm, visualization_mllm_with_object

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=None, help="MLLM image batch size; default uses all candidates at once.")
args = parser.parse_args()

# %% Cell 3
def imshow(img):
    """
    Visualizing images inside jupyter notebook
    """
    plt.axis('off')
    if len(img.shape)==3:
        img = img[:,:,::-1] 	# transform image to rgb
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plt.imshow(img)
    plt.show()

# %% Cell 4
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
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None
        self.text_prompt = None

    def _tensor_to_pil(self, image):
        if isinstance(image, torch.Tensor):
            if image.shape[-1] == 3:
                image_tensor = image[..., [2, 1, 0]]  # BGR to RGB
                image_tensor = image_tensor.permute(2, 0, 1)
                image_tensor = image_tensor.clamp(0, 255).byte().cpu()
                return TF.to_pil_image(image_tensor)
        return image

    def _normalize_images(self, images):
        single_input = False
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = images.unsqueeze(0)
                single_input = True
            return [self._tensor_to_pil(image) for image in images], single_input
        if isinstance(images, (list, tuple)):
            return [self._tensor_to_pil(image) for image in images], single_input
        return [self._tensor_to_pil(images)], True
    
    def forward(self, images):
        image_list, single_input = self._normalize_images(images)
        info_list = []
        for image in image_list:
            content = [{"type": "image", "image": image}]
            if self.text_prompt is not None:
                content.append({"type": "text", "text": self.text_prompt})
            info_list.append({"role": "user", "content": content})

        texts = [
            self.processor.apply_chat_template([info], tokenize=False, add_generation_prompt=True)
            for info in info_list
        ]
        image_inputs, video_inputs = process_vision_info(info_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        generated_ids = self.generated_ids[:, :max(self.target_token_position)]
        generated_ids = generated_ids.expand(len(image_list), -1).clone()
        inputs["input_ids"] = generated_ids
        inputs["attention_mask"] = torch.ones_like(generated_ids)
        if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"]
            if mm_token_type_ids.shape[1] < generated_ids.shape[1]:
                text_tail = torch.zeros(
                    (mm_token_type_ids.shape[0], generated_ids.shape[1] - mm_token_type_ids.shape[1]),
                    dtype=mm_token_type_ids.dtype,
                )
                mm_token_type_ids = torch.cat([mm_token_type_ids, text_tail], dim=1)
            inputs["mm_token_type_ids"] = mm_token_type_ids[:, :generated_ids.shape[1]]
        for stale_key in ("token_type_ids", "position_ids", "cache_position"):
            inputs.pop(stale_key, None)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                return_dict=True,
                use_cache=True,
            )
            all_logits = outputs.logits

        returned_logits = all_logits[:, self.target_token_position - 1]
        returned_logits = self.softmax(returned_logits)
        if self.selected_interpretation_token_word_id is not None:
            selected_token_ids = torch.as_tensor(
                self.selected_interpretation_token_word_id, device=self.model.device
            )
            indices = selected_token_ids.unsqueeze(0).unsqueeze(-1).expand(len(image_list), -1, -1)
            returned_logits = returned_logits.gather(dim=2, index=indices).squeeze(-1)
        return returned_logits[0] if single_input else returned_logits

# ## Load the Qwen Model

# %% Cell 6
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

# ## Load the Explainer

# %% Cell 8
explainer = EfficientMLLMSubModularExplanationVisionV2(
    Qwen,
    search_scope = 8,
    pending_samples = 4,
    update_step = 10,
    batch_size = args.batch_size
)

# ## Load the Image

# %% Cell 10
image_path = "./examples/cat_on_a_tree.jpg"
text_prompt = "Describe the image in detail."

prompt_template = """{}"""

label = "yes"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt_template.format(text_prompt)},
        ],
    }
]

# %% Cell 11
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

# Inference: Generation of the output
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        do_sample=False,      # Disable sampling and use greedy search instead
        num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
        max_new_tokens=128)
    generated_ids_trimmed = [   # 去掉图像和prompt的文本
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
output_token_ids = generated_ids_trimmed[0]
output_words = [
    processor.decode([token_id.item()], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for token_id in output_token_ids
]
print(output_text)
print(output_words)
print(generated_ids_trimmed)

# ## 😮 Why answering that

# %% Cell 13
# Select all words to explain
selected_interpretation_token_id = [i for i in range(len(output_words))]
selected_interpretation_token_word_id = generated_ids_trimmed[0].tolist()

## Equip the model with the generated ids and the target token position to be explained
Qwen.generated_ids = generated_ids
Qwen.target_token_position = np.array(selected_interpretation_token_id) + len(inputs['input_ids'][0])
Qwen.selected_interpretation_token_word_id = selected_interpretation_token_word_id
Qwen.text_prompt = text_prompt

# %% Cell 14
# Image division
image = cv2.imread(image_path)
    
# Sub-region division
region_size = int((image.shape[0] * image.shape[1] / 50) ** 0.5)
V_set = SubRegionDivision(image, region_size = region_size)

# %% Cell 15
## Begin to explain
S_set, saved_json_file = explainer(image, V_set)
saved_json_file["selected_interpretation_token_id"] = selected_interpretation_token_id
saved_json_file["selected_interpretation_token_word_id"] = selected_interpretation_token_word_id
saved_json_file["words"] = output_words

# %% Cell 16
saved_json_file["smdl_score"] = (np.array(saved_json_file["insertion_score"]) + 1 - np.array(saved_json_file["deletion_score"])).tolist()

# %% Cell 17
## Visualization
visualization_mllm(image_path, S_set, saved_json_file, save_path="./test2_visualization_v2.jpg")
sentence_level_visualization_img = cv2.imread("./test2_visualization_v2.jpg")
imshow(sentence_level_visualization_img)

print("Saved visualization to ./test2_visualization_v2.jpg")
