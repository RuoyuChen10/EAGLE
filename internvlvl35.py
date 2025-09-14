import os
# Set the huggingface mirror and cache path
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"

import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoConfig, AutoModelForImageTextToText

model_name = "OpenGVLab/InternVL3_5-4B-HF"
model = AutoModelForImageTextToText.from_pretrained(
   model_name,
    device_map="auto",
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)