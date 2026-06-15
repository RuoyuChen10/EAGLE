import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenTextAdaptor(nn.Module):
    def __init__(self, model, tokenizer, model_name="Qwen/Qwen3-8B", device=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device or getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.softmax = nn.Softmax(dim=-1)

        self.prompt_length = None
        self.generated_answer_ids = None
        self.target_token_position = None
        self.selected_interpretation_token_word_id = None

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is not None:
            self.mask_token_id = self.tokenizer.pad_token_id
        else:
            self.mask_token_id = self.tokenizer.eos_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer needs either pad_token_id or eos_token_id for text masking.")

    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, model_name=model_name)

    def tokenize_rendered_chat(self, rendered_text):
        return self.tokenizer(
            [rendered_text],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.model.device)

    def generate_from_rendered_chat(self, rendered_text, max_new_tokens=128, do_sample=False):
        inputs = self.tokenize_rendered_chat(rendered_text)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=do_sample,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )

        prompt_length = inputs["input_ids"].shape[1]
        answer_ids = generated_ids[0, prompt_length:]
        output_text = self.tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_tokens = [
            self.tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for token_id in answer_ids
        ]

        self.prompt_length = prompt_length
        self.generated_answer_ids = answer_ids.detach().clone()
        self.target_token_position = torch.arange(
            prompt_length,
            prompt_length + len(answer_ids),
            device=self.model.device,
            dtype=torch.long,
        )
        self.selected_interpretation_token_word_id = answer_ids.detach().clone()

        return {
            "input_ids": inputs["input_ids"][0].detach().cpu(),
            "attention_mask": inputs["attention_mask"][0].detach().cpu(),
            "generated_ids": generated_ids.detach().cpu(),
            "generated_answer_ids": answer_ids.detach().cpu(),
            "output_text": output_text,
            "output_tokens": output_tokens,
        }

    def set_targets(self, generated_answer_ids, selected_output_token_indices=None):
        if self.prompt_length is None:
            raise ValueError("prompt_length is not set. Run generate_from_rendered_chat first.")

        answer_ids = torch.as_tensor(generated_answer_ids, device=self.model.device, dtype=torch.long)
        if selected_output_token_indices is None:
            selected_output_token_indices = list(range(len(answer_ids)))
        selected_indices = torch.as_tensor(
            selected_output_token_indices,
            device=self.model.device,
            dtype=torch.long,
        )
        self.generated_answer_ids = answer_ids
        self.target_token_position = self.prompt_length + selected_indices
        self.selected_interpretation_token_word_id = answer_ids[selected_indices]

    def forward(self, input_ids, attention_mask=None):
        if self.prompt_length is None or self.generated_answer_ids is None:
            raise ValueError("Generation targets are not initialized.")
        if self.target_token_position is None or self.selected_interpretation_token_word_id is None:
            raise ValueError("Selected interpretation targets are not initialized.")

        single_input = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            single_input = True
        input_ids = input_ids.to(self.model.device, dtype=torch.long)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.model.device)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.model.device, dtype=torch.long)

        max_target_position = int(self.target_token_position.max().item())
        generated_prefix_len = max(0, max_target_position - self.prompt_length)
        generated_prefix = self.generated_answer_ids[:generated_prefix_len]
        if generated_prefix_len:
            prefix_batch = generated_prefix.unsqueeze(0).expand(input_ids.shape[0], -1)
            input_ids = torch.cat([input_ids, prefix_batch], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(prefix_batch, device=self.model.device)],
                dim=1,
            )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )

        logits = outputs.logits[:, self.target_token_position - 1]
        probabilities = self.softmax(logits)
        selected_token_ids = self.selected_interpretation_token_word_id.to(self.model.device)
        gather_indices = selected_token_ids.unsqueeze(0).unsqueeze(-1).expand(
            input_ids.shape[0], -1, -1
        )
        selected_probabilities = probabilities.gather(dim=2, index=gather_indices).squeeze(-1)
        return selected_probabilities[0] if single_input else selected_probabilities
