import csv
import html
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .text_report import input_color, normalize_scores, save_input_attribution_report_svg


def _as_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class AtManTextAttributor:
    """Text adaptation of AtMan-style attention manipulation.

    Each text span is suppressed by adding log(suppression_factor) to the
    attention logits of that span's key positions throughout the causal
    self-attention mask. Attribution is the drop in selected target-token
    autoregressive log-probability compared with the unsuppressed pass.
    """

    def __init__(self, model, tokenizer, suppression_factor=0.1, score_reduction="sum", batch_size=8):
        if not 0.0 < suppression_factor <= 1.0:
            raise ValueError("suppression_factor must be in (0, 1].")
        if score_reduction not in {"sum", "mean"}:
            raise ValueError("score_reduction must be 'sum' or 'mean'.")
        self.model = model
        self.tokenizer = tokenizer
        self.suppression_factor = float(suppression_factor)
        self.score_reduction = score_reduction
        self.batch_size = batch_size

    @property
    def device(self):
        return getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _model_base(self):
        return getattr(self.model, "model", self.model)

    def _layer_mask_mapping(self, attention_mask):
        base_model = self._model_base()
        layer_types = getattr(getattr(base_model, "config", None), "layer_types", None)
        if layer_types is None:
            return {"full_attention": attention_mask, "sliding_attention": attention_mask}
        return {layer_type: attention_mask for layer_type in set(layer_types)}

    def _make_attention_mask(self, seq_len, suppressed_token_indices=None, batch_size=1):
        dtype = self.model.get_input_embeddings().weight.dtype
        device = self.device
        min_value = torch.finfo(dtype).min
        causal = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

        if suppressed_token_indices is not None:
            penalty = float(np.log(self.suppression_factor))
            for batch_index, indices in enumerate(suppressed_token_indices):
                if len(indices) == 0:
                    continue
                causal[batch_index, :, :, indices] += penalty
        return causal

    def _score_logprobs(self, logits, prompt_len, generated_answer_ids, selected_output_token_indices):
        scores = []
        for output_index in selected_output_token_indices:
            target_position = prompt_len + int(output_index) - 1
            target_id = int(generated_answer_ids[int(output_index)].item())
            logprob = F.log_softmax(logits[:, target_position].float(), dim=-1)[:, target_id]
            scores.append(logprob)
        scores = torch.stack(scores, dim=-1)
        if self.score_reduction == "mean":
            return scores.mean(dim=-1), scores
        return scores.sum(dim=-1), scores

    def _forward_scores(self, input_ids, attention_mask_mapping, prompt_len, generated_answer_ids, selected_output_token_indices):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask_mapping,
            return_dict=True,
            use_cache=False,
        )
        return self._score_logprobs(
            outputs.logits,
            prompt_len=prompt_len,
            generated_answer_ids=generated_answer_ids,
            selected_output_token_indices=selected_output_token_indices,
        )

    def attribute(self, input_ids, generated_answer_ids, spans, selected_output_token_indices=None):
        model = self.model
        model.eval()
        device = self.device

        prompt_ids = torch.as_tensor(input_ids, device=device, dtype=torch.long)
        generated_answer_ids = torch.as_tensor(generated_answer_ids, device=device, dtype=torch.long)
        if selected_output_token_indices is None:
            selected_output_token_indices = list(range(int(generated_answer_ids.numel())))
        selected_output_token_indices = [int(index) for index in selected_output_token_indices]
        if not selected_output_token_indices:
            raise ValueError("No generated tokens were selected for AtMan attribution.")

        max_selected = max(selected_output_token_indices)
        prefix_ids = generated_answer_ids[:max_selected]
        full_input_ids = torch.cat([prompt_ids, prefix_ids], dim=0)
        seq_len = int(full_input_ids.numel())
        prompt_len = int(prompt_ids.numel())

        with torch.no_grad():
            original_mask = self._layer_mask_mapping(self._make_attention_mask(seq_len, batch_size=1))
            original_score, original_token_logprobs = self._forward_scores(
                full_input_ids.unsqueeze(0),
                original_mask,
                prompt_len=prompt_len,
                generated_answer_ids=generated_answer_ids,
                selected_output_token_indices=selected_output_token_indices,
            )
            original_score_value = float(original_score[0].detach().cpu().item())

            span_scores = []
            suppressed_scores = []
            suppressed_token_logprobs = []
            for start in range(0, len(spans), self.batch_size):
                chunk = spans[start : start + self.batch_size]
                batch_indices = [list(span.token_indices) for span in chunk]
                batch_size = len(batch_indices)
                batch_input_ids = full_input_ids.unsqueeze(0).expand(batch_size, -1)
                batch_mask = self._layer_mask_mapping(
                    self._make_attention_mask(seq_len, batch_indices, batch_size=batch_size)
                )
                batch_scores, batch_token_logprobs = self._forward_scores(
                    batch_input_ids,
                    batch_mask,
                    prompt_len=prompt_len,
                    generated_answer_ids=generated_answer_ids,
                    selected_output_token_indices=selected_output_token_indices,
                )
                deltas = original_score - batch_scores
                span_scores.extend(deltas.detach().cpu().float().tolist())
                suppressed_scores.extend(batch_scores.detach().cpu().float().tolist())
                suppressed_token_logprobs.extend(batch_token_logprobs.detach().cpu().float().tolist())

        span_scores = np.asarray(span_scores, dtype=np.float32)
        span_scores_normalized = normalize_scores(span_scores)
        return {
            "method": "atman",
            "suppression_factor": self.suppression_factor,
            "score_reduction": self.score_reduction,
            "batch_size": self.batch_size,
            "selected_output_token_indices": selected_output_token_indices,
            "original_score": original_score_value,
            "original_token_logprobs": original_token_logprobs[0].detach().cpu().float().tolist(),
            "suppressed_scores": suppressed_scores,
            "suppressed_token_logprobs": suppressed_token_logprobs,
            "span_scores": span_scores.astype(float).tolist(),
            "span_scores_normalized": span_scores_normalized.astype(float).tolist(),
        }


def build_atman_result(
    attribution,
    model_name,
    system_prompt,
    user_prompt,
    rendered_prompt,
    input_granularity,
    input_ids,
    spans,
    generation,
):
    result = dict(attribution)
    result.update(
        {
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "rendered_prompt": rendered_prompt,
            "input_granularity": input_granularity,
            "input_ids": _as_list(input_ids),
            "spans": [span.to_dict() for span in spans],
            "generated_answer_ids": _as_list(generation["generated_answer_ids"]),
            "output_text": generation["output_text"],
            "output_tokens": generation["output_tokens"],
        }
    )
    return result


def save_atman_attribution_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_atman_span_scores_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["span_index", "role", "text", "atman_score", "normalized_atman_score", "token_indices"]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, span in enumerate(result["spans"]):
            writer.writerow(
                {
                    "span_index": index,
                    "role": span["role"],
                    "text": span["text"],
                    "atman_score": result["span_scores"][index],
                    "normalized_atman_score": result["span_scores_normalized"][index],
                    "token_indices": span["token_indices"],
                }
            )
    return save_path


def save_atman_text_attribution_html(result, save_path):
    span_items = []
    for index, span in enumerate(result["spans"]):
        score = float(result["span_scores_normalized"][index])
        raw_score = float(result["span_scores"][index])
        title = html.escape(f"{span['role']} span {index}; AtMan drop {raw_score:.6f}; normalized {score:.3f}")
        text = html.escape(span["text"] if span["text"] else " ")
        span_items.append(
            f'<span class="region" title="{title}" style="background:{input_color(score)}">{text}</span>'
        )

    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>AtMan Text Attribution</title>"
        "<style>"
        "body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}"
        "section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        ".region{display:inline-block;border:1px solid rgba(23,32,51,.14);border-radius:5px;margin:2px;padding:3px 6px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;}"
        ".legend{height:14px;width:380px;border-radius:7px;background:linear-gradient(90deg,rgb(255,255,255),rgb(234,88,12));display:inline-block;vertical-align:middle;margin:0 12px;border:1px solid rgba(23,32,51,.14);}"
        "pre{white-space:pre-wrap;line-height:1.55;}"
        "</style></head><body><main>"
        f"<section><h1>AtMan Text Attribution</h1><p>Model: {html.escape(result['model_name'])}</p>"
        f"<p>Score: original selected-token logprob minus logprob after suppressing each input span in attention; suppression factor {result['suppression_factor']}.</p></section>"
        "<section><h2>System Prompt</h2><pre>" + html.escape(result.get("system_prompt", "")) + "</pre></section>"
        "<section><h2>User Prompt</h2><pre>" + html.escape(result.get("user_prompt", "")) + "</pre></section>"
        "<section><h2>Input AtMan Attribution</h2><div>" + " ".join(span_items) + "</div>"
        "<p>lower input attribution <span class='legend'></span> higher input attribution</p></section>"
        "<section><h2>Generated Output</h2><pre>" + html.escape(result.get("output_text", "")) + "</pre></section>"
        "</main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_atman_text_saliency_svg(result, save_path):
    subtitle = (
        "AtMan baseline: input-level attribution from selected-token logprob drop "
        f"after attention suppression with factor {result['suppression_factor']}."
    )
    return save_input_attribution_report_svg(result, save_path, subtitle=subtitle)
