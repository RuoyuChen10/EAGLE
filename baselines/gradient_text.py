import csv
import html
import json
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _as_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span > 1e-8:
        return (values - values.min()) / span
    if float(np.abs(values).max()) > 1e-8:
        return np.ones_like(values)
    return np.zeros_like(values)


def _color(score):
    score = float(np.clip(score, 0.0, 1.0))
    r = int(255 * (1 - score) + 234 * score)
    g = int(255 * (1 - score) + 88 * score)
    b = int(255 * (1 - score) + 12 * score)
    return f"rgb({r},{g},{b})"


REPORT_WIDTH = 1320
REPORT_CARD_X = 16
REPORT_CONTENT_X = 34
REPORT_CONTENT_RIGHT = REPORT_WIDTH - 46
REPORT_LINE_H = 23
REPORT_CHIP_H = 27
REPORT_CHIP_GAP_X = 7
REPORT_CHIP_GAP_Y = 7


def _plain_text(value):
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def _wrap_lines(text, width=146):
    lines = []
    for raw_line in _plain_text(text).split("\n"):
        if not raw_line:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def _svg_text_block(elements, title, text, y, max_lines=None):
    elements.append(
        f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">{html.escape(title)}</text>'
    )
    y += 28
    lines = _wrap_lines(text)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    for line in lines:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="15" fill="#172033">{html.escape(line)}</text>'
        )
        y += REPORT_LINE_H
    return y + 38


def _chip_label(text):
    label = _plain_text(text).replace("\n", "\\n").replace("\t", "\\t")
    return label if label else " "


def _chip_width(label):
    return max(30, min(360, 18 + len(label) * 8.12))


def _svg_chip_flow(elements, items, y, title_prefix):
    x = REPORT_CONTENT_X
    for item in items:
        label = _chip_label(item["label"])
        width = _chip_width(label)
        if x + width > REPORT_CONTENT_RIGHT:
            x = REPORT_CONTENT_X
            y += REPORT_CHIP_H + REPORT_CHIP_GAP_Y
        title = html.escape(f"{title_prefix} input attribution score {item['raw_score']:.4f}")
        fill = _color(item["norm_score"])
        elements.append(
            f'<g><title>{title}</title>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{REPORT_CHIP_H}" rx="5" fill="{fill}" stroke="rgba(23,32,51,0.14)"/>'
            f'<text x="{x + 9}" y="{y + 18}" font-size="14" fill="#172033" xml:space="preserve">{html.escape(label[:42])}</text>'
            "</g>"
        )
        x += width + REPORT_CHIP_GAP_X
    return y + REPORT_CHIP_H + 44


def _gradient_span_items(result, role):
    items = []
    for index, span in enumerate(result["spans"]):
        if span.get("role") != role:
            continue
        items.append(
            {
                "label": span.get("text", ""),
                "raw_score": float(result["span_scores"][index]),
                "norm_score": float(result["span_scores_normalized"][index]),
            }
        )
    return items


def _svg_input_legend(elements, y):
    elements.append(
        f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="16" font-weight="700" fill="#172033">Input attribution bar</text>'
    )
    elements.append(
        '<linearGradient id="inputAttribution" x1="0%" y1="0%" x2="100%">'
        '<stop offset="0%" stop-color="rgb(255,255,255)"/>'
        '<stop offset="100%" stop-color="rgb(234,88,12)"/>'
        "</linearGradient>"
    )
    elements.append(
        f'<text x="{REPORT_CONTENT_X + 180}" y="{y + 22}" font-size="13" fill="#536176">lower input attribution</text>'
    )
    elements.append(
        f'<rect x="{REPORT_CONTENT_X + 180}" y="{y - 13}" width="380" height="14" rx="7" fill="url(#inputAttribution)" stroke="rgba(23,32,51,0.14)"/>'
    )
    elements.append(
        f'<text x="{REPORT_CONTENT_X + 578}" y="{y + 22}" font-size="13" fill="#536176">higher input attribution</text>'
    )
    return y + 54


class GradientTextAttributor:
    """Gradient baseline over pure text LLM prompts.

    For each selected generated token m, this computes the gradient of either
    the target token logit or log-probability under the autoregressive prefix
    prompt + generated[:m].  Token importance is the embedding-gradient norm
    for each original prompt token, summed over selected target tokens.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_score="logit",
        multiply_by_inputs=False,
        span_reduction="sum",
        method_name=None,
    ):
        if target_score not in {"logit", "logprob"}:
            raise ValueError("target_score must be 'logit' or 'logprob'.")
        if span_reduction not in {"sum", "mean", "max"}:
            raise ValueError("span_reduction must be 'sum', 'mean', or 'max'.")
        self.model = model
        self.tokenizer = tokenizer
        self.target_score = target_score
        self.multiply_by_inputs = multiply_by_inputs
        self.span_reduction = span_reduction
        if method_name is None:
            method_name = "input_x_gradient" if multiply_by_inputs else "gradient"
        self.method_name = method_name

    @property
    def device(self):
        return getattr(self.model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _target_objective(self, logits, target_token_id):
        target_logit = logits[..., int(target_token_id)]
        if self.target_score == "logit":
            return target_logit
        return F.log_softmax(logits.float(), dim=-1)[..., int(target_token_id)]

    def _reduce_span(self, token_scores, token_indices):
        scores = token_scores[token_indices]
        if scores.size == 0:
            return 0.0
        if self.span_reduction == "mean":
            return float(scores.mean())
        if self.span_reduction == "max":
            return float(scores.max())
        return float(scores.sum())

    def attribute(self, input_ids, generated_answer_ids, spans, selected_output_token_indices=None):
        model = self.model
        model.eval()
        embedding_layer = model.get_input_embeddings()
        device = self.device
        input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.long)
        generated_answer_ids = torch.as_tensor(generated_answer_ids, device=device, dtype=torch.long)

        if selected_output_token_indices is None:
            selected_output_token_indices = list(range(int(generated_answer_ids.numel())))
        selected_output_token_indices = [int(index) for index in selected_output_token_indices]
        if not selected_output_token_indices:
            raise ValueError("No generated tokens were selected for gradient attribution.")

        original_requires_grad = [param.requires_grad for param in model.parameters()]
        try:
            for param in model.parameters():
                param.requires_grad_(False)

            prompt_embeds_base = embedding_layer(input_ids.unsqueeze(0)).detach()
            per_output_token_scores = []
            objective_values = []

            for output_index in selected_output_token_indices:
                if output_index < 0 or output_index >= int(generated_answer_ids.numel()):
                    raise ValueError(f"Selected output token index {output_index} is out of range.")

                prompt_embeds = prompt_embeds_base.detach().clone().requires_grad_(True)
                prefix_ids = generated_answer_ids[:output_index]
                if int(prefix_ids.numel()) > 0:
                    prefix_embeds = embedding_layer(prefix_ids.unsqueeze(0)).detach()
                    inputs_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)
                else:
                    inputs_embeds = prompt_embeds

                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                )
                target_position = input_ids.numel() + output_index - 1
                target_id = int(generated_answer_ids[output_index].item())
                objective = self._target_objective(outputs.logits[0, target_position], target_id)

                model.zero_grad(set_to_none=True)
                if prompt_embeds.grad is not None:
                    prompt_embeds.grad.zero_()
                objective.backward()

                gradient = prompt_embeds.grad.detach()[0]
                if self.multiply_by_inputs:
                    gradient = gradient * prompt_embeds.detach()[0]
                token_scores = gradient.float().norm(p=2, dim=-1).cpu().numpy()
                per_output_token_scores.append(token_scores)
                objective_values.append(float(objective.detach().float().cpu().item()))

            per_output_token_scores = np.stack(per_output_token_scores, axis=0)
            input_token_scores = per_output_token_scores.sum(axis=0)
            input_token_scores_normalized = _normalize(input_token_scores)

            span_scores = np.asarray(
                [
                    self._reduce_span(input_token_scores, np.asarray(span.token_indices, dtype=np.int64))
                    for span in spans
                ],
                dtype=np.float32,
            )
            span_scores_normalized = _normalize(span_scores)

            return {
                "method": self.method_name,
                "target_score": self.target_score,
                "multiply_by_inputs": self.multiply_by_inputs,
                "span_reduction": self.span_reduction,
                "selected_output_token_indices": selected_output_token_indices,
                "objective_values": objective_values,
                "input_token_scores": input_token_scores.astype(float).tolist(),
                "input_token_scores_normalized": input_token_scores_normalized.astype(float).tolist(),
                "per_output_token_input_scores": per_output_token_scores.astype(float).tolist(),
                "span_scores": span_scores.astype(float).tolist(),
                "span_scores_normalized": span_scores_normalized.astype(float).tolist(),
            }
        finally:
            for param, requires_grad in zip(model.parameters(), original_requires_grad):
                param.requires_grad_(requires_grad)


class InputXGradientTextAttributor(GradientTextAttributor):
    """Input x Gradient baseline over prompt embeddings."""

    def __init__(self, model, tokenizer, target_score="logit", span_reduction="sum"):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            target_score=target_score,
            multiply_by_inputs=True,
            span_reduction=span_reduction,
            method_name="input_x_gradient",
        )


def _method_title(result):
    if result.get("method") == "input_x_gradient" or result.get("multiply_by_inputs"):
        return "Input x Gradient Text Attribution"
    return "Gradient Text Attribution"


def _method_description(result):
    target_score = html.escape(result["target_score"])
    if result.get("method") == "input_x_gradient" or result.get("multiply_by_inputs"):
        return (
            f"Objective: sum of autoregressive target-token {target_score} values; "
            "score: L2 norm of input embedding multiplied by its gradient."
        )
    return (
        f"Objective: sum of autoregressive target-token {target_score} values; "
        "score: embedding-gradient L2 norm."
    )


def _svg_method_description(result):
    target_score = html.escape(result["target_score"])
    if result.get("method") == "input_x_gradient" or result.get("multiply_by_inputs"):
        return (
            f"Input x Gradient baseline: input-level attribution over system/user prompt from "
            f"embedding times autoregressive target-token {target_score} gradient."
        )
    return (
        f"Gradient baseline: input-level attribution over system/user prompt from "
        f"autoregressive target-token {target_score} gradients."
    )


def save_gradient_attribution_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_gradient_span_scores_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["span_index", "role", "text", "gradient_score", "normalized_gradient_score", "token_indices"]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, span in enumerate(result["spans"]):
            writer.writerow(
                {
                    "span_index": index,
                    "role": span["role"],
                    "text": span["text"],
                    "gradient_score": result["span_scores"][index],
                    "normalized_gradient_score": result["span_scores_normalized"][index],
                    "token_indices": span["token_indices"],
                }
            )
    return save_path


def save_gradient_text_attribution_html(result, save_path):
    method_title = _method_title(result)
    method_description = _method_description(result)
    span_items = []
    for index, span in enumerate(result["spans"]):
        score = float(result["span_scores_normalized"][index])
        raw_score = float(result["span_scores"][index])
        title = html.escape(f"{span['role']} span {index}; gradient {raw_score:.6f}; normalized {score:.3f}")
        text = html.escape(span["text"] if span["text"] else " ")
        span_items.append(
            f'<span class="region" title="{title}" style="background:{_color(score)}">{text}</span>'
        )

    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(method_title)}</title>"
        "<style>"
        "body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}"
        "section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        ".region{display:inline-block;border:1px solid rgba(23,32,51,.14);border-radius:5px;margin:2px;padding:3px 6px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;}"
        ".legend{height:14px;width:380px;border-radius:7px;background:linear-gradient(90deg,rgb(255,255,255),rgb(234,88,12));display:inline-block;vertical-align:middle;margin:0 12px;border:1px solid rgba(23,32,51,.14);}"
        "pre{white-space:pre-wrap;line-height:1.55;}"
        "</style></head><body><main>"
        f"<section><h1>{html.escape(method_title)}</h1><p>Model: {html.escape(result['model_name'])}</p>"
        f"<p>{method_description}</p></section>"
        "<section><h2>System Prompt</h2><pre>" + html.escape(result.get("system_prompt", "")) + "</pre></section>"
        "<section><h2>User Prompt</h2><pre>" + html.escape(result.get("user_prompt", "")) + "</pre></section>"
        "<section><h2>Input Gradient Attribution</h2><div>" + " ".join(span_items) + "</div>"
        "<p>lower input attribution <span class='legend'></span> higher input attribution</p></section>"
        "<section><h2>Generated Output</h2><pre>" + html.escape(result.get("output_text", "")) + "</pre></section>"
        "</main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_gradient_text_saliency_svg(result, save_path):
    method_description = _svg_method_description(result)
    elements = [
        f'<text x="{REPORT_CONTENT_X}" y="34" font-size="26" font-weight="750" fill="#172033">Text Attribution Report</text>',
        f'<text x="{REPORT_CONTENT_X}" y="62" font-size="14" fill="#536176">{method_description}</text>',
    ]

    y = 96
    y = _svg_text_block(elements, "System Prompt", result.get("system_prompt", ""), y)
    y = _svg_text_block(elements, "User Prompt", result.get("user_prompt", ""), y)
    y = _svg_text_block(elements, "Generated Output", result.get("output_text", ""), y, max_lines=18)

    system_items = _gradient_span_items(result, "system")
    if system_items:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: System Prompt</text>'
        )
        y = _svg_chip_flow(elements, system_items, y + 9, "system")

    user_items = _gradient_span_items(result, "user")
    if user_items:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: User Prompt</text>'
        )
        y = _svg_chip_flow(elements, user_items, y + 9, "user")

    if system_items or user_items:
        y = _svg_input_legend(elements, y + 12)

    height = int(y + 34)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{REPORT_WIDTH}" height="{height}" viewBox="0 0 {REPORT_WIDTH} {height}">\n'
        '<rect width="100%" height="100%" fill="#f7f8fb"/>\n'
        f'<rect x="{REPORT_CARD_X}" y="16" width="{REPORT_WIDTH - 32}" height="{height - 32}" rx="8" fill="#ffffff" stroke="#dfe4ee"/>\n'
        + "".join(elements)
        + "\n</svg>\n"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(svg, encoding="utf-8")
    return save_path


def build_gradient_result(
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
