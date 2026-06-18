import csv
import html
import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .text_report import input_color, normalize_scores, save_input_attribution_report_svg


def _as_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_tokenshap(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    shifted = values - values.min()
    total = float(shifted.sum())
    if total > 1e-8:
        return shifted / total
    return normalize_scores(values)


class TokenSHAPTextAttributor:
    """TokenSHAP-style black-box Shapley attribution over text spans.

    This follows the original TokenSHAP setup: generate a baseline response for
    the full prompt, generate responses for sampled coalitions of input spans,
    score each coalition by text similarity to the baseline response, and
    estimate each span's contribution from average similarity with vs without
    that span.
    """

    def __init__(
        self,
        qwen_adaptor,
        messages,
        spans,
        sampling_ratio=0.0,
        max_combinations=1000,
        random_seed=0,
        max_new_tokens=96,
    ):
        self.qwen_adaptor = qwen_adaptor
        self.messages = messages
        self.spans = spans
        self.sampling_ratio = float(sampling_ratio)
        self.max_combinations = max_combinations
        self.random_seed = int(random_seed)
        self.max_new_tokens = int(max_new_tokens)

    def _render_and_generate(self, messages):
        rendered_text = self.qwen_adaptor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        generation = self.qwen_adaptor.generate_from_rendered_chat(
            rendered_text,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return generation["output_text"]

    def _full_response(self):
        return self._render_and_generate(self.messages)

    def _coalition_messages(self, coalition):
        coalition = set(coalition)
        role_parts = {"system": [], "user": []}
        for index, span in enumerate(self.spans):
            if index in coalition and span.role in role_parts:
                role_parts[span.role].append(span.text)

        messages = []
        for original_message in self.messages:
            role = original_message.get("role", "")
            if role in role_parts:
                content = " ".join(part for part in role_parts[role] if part).strip()
                messages.append({"role": role, "content": content})
            else:
                messages.append(original_message)
        return messages

    def _generate_random_combinations(self, span_count, count, excluded):
        rng = random.Random(self.random_seed)
        sampled = []
        sampled_set = set()
        max_attempts = max(100, count * 10)
        attempts = 0
        while len(sampled) < count and attempts < max_attempts:
            attempts += 1
            if span_count <= 1:
                break
            bits = rng.randint(1, 2**span_count - 2)
            indexes = tuple(i for i in range(span_count) if bits & (1 << i))
            if indexes in excluded or indexes in sampled_set:
                continue
            sampled.append(indexes)
            sampled_set.add(indexes)
        return sampled

    def _sample_combinations(self, span_count):
        essential = []
        essential_set = set()
        for missing_index in range(span_count):
            indexes = tuple(i for i in range(span_count) if i != missing_index)
            essential.append(indexes)
            essential_set.add(indexes)

        if span_count == 1:
            full = (0,)
            if full not in essential_set:
                essential.append(full)
            return essential

        num_essential = len(essential)
        max_combinations = self.max_combinations
        if max_combinations is not None and max_combinations < num_essential:
            max_combinations = num_essential

        remaining_budget = float("inf") if max_combinations is None else max(0, max_combinations - num_essential)
        if self.sampling_ratio < 1.0:
            theoretical_total = 2**span_count - 1
            theoretical_additional = max(0, theoretical_total - num_essential)
            desired_additional = int(theoretical_additional * self.sampling_ratio)
            additional_count = min(desired_additional, remaining_budget)
        else:
            additional_count = remaining_budget

        if additional_count == float("inf"):
            additional_count = max(0, 2**span_count - 1 - num_essential)
        additional = self._generate_random_combinations(span_count, int(additional_count), essential_set)
        return essential + additional

    def _similarities(self, baseline_response, responses):
        texts = [baseline_response] + responses
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        return cosine_similarity(base_vector, comparison_vectors).flatten()

    def attribute(self):
        span_count = len(self.spans)
        if span_count == 0:
            raise ValueError("No text spans were found for TokenSHAP.")

        baseline_response = self._full_response()
        combinations = self._sample_combinations(span_count)
        responses = []
        rows = []
        for indexes in tqdm(combinations, desc="TokenSHAP combinations"):
            response = self._render_and_generate(self._coalition_messages(indexes))
            responses.append(response)
            rows.append({"indexes": list(indexes), "response": response})

        similarities = self._similarities(baseline_response, responses)
        for row, similarity in zip(rows, similarities):
            row["similarity"] = float(similarity)

        shapley_scores = []
        for span_index in range(span_count):
            with_values = [row["similarity"] for row in rows if span_index in row["indexes"]]
            without_values = [row["similarity"] for row in rows if span_index not in row["indexes"]]
            if not with_values:
                with_values = [1.0]
            if not without_values:
                without_values = [0.0]
            shapley_scores.append(float(np.mean(with_values) - np.mean(without_values)))

        shapley_scores = np.asarray(shapley_scores, dtype=np.float32)
        normalized_scores = _normalize_tokenshap(shapley_scores)
        return {
            "method": "tokenshap",
            "value_function": "response_similarity_to_full_prompt_generation",
            "sampling_ratio": self.sampling_ratio,
            "max_combinations": self.max_combinations,
            "random_seed": self.random_seed,
            "baseline_response": baseline_response,
            "combination_rows": rows,
            "span_scores": shapley_scores.astype(float).tolist(),
            "span_scores_normalized": normalized_scores.astype(float).tolist(),
        }


def build_tokenshap_result(
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


def save_tokenshap_attribution_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_tokenshap_span_scores_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["span_index", "role", "text", "tokenshap_score", "normalized_tokenshap_score", "token_indices"]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for index, span in enumerate(result["spans"]):
            writer.writerow(
                {
                    "span_index": index,
                    "role": span["role"],
                    "text": span["text"],
                    "tokenshap_score": result["span_scores"][index],
                    "normalized_tokenshap_score": result["span_scores_normalized"][index],
                    "token_indices": span["token_indices"],
                }
            )
    return save_path


def save_tokenshap_text_attribution_html(result, save_path):
    span_items = []
    for index, span in enumerate(result["spans"]):
        score = float(result["span_scores_normalized"][index])
        raw_score = float(result["span_scores"][index])
        title = html.escape(f"{span['role']} span {index}; TokenSHAP {raw_score:.6f}; normalized {score:.3f}")
        text = html.escape(span["text"] if span["text"] else " ")
        span_items.append(
            f'<span class="region" title="{title}" style="background:{input_color(score)}">{text}</span>'
        )

    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>TokenSHAP Text Attribution</title>"
        "<style>"
        "body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}"
        "section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        ".region{display:inline-block;border:1px solid rgba(23,32,51,.14);border-radius:5px;margin:2px;padding:3px 6px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;}"
        ".legend{height:14px;width:380px;border-radius:7px;background:linear-gradient(90deg,rgb(255,255,255),rgb(234,88,12));display:inline-block;vertical-align:middle;margin:0 12px;border:1px solid rgba(23,32,51,.14);}"
        "pre{white-space:pre-wrap;line-height:1.55;}"
        "</style></head><body><main>"
        f"<section><h1>TokenSHAP Text Attribution</h1><p>Model: {html.escape(result['model_name'])}</p>"
        "<p>Black-box TokenSHAP: generate responses for sampled input-span coalitions and score them by similarity to the full-prompt response.</p></section>"
        "<section><h2>System Prompt</h2><pre>" + html.escape(result.get("system_prompt", "")) + "</pre></section>"
        "<section><h2>User Prompt</h2><pre>" + html.escape(result.get("user_prompt", "")) + "</pre></section>"
        "<section><h2>Input TokenSHAP Attribution</h2><div>" + " ".join(span_items) + "</div>"
        "<p>lower input attribution <span class='legend'></span> higher input attribution</p></section>"
        "<section><h2>Generated Output</h2><pre>" + html.escape(result.get("output_text", "")) + "</pre></section>"
        "</main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_tokenshap_text_saliency_svg(result, save_path):
    subtitle = (
        "TokenSHAP baseline: black-box Shapley over system/user input spans; "
        "value is response similarity to full-prompt generation."
    )
    return save_input_attribution_report_svg(result, save_path, subtitle=subtitle)
