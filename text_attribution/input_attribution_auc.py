import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def _auc(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return 0.0
    if y.size == 1:
        return float(y[0])
    return float(np.trapezoid(y, x))


def _span_order(result, score_field="span_scores"):
    scores = result.get(score_field)
    if scores is None:
        raise ValueError(f"Result does not contain {score_field}.")
    spans = result.get("spans", [])
    if len(spans) != len(scores):
        raise ValueError(f"spans length {len(spans)} does not match {score_field} length {len(scores)}.")
    return sorted(range(len(spans)), key=lambda index: float(scores[index]), reverse=True)


def _explainable_positions(spans):
    positions = set()
    for span in spans:
        positions.update(int(token_index) for token_index in span.get("token_indices", []))
    return sorted(positions)


def _score_prompt_variant_probabilities_from_logits(qwen, batch_prompt_ids):
    input_ids = batch_prompt_ids.to(qwen.model.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=qwen.model.device)

    max_target_position = int(qwen.target_token_position.max().item())
    generated_prefix_len = max(0, max_target_position - qwen.prompt_length)
    if generated_prefix_len:
        generated_prefix = qwen.generated_answer_ids[:generated_prefix_len].to(
            qwen.model.device,
            dtype=torch.long,
        )
        prefix_batch = generated_prefix.unsqueeze(0).expand(input_ids.shape[0], -1)
        input_ids = torch.cat([input_ids, prefix_batch], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(prefix_batch, device=qwen.model.device)],
            dim=1,
        )

    with torch.no_grad():
        outputs = qwen.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
        )
    target_positions = qwen.target_token_position.to(qwen.model.device, dtype=torch.long) - 1
    logits = outputs.logits[:, target_positions]
    probabilities = torch.softmax(logits, dim=-1)
    selected_token_ids = qwen.selected_interpretation_token_word_id.to(qwen.model.device, dtype=torch.long)
    gather_indices = selected_token_ids.unsqueeze(0).unsqueeze(-1).expand(input_ids.shape[0], -1, -1)
    return probabilities.gather(dim=2, index=gather_indices).squeeze(-1).to(torch.float32)


def _score_prompt_variants(qwen, prompt_variants, batch_size, score_type="probability"):
    scores = []
    batch_size = batch_size or len(prompt_variants)
    for start in tqdm(range(0, len(prompt_variants), batch_size), desc="Scoring prompt variants", leave=False):
        batch = torch.stack(prompt_variants[start : start + batch_size], dim=0)
        if score_type == "probability":
            batch_scores = qwen(batch).to(torch.float32)
        elif score_type == "logit":
            batch_scores = _score_prompt_variant_probabilities_from_logits(qwen, batch)
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")
        batch_scores = batch_scores.detach().cpu().numpy()
        if batch_scores.ndim == 1:
            batch_scores = batch_scores[None, :]
        scores.append(batch_scores)
    return np.concatenate(scores, axis=0)


def _build_insertion_deletion_variants(prompt_ids, spans, ordered_span_indices, mask_token_id):
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    explainable_positions = _explainable_positions(spans)
    total_positions = max(len(explainable_positions), 1)

    insertion = prompt_ids.clone()
    deletion = prompt_ids.clone()
    if explainable_positions:
        insertion[torch.as_tensor(explainable_positions, dtype=torch.long)] = int(mask_token_id)

    insertion_variants = [insertion.clone()]
    deletion_variants = [deletion.clone()]
    region_areas = [0.0]
    inserted_positions = set()
    deleted_positions = set()

    for span_index in ordered_span_indices:
        token_indices = [int(token_index) for token_index in spans[span_index].get("token_indices", [])]
        for token_index in token_indices:
            insertion[token_index] = prompt_ids[token_index]
            deletion[token_index] = int(mask_token_id)
            inserted_positions.add(token_index)
            deleted_positions.add(token_index)
        insertion_variants.append(insertion.clone())
        deletion_variants.append(deletion.clone())
        region_areas.append(float(len(inserted_positions) / total_positions))

    return insertion_variants, deletion_variants, region_areas


def compute_input_attribution_auc(
    qwen,
    result,
    score_field="span_scores",
    batch_size=8,
    score_type="probability",
    target_token_limit=None,
    ordered_span_indices=None,
):
    prompt_input_ids = result.get("input_ids")
    generated_answer_ids = result.get("generated_answer_ids")
    spans = result.get("spans", [])
    if prompt_input_ids is None or generated_answer_ids is None:
        raise ValueError("Result must contain input_ids and generated_answer_ids.")
    if not spans:
        raise ValueError("Result must contain spans.")

    selected_token_indices = result.get("selected_output_token_indices")
    if selected_token_indices is None:
        selected_token_indices = result.get("selected_interpretation_token_id")
    if selected_token_indices is None:
        selected_token_indices = list(range(len(generated_answer_ids)))
    if target_token_limit is not None:
        selected_token_indices = selected_token_indices[: int(target_token_limit)]
    if not selected_token_indices:
        raise ValueError("No output tokens were selected for AUC evaluation.")

    qwen.prompt_length = len(prompt_input_ids)
    qwen.set_targets(generated_answer_ids, selected_token_indices)

    if ordered_span_indices is None:
        ordered_span_indices = _span_order(result, score_field=score_field)
    insertion_variants, deletion_variants, region_areas = _build_insertion_deletion_variants(
        prompt_input_ids,
        spans,
        ordered_span_indices,
        qwen.mask_token_id,
    )
    all_variants = insertion_variants + deletion_variants
    all_scores = _score_prompt_variants(
        qwen,
        all_variants,
        batch_size=batch_size,
        score_type=score_type,
    )
    split_at = len(insertion_variants)
    insertion_word_scores = all_scores[:split_at]
    deletion_word_scores = all_scores[split_at:]

    insertion_scores = insertion_word_scores.mean(axis=1)
    deletion_scores = deletion_word_scores.mean(axis=1)
    insertion_token_auc = [
        _auc(region_areas, insertion_word_scores[:, token_index])
        for token_index in range(insertion_word_scores.shape[1])
    ]
    deletion_token_auc = [
        _auc(region_areas, deletion_word_scores[:, token_index])
        for token_index in range(deletion_word_scores.shape[1])
    ]

    output_tokens = result.get("output_tokens", [])
    selected_tokens = [
        output_tokens[index] if index < len(output_tokens) else ""
        for index in selected_token_indices
    ]
    ordered_spans = []
    scores = result.get(score_field)
    for rank, span_index in enumerate(ordered_span_indices):
        span = spans[span_index]
        ordered_spans.append(
            {
                "rank": rank,
                "span_index": int(span_index),
                "role": span.get("role", ""),
                "text": span.get("text", ""),
                "score": float(scores[span_index]) if scores is not None else None,
                "token_indices": span.get("token_indices", []),
            }
        )

    return {
        "method": result.get("method", "ours" if "selected_region_indices" in result else "unknown"),
        "model_name": result.get("model_name", getattr(qwen, "model_name", "")),
        "score_field": score_field,
        "score_type": score_type,
        "score_value": "target_token_probability_after_full_vocab_softmax"
        if score_type == "logit"
        else "target_token_probability",
        "num_spans": len(spans),
        "num_prompt_tokens": len(prompt_input_ids),
        "num_output_tokens": len(selected_token_indices),
        "region_area": [float(value) for value in region_areas],
        "insertion_score": insertion_scores.astype(float).tolist(),
        "deletion_score": deletion_scores.astype(float).tolist(),
        "insertion_word_score": insertion_word_scores.astype(float).tolist(),
        "deletion_word_score": deletion_word_scores.astype(float).tolist(),
        "insertion_token_auc": [float(value) for value in insertion_token_auc],
        "deletion_token_auc": [float(value) for value in deletion_token_auc],
        "insertion_auc": _auc(region_areas, insertion_scores),
        "deletion_auc": _auc(region_areas, deletion_scores),
        "selected_output_token_indices": [int(index) for index in selected_token_indices],
        "selected_output_tokens": selected_tokens,
        "ordered_spans": ordered_spans,
        "auc_definition": (
            "Insertion starts from masked system/user prompt spans and restores spans in descending attribution order. "
            "Deletion starts from the full prompt and masks spans in the same order. "
            "Generated autoregressive prefixes remain visible, and each curve value is the target token "
            "probability after full-vocabulary softmax."
        ),
    }


def compute_existing_input_attribution_auc(result):
    region_area = result.get("region_area")
    insertion_score = result.get("insertion_score")
    deletion_score = result.get("deletion_score")
    if region_area is None or insertion_score is None or deletion_score is None:
        raise ValueError("Result must contain region_area, insertion_score, and deletion_score.")

    insertion_word_score = result.get("insertion_word_score")
    deletion_word_score = result.get("deletion_word_score")
    baseline_score = result.get("baseline_score", [deletion_score[-1]])
    original_score = result.get("org_score", [insertion_score[-1]])
    insertion_start = float(np.mean(np.asarray(baseline_score, dtype=np.float32)))
    deletion_start = float(np.mean(np.asarray(original_score, dtype=np.float32)))

    areas = [0.0] + [float(value) for value in region_area]
    insertion_scores = [insertion_start] + [float(value) for value in insertion_score]
    deletion_scores = [deletion_start] + [float(value) for value in deletion_score]

    insertion_token_auc = []
    deletion_token_auc = []
    if insertion_word_score and deletion_word_score:
        baseline_word = np.asarray(baseline_score, dtype=np.float32)
        original_word = np.asarray(original_score, dtype=np.float32)
        insertion_words = np.asarray([baseline_word.tolist()] + insertion_word_score, dtype=np.float32)
        deletion_words = np.asarray([original_word.tolist()] + deletion_word_score, dtype=np.float32)
        insertion_token_auc = [
            _auc(areas, insertion_words[:, token_index])
            for token_index in range(insertion_words.shape[1])
        ]
        deletion_token_auc = [
            _auc(areas, deletion_words[:, token_index])
            for token_index in range(deletion_words.shape[1])
        ]

    output_tokens = result.get("output_tokens", [])
    selected_token_indices = result.get("selected_interpretation_token_id")
    if selected_token_indices is None:
        selected_token_indices = result.get("selected_output_token_indices")
    if selected_token_indices is None:
        selected_token_indices = list(range(len(output_tokens)))
    selected_tokens = [
        output_tokens[index] if index < len(output_tokens) else ""
        for index in selected_token_indices
    ]

    return {
        "method": result.get("method", "ours"),
        "model_name": result.get("model_name", ""),
        "score_field": "selected_region_indices",
        "num_spans": len(result.get("spans", [])),
        "num_prompt_tokens": len(result.get("input_ids", [])),
        "num_output_tokens": len(selected_token_indices),
        "region_area": areas,
        "insertion_score": insertion_scores,
        "deletion_score": deletion_scores,
        "insertion_word_score": (
            [np.asarray(baseline_score, dtype=np.float32).astype(float).tolist()] + insertion_word_score
            if insertion_word_score
            else []
        ),
        "deletion_word_score": (
            [np.asarray(original_score, dtype=np.float32).astype(float).tolist()] + deletion_word_score
            if deletion_word_score
            else []
        ),
        "insertion_token_auc": [float(value) for value in insertion_token_auc],
        "deletion_token_auc": [float(value) for value in deletion_token_auc],
        "insertion_auc": _auc(areas, insertion_scores),
        "deletion_auc": _auc(areas, deletion_scores),
        "selected_output_token_indices": [int(index) for index in selected_token_indices],
        "selected_output_tokens": selected_tokens,
        "ordered_spans": [
            {
                "rank": rank,
                "span_index": int(span_index),
                "role": result.get("spans", [{}])[span_index].get("role", "")
                if span_index < len(result.get("spans", []))
                else "",
                "text": result.get("spans", [{}])[span_index].get("text", "")
                if span_index < len(result.get("spans", []))
                else "",
                "score": None,
                "token_indices": result.get("spans", [{}])[span_index].get("token_indices", [])
                if span_index < len(result.get("spans", []))
                else [],
            }
            for rank, span_index in enumerate(result.get("selected_region_indices", []))
        ],
        "auc_definition": (
            "AUC computed from saved insertion/deletion curves. The 0-area point is restored from "
            "baseline_score for insertion and org_score for deletion."
        ),
    }


def save_input_attribution_auc_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_input_attribution_auc_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "region_area", "insertion_score", "deletion_score"])
        for step, (area, insertion_score, deletion_score) in enumerate(
            zip(result["region_area"], result["insertion_score"], result["deletion_score"])
        ):
            writer.writerow([step, area, insertion_score, deletion_score])
    return save_path


def save_input_attribution_token_auc_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "output_token_index",
                "token",
                "insertion_auc",
                "deletion_auc",
            ]
        )
        for rank, token_index in enumerate(result["selected_output_token_indices"]):
            token = result["selected_output_tokens"][rank]
            writer.writerow(
                [
                    rank,
                    token_index,
                    token,
                    result["insertion_token_auc"][rank],
                    result["deletion_token_auc"][rank],
                ]
            )
    return save_path


def save_input_attribution_auc_summary(rows, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "result_path",
        "num_spans",
        "num_output_tokens",
        "insertion_auc",
        "deletion_auc",
        "insertion_start",
        "insertion_end",
        "deletion_start",
        "deletion_end",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return save_path
