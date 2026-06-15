import csv
import html
import json
from pathlib import Path

import numpy as np
import torch


def _auc(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return 0.0
    if len(y) == 1:
        return float(y[0])
    return float(np.trapezoid(y, x))


def _saliency_from_curve(scores, areas):
    if not scores:
        return 0.0
    values = np.asarray(scores, dtype=np.float32)
    return _auc(areas, values) - float(values.min())


def _span_dict(span):
    return span.to_dict() if hasattr(span, "to_dict") else span


def _prompt_regions(spans, role=None):
    regions = []
    for index, span in enumerate(spans):
        item = _span_dict(span)
        if role is not None and item.get("role") != role:
            continue
        token_indices = [int(token_index) for token_index in item.get("token_indices", [])]
        if not token_indices:
            continue
        regions.append(
            {
                "source": item.get("role", "prompt"),
                "label": item.get("text", ""),
                "token_indices": token_indices,
                "order": min(token_indices),
            }
        )
    regions.sort(key=lambda item: item["order"])
    return regions


def _generated_regions(prompt_length, output_tokens, target_token_index):
    regions = []
    for index in range(int(target_token_index)):
        regions.append(
            {
                "source": "generated_prefix",
                "label": output_tokens[index] if index < len(output_tokens) else str(index),
                "token_indices": [prompt_length + index],
                "order": prompt_length + index,
            }
        )
    return regions


def _mask_positions(ids, positions, mask_token_id):
    if positions:
        ids[torch.as_tensor(sorted(set(positions)), dtype=torch.long)] = int(mask_token_id)


def _score_batch(qwen, batch_ids, target_position, target_token_id):
    input_ids = torch.stack(batch_ids, dim=0).to(qwen.model.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=qwen.model.device)
    with torch.no_grad():
        outputs = qwen.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=True,
        )
        logits = outputs.logits[:, int(target_position) - 1]
        probs = torch.softmax(logits, dim=-1)
        selected = probs[:, int(target_token_id)]
    return selected.float().detach().cpu().numpy().astype(float).tolist()


def _source_curve(qwen, base_ids, baseline_ids, regions, target_position, target_token_id):
    if not regions:
        baseline_score = _score_batch(qwen, [baseline_ids], target_position, target_token_id)[0]
        return {
            "areas": [0.0],
            "scores": [baseline_score],
            "saliency": 0.0,
        }

    total_tokens = sum(len(region["token_indices"]) for region in regions)
    visible = baseline_ids.clone()
    variants = [visible.clone()]
    areas = [0.0]
    visible_count = 0
    for region in regions:
        for token_index in region["token_indices"]:
            visible[token_index] = base_ids[token_index]
        visible_count += len(region["token_indices"])
        variants.append(visible.clone())
        areas.append(float(visible_count / max(total_tokens, 1)))
    scores = _score_batch(qwen, variants, target_position, target_token_id)
    return {
        "areas": areas,
        "scores": scores,
        "saliency": _saliency_from_curve(scores, areas),
    }


def compute_output_source_auc(qwen, prompt_input_ids, generated_answer_ids, spans, output_tokens, selected_token_indices):
    prompt_ids = torch.as_tensor(prompt_input_ids, dtype=torch.long)
    answer_ids = torch.as_tensor(generated_answer_ids, dtype=torch.long)
    prompt_length = int(prompt_ids.numel())
    prompt_regions = _prompt_regions(spans)
    system_regions = _prompt_regions(spans, role="system")
    user_regions = _prompt_regions(spans, role="user")
    prompt_mask_positions = []
    for region in prompt_regions:
        prompt_mask_positions.extend(region["token_indices"])

    rows = []
    curves = []
    for token_index in selected_token_indices:
        token_index = int(token_index)
        prefix_ids = answer_ids[:token_index]
        base_ids = torch.cat([prompt_ids, prefix_ids], dim=0)
        baseline_ids = base_ids.clone()
        _mask_positions(baseline_ids, prompt_mask_positions, qwen.mask_token_id)
        generated_positions = list(range(prompt_length, prompt_length + token_index))
        _mask_positions(baseline_ids, generated_positions, qwen.mask_token_id)

        target_position = prompt_length + token_index
        target_token_id = int(answer_ids[token_index].item())
        full_prob = _score_batch(qwen, [base_ids], target_position, target_token_id)[0]
        generated_regions = _generated_regions(prompt_length, output_tokens, token_index)
        baseline_prob = _score_batch(qwen, [baseline_ids], target_position, target_token_id)[0]
        source_sets = {
            "system": system_regions,
            "user": user_regions,
            "generated_prefix": generated_regions,
        }
        token_curves = {}
        saliencies = {}
        for source, regions in source_sets.items():
            curve = _source_curve(qwen, base_ids, baseline_ids, regions, target_position, target_token_id)
            token_curves[source] = curve
            saliencies[source] = float(curve["saliency"])
        saliencies["prompt"] = saliencies["system"] + saliencies["user"]

        source_candidates = {
            "system": saliencies["system"],
            "user": saliencies["user"],
            "generated_prefix": saliencies["generated_prefix"],
        }
        stronger_source = max(source_candidates, key=source_candidates.get)
        if source_candidates[stronger_source] <= 1e-8:
            stronger_source = "none"
        rows.append(
            {
                "output_token_index": token_index,
                "token": output_tokens[token_index] if token_index < len(output_tokens) else "",
                "token_id": target_token_id,
                "full_prob": full_prob,
                "baseline_prob": float(baseline_prob),
                "system_auc_saliency": saliencies["system"],
                "user_auc_saliency": saliencies["user"],
                "prompt_auc_saliency": saliencies["prompt"],
                "generated_prefix_auc_saliency": saliencies["generated_prefix"],
                "stronger_source": stronger_source,
            }
        )
        curves.append(
            {
                "output_token_index": token_index,
                "token": output_tokens[token_index] if token_index < len(output_tokens) else "",
                "curves": token_curves,
            }
        )

    summary_values = {
        "system_auc_saliency": float(np.mean([row["system_auc_saliency"] for row in rows])) if rows else 0.0,
        "user_auc_saliency": float(np.mean([row["user_auc_saliency"] for row in rows])) if rows else 0.0,
        "prompt_auc_saliency": float(np.mean([row["prompt_auc_saliency"] for row in rows])) if rows else 0.0,
        "generated_prefix_auc_saliency": float(np.mean([row["generated_prefix_auc_saliency"] for row in rows])) if rows else 0.0,
    }
    compare = {
        "system": summary_values["system_auc_saliency"],
        "user": summary_values["user_auc_saliency"],
        "generated_prefix": summary_values["generated_prefix_auc_saliency"],
    }
    overall = max(compare, key=compare.get) if compare else "none"
    if compare and compare[overall] <= 1e-8:
        overall = "none"
    return {
        "summary": {
            **summary_values,
            "overall_stronger_source": overall,
            "method": "AUC over ordered insertion curves, matching visualization.visualization.get_word_saliency",
        },
        "rows": rows,
        "curves": curves,
    }


def save_output_source_auc_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_output_source_auc_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "output_token_index",
        "token",
        "token_id",
        "full_prob",
        "baseline_prob",
        "system_auc_saliency",
        "user_auc_saliency",
        "prompt_auc_saliency",
        "generated_prefix_auc_saliency",
        "stronger_source",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    return save_path


def save_output_source_auc_html(result, save_path):
    summary = result["summary"]
    row_html = []
    for row in result["rows"]:
        row_html.append(
            "<tr>"
            f"<td>{row['output_token_index']}</td>"
            f"<td><code>{html.escape(row['token'])}</code></td>"
            f"<td>{row['full_prob']:.4f}</td>"
            f"<td>{row['system_auc_saliency']:.4f}</td>"
            f"<td>{row['user_auc_saliency']:.4f}</td>"
            f"<td>{row['generated_prefix_auc_saliency']:.4f}</td>"
            f"<td>{html.escape(row['stronger_source'])}</td>"
            "</tr>"
        )
    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Output Source AUC</title>"
        "<style>body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        "table{width:100%;border-collapse:collapse;font-size:14px;}th,td{border-bottom:1px solid #e5e9f2;padding:8px;text-align:left;}th{color:#536176;}code{white-space:pre-wrap;}</style></head><body><main>"
        "<section><h1>Output Source AUC Attribution</h1>"
        f"<p>Overall stronger source: <strong>{html.escape(summary['overall_stronger_source'])}</strong></p>"
        f"<p>System mean AUC saliency: {summary['system_auc_saliency']:.4f}; User: {summary['user_auc_saliency']:.4f}; Generated prefix: {summary['generated_prefix_auc_saliency']:.4f}; Prompt all: {summary['prompt_auc_saliency']:.4f}</p>"
        "</section><section><table><thead><tr><th>#</th><th>Output token</th><th>Full prob</th><th>System AUC</th><th>User AUC</th><th>Generated-prefix AUC</th><th>Winner</th></tr></thead><tbody>"
        + "".join(row_html)
        + "</tbody></table></section></main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_output_source_auc_svg(result, save_path):
    rows = result["rows"]
    width = 1180
    margin = 32
    label_w = 230
    chart_w = 820
    row_h = 32
    height = margin * 2 + 74 + max(1, len(rows)) * row_h
    max_value = max(
        [row["system_auc_saliency"] for row in rows]
        + [row["user_auc_saliency"] for row in rows]
        + [row["generated_prefix_auc_saliency"] for row in rows]
        + [1e-8]
    )
    elements = []
    summary = result["summary"]
    elements.append(f'<text x="{margin}" y="{margin}" font-size="22" font-weight="700" fill="#172033">Output Source AUC Attribution</text>')
    elements.append(f'<text x="{margin}" y="{margin + 28}" font-size="14" fill="#536176">Overall: {html.escape(summary["overall_stronger_source"])} | system {summary["system_auc_saliency"]:.4f} | user {summary["user_auc_saliency"]:.4f} | generated prefix {summary["generated_prefix_auc_saliency"]:.4f}</text>')
    y = margin + 74
    for row in rows:
        label = html.escape(f"{row['output_token_index']}: {row['token']}")
        elements.append(f'<text x="{margin}" y="{y}" font-size="13" fill="#172033">{label}</text>')
        x0 = margin + label_w
        values = [
            ("system", row["system_auc_saliency"], "#2563eb"),
            ("user", row["user_auc_saliency"], "#16a34a"),
            ("generated", row["generated_prefix_auc_saliency"], "#dc2626"),
        ]
        for index, (_, value, color) in enumerate(values):
            bar_w = chart_w * float(value) / max_value
            bar_y = y - 18 + index * 8
            elements.append(f'<rect x="{x0}" y="{bar_y}" width="{bar_w}" height="6" fill="{color}" opacity="0.78"/>')
        y += row_h
    legend_y = height - 20
    elements.append(f'<text x="{margin + label_w}" y="{legend_y}" font-size="13" fill="#2563eb">blue system</text>')
    elements.append(f'<text x="{margin + label_w + 110}" y="{legend_y}" font-size="13" fill="#16a34a">green user</text>')
    elements.append(f'<text x="{margin + label_w + 220}" y="{legend_y}" font-size="13" fill="#dc2626">red generated prefix</text>')
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#f7f8fb"/>\n'
        f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" rx="8" fill="#ffffff" stroke="#dfe4ee"/>\n'
        + "".join(elements)
        + "\n</svg>\n"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(svg, encoding="utf-8")
    return save_path
