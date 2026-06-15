import csv
import html
import json
from pathlib import Path

import numpy as np


def _auc(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return 0.0
    if len(y) == 1:
        return float(y[0])
    return float(np.trapezoid(y, x))


def compute_output_token_input_influence(saved_json_file):
    insertion_word_scores = [saved_json_file["baseline_score"]] + saved_json_file["insertion_word_score"]
    insertion_word_scores = np.asarray(insertion_word_scores, dtype=np.float32).T
    regions = np.asarray([0.0] + saved_json_file["region_area"], dtype=np.float32)
    output_tokens = saved_json_file.get("output_tokens", [])
    output_token_ids = saved_json_file.get("generated_answer_ids", [])

    rows = []
    for token_index, insertion_curve in enumerate(insertion_word_scores):
        auc_value = _auc(regions, insertion_curve)
        curve_min = float(insertion_curve.min())
        input_influence = float(auc_value - curve_min)
        rows.append(
            {
                "output_token_index": token_index,
                "token": output_tokens[token_index] if token_index < len(output_tokens) else "",
                "token_id": output_token_ids[token_index] if token_index < len(output_token_ids) else None,
                "input_influence_score": input_influence,
                "insertion_auc": float(auc_value),
                "curve_min": curve_min,
                "curve_max": float(insertion_curve.max()),
                "baseline_score": float(insertion_curve[0]),
                "final_insertion_score": float(insertion_curve[-1]),
                "interpretation": "input_prompt" if input_influence > 0 else "generated_prefix_or_prior",
                "insertion_curve": insertion_curve.astype(float).tolist(),
            }
        )

    scores = np.asarray([row["input_influence_score"] for row in rows], dtype=np.float32)
    if len(scores) and float(scores.max() - scores.min()) > 1e-8:
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized = np.zeros_like(scores)
    for row, score in zip(rows, normalized):
        row["normalized_input_influence"] = float(score)
        row["relative_source"] = "input_prompt" if score >= 0.5 else "generated_prefix_or_prior"

    summary = {
        "method": "Same as visualization.visualization.get_word_saliency: AUC([baseline]+insertion_word_score) - curve_min for each output token.",
        "token_count": len(rows),
        "mean_input_influence_score": float(scores.mean()) if len(scores) else 0.0,
        "max_input_influence_score": float(scores.max()) if len(scores) else 0.0,
        "min_input_influence_score": float(scores.min()) if len(scores) else 0.0,
    }
    return {"summary": summary, "regions": regions.astype(float).tolist(), "rows": rows}


def save_output_token_input_influence_json(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path


def save_output_token_input_influence_csv(result, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "output_token_index",
        "token",
        "token_id",
        "input_influence_score",
        "normalized_input_influence",
        "insertion_auc",
        "curve_min",
        "curve_max",
        "baseline_score",
        "final_insertion_score",
        "relative_source",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({field: row.get(field) for field in fields})
    return save_path


def _color(score):
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.5:
        ratio = score / 0.5
        r = int(220 * ratio + 71 * (1 - ratio))
        g = int(235 * ratio + 85 * (1 - ratio))
        b = int(255 * ratio + 105 * (1 - ratio))
    else:
        ratio = (score - 0.5) / 0.5
        r = int(220 * (1 - ratio) + 185 * ratio)
        g = int(235 * (1 - ratio) + 28 * ratio)
        b = int(255 * (1 - ratio) + 28 * ratio)
    return f"rgb({r},{g},{b})"


def save_output_token_input_influence_html(result, save_path):
    token_spans = []
    for row in result["rows"]:
        token = html.escape(row["token"] if row["token"] else " ")
        color = _color(row["normalized_input_influence"])
        title = html.escape(
            f"token {row['output_token_index']} input influence {row['input_influence_score']:.6f}"
        )
        token_spans.append(
            f'<span class="token" style="background:{color}" title="{title}">{token}</span>'
        )

    table_rows = []
    for row in result["rows"]:
        table_rows.append(
            "<tr>"
            f"<td>{row['output_token_index']}</td>"
            f"<td><code>{html.escape(row['token'])}</code></td>"
            f"<td>{row['input_influence_score']:.6f}</td>"
            f"<td>{row['normalized_input_influence']:.3f}</td>"
            f"<td>{row['insertion_auc']:.6f}</td>"
            f"<td>{row['curve_min']:.6f}</td>"
            f"<td>{html.escape(row['relative_source'])}</td>"
            "</tr>"
        )

    summary = result["summary"]
    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Output Token Input Influence</title>"
        "<style>body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        ".token{display:inline-block;border-radius:5px;padding:2px 5px;margin:2px 1px;border:1px solid rgba(23,32,51,.08);white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;}"
        "table{width:100%;border-collapse:collapse;font-size:14px;}th,td{border-bottom:1px solid #e5e9f2;padding:8px;text-align:left;}th{color:#536176;}code{white-space:pre-wrap;}"
        "</style></head><body><main>"
        "<section><h1>Output Token Input Influence</h1>"
        f"<p>{html.escape(summary['method'])}</p>"
        f"<p>Mean: {summary['mean_input_influence_score']:.6f}; max: {summary['max_input_influence_score']:.6f}; min: {summary['min_input_influence_score']:.6f}</p>"
        "<p>Redder tokens have higher prompt/system+user input influence; bluer tokens are lower and therefore more likely driven by previously generated tokens or prior language-model dynamics.</p>"
        "</section><section><h2>Token Heatmap</h2><div>" + " ".join(token_spans) + "</div></section>"
        "<section><h2>Scores</h2><table><thead><tr><th>#</th><th>Token</th><th>Input influence</th><th>Norm</th><th>AUC</th><th>Curve min</th><th>Relative source</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></section></main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_output_token_input_influence_svg(result, save_path):
    rows = result["rows"]
    width = 1180
    margin = 32
    label_w = 260
    chart_w = 820
    row_h = 28
    height = margin * 2 + 78 + max(1, len(rows)) * row_h
    max_score = max([row["input_influence_score"] for row in rows] + [1e-8])
    elements = []
    elements.append(f'<text x="{margin}" y="{margin}" font-size="22" font-weight="700" fill="#172033">Output Token Input Influence</text>')
    elements.append(f'<text x="{margin}" y="{margin + 28}" font-size="14" fill="#536176">AUC minus curve minimum, computed from insertion_word_score per output token</text>')
    y = margin + 76
    for row in rows:
        label = html.escape(f"{row['output_token_index']}: {row['token']}")
        bar_w = chart_w * float(row["input_influence_score"]) / max_score
        color = _color(row["normalized_input_influence"])
        elements.append(f'<text x="{margin}" y="{y}" font-size="13" fill="#172033">{label}</text>')
        elements.append(f'<rect x="{margin + label_w}" y="{y - 14}" width="{bar_w}" height="14" rx="3" fill="{color}"/>')
        elements.append(f'<text x="{margin + label_w + bar_w + 8}" y="{y - 2}" font-size="12" fill="#536176">{row["input_influence_score"]:.4f}</text>')
        y += row_h
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
