import html
from pathlib import Path

import numpy as np


def _as_array(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype)


def _normalize(scores):
    scores = _as_array(scores)
    if scores.size == 0:
        return scores
    span = float(scores.max() - scores.min())
    if span > 1e-8:
        return (scores - scores.min()) / span
    if float(np.abs(scores).max()) > 1e-8:
        return np.ones_like(scores)
    return np.zeros_like(scores)


def _input_attribution_scores(saved_json_file, span_count):
    """Text equivalent of visualization.visualization.add_value.

    S_set is represented by saved_json_file["ordered_masks"] and each mask maps back to
    a text span through saved_json_file["selected_region_indices"].  The value assigned
    to a region follows add_value: compare each SMDL score with the previous step, keep
    accumulating negative absolute deltas, then normalize after shifting by the minimum.
    """
    selected = [int(index) for index in saved_json_file.get("selected_region_indices", [])]
    smdl_scores = _as_array(saved_json_file.get("smdl_score", []))
    raw_scores = np.zeros(span_count, dtype=np.float32)

    if span_count == 0 or smdl_scores.size == 0 or not selected:
        return raw_scores, raw_scores

    baseline = _as_array(saved_json_file.get("baseline_score", [0.0]))
    original = _as_array(saved_json_file.get("org_score", [0.0]))
    first_reference = float(np.mean(1 - original + baseline))
    previous_scores = np.concatenate([[first_reference], smdl_scores[:-1]])
    marginal_values = smdl_scores - previous_scores

    running_value = 0.0
    for region_index, marginal_value in zip(selected, marginal_values):
        if 0 <= region_index < span_count:
            running_value -= abs(float(marginal_value))
            raw_scores[region_index] = running_value

    normalized_scores = _normalize(raw_scores - raw_scores.min())
    return raw_scores, normalized_scores


def _output_token_saliency(saved_json_file, output_token_count):
    """Text equivalent of visualization.visualization.get_word_saliency."""
    insertion_word_scores = saved_json_file.get("insertion_word_score", [])
    if not insertion_word_scores:
        return np.zeros(output_token_count, dtype=np.float32), np.zeros(output_token_count, dtype=np.float32)

    curves = _as_array([saved_json_file.get("baseline_score", [])] + insertion_word_scores).T
    regions = _as_array([0.0] + saved_json_file.get("region_area", []))
    raw_scores = []
    for insertion_curve in curves:
        auc = float(np.trapezoid(insertion_curve, regions)) if len(insertion_curve) > 1 else float(insertion_curve[0])
        raw_scores.append(auc - float(insertion_curve.min()))

    raw_scores = _as_array(raw_scores)
    if output_token_count > len(raw_scores):
        raw_scores = np.pad(raw_scores, (0, output_token_count - len(raw_scores)), constant_values=0.0)
    else:
        raw_scores = raw_scores[:output_token_count]
    return raw_scores, _normalize(raw_scores)


def _input_color(score):
    score = float(np.clip(score, 0.0, 1.0))
    r = int(255 * (1 - score) + 234 * score)
    g = int(255 * (1 - score) + 88 * score)
    b = int(255 * (1 - score) + 12 * score)
    return f"rgb({r},{g},{b})"


def _output_color(score):
    score = float(np.clip(score, 0.0, 1.0))
    if score < 0.5:
        ratio = score / 0.5
        r = int(49 * (1 - ratio) + 248 * ratio)
        g = int(130 * (1 - ratio) + 250 * ratio)
        b = int(206 * (1 - ratio) + 252 * ratio)
    else:
        ratio = (score - 0.5) / 0.5
        r = int(248 * (1 - ratio) + 185 * ratio)
        g = int(250 * (1 - ratio) + 28 * ratio)
        b = int(252 * (1 - ratio) + 28 * ratio)
    return f"rgb({r},{g},{b})"


def _span_flow_html(spans, raw_scores, normalized_scores):
    span_items = []
    for index, span in enumerate(spans):
        score = float(normalized_scores[index]) if index < len(normalized_scores) else 0.0
        raw_score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        title = html.escape(f"{span.role} region {index}; add_value {raw_score:.6f}; normalized {score:.3f}")
        text = html.escape(span.text if span.text else " ")
        span_items.append(
            f'<span class="region" title="{title}" style="background:{_input_color(score)}">{text}</span>'
        )
    return " ".join(span_items)


def _token_flow_html(output_tokens, raw_scores, normalized_scores):
    token_items = []
    for index, token in enumerate(output_tokens):
        score = float(normalized_scores[index]) if index < len(normalized_scores) else 0.0
        raw_score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        title = html.escape(f"token {index}; AUC - curve_min {raw_score:.6f}; normalized {score:.3f}")
        text = html.escape(token if token else " ")
        token_items.append(
            f'<span class="token" title="{title}" style="background:{_output_color(score)}">{text}</span>'
        )
    return " ".join(token_items)


def save_text_attribution_html(saved_json_file, spans, output_tokens, output_text, model_name, save_path):
    input_raw, input_norm = _input_attribution_scores(saved_json_file, len(spans))
    output_raw, output_norm = _output_token_saliency(saved_json_file, len(output_tokens))

    html_text = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Text Attribution</title>"
        "<style>"
        "body{margin:0;padding:32px;background:#f7f8fb;color:#172033;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;}"
        "main{max-width:1120px;margin:0 auto;}"
        "section{background:#fff;border:1px solid #dfe4ee;border-radius:8px;padding:20px;margin-bottom:18px;}"
        ".region,.token{display:inline-block;border:1px solid rgba(23,32,51,.14);border-radius:5px;margin:2px;padding:3px 6px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;}"
        "pre{white-space:pre-wrap;line-height:1.55;}"
        ".input-legend,.output-legend{height:14px;width:380px;border-radius:7px;display:inline-block;vertical-align:middle;margin:0 12px;border:1px solid rgba(23,32,51,.14);}"
        ".input-legend{background:linear-gradient(90deg,rgb(255,255,255),rgb(234,88,12));}"
        ".output-legend{background:linear-gradient(90deg,rgb(49,130,206),rgb(248,250,252),rgb(185,28,28));}"
        "</style></head><body><main>"
        f"<section><h1>Text Attribution</h1><p>Model: {html.escape(model_name)}</p>"
        "<p>Input colors follow the text equivalent of <code>add_value</code>. Output colors follow <code>get_word_saliency</code>: AUC of each output token insertion curve minus that curve's minimum.</p></section>"
        "<section><h2>Input-Level Attribution</h2><div>"
        + _span_flow_html(spans, input_raw, input_norm)
        + "</div><p>lower input attribution <span class='input-legend'></span> higher input attribution</p></section>"
        "<section><h2>Generated Output</h2><pre>"
        + html.escape(output_text)
        + "</pre></section>"
        "<section><h2>Output-Level Attribution</h2><div>"
        + _token_flow_html(output_tokens, output_raw, output_norm)
        + "</div><p>generated-prefix / LM-prior related <span class='output-legend'></span> input-text related</p></section>"
        "</main></body></html>"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html_text, encoding="utf-8")
    return save_path


def save_text_saliency_svg(saved_json_file, spans, save_path):
    raw_scores, normalized_scores = _input_attribution_scores(saved_json_file, len(spans))
    width = 1180
    margin = 32
    label_w = 760
    chart_w = 300
    row_h = 30
    height = margin * 2 + 64 + max(1, len(spans)) * row_h
    elements = [
        f'<text x="{margin}" y="{margin}" font-size="22" font-weight="700" fill="#172033">Input Region Saliency</text>',
        f'<text x="{margin}" y="{margin + 28}" font-size="14" fill="#536176">Text equivalent of visualization.visualization.add_value over all system/user input regions.</text>',
    ]
    y = margin + 76
    for index, span in enumerate(spans):
        score = float(normalized_scores[index]) if index < len(normalized_scores) else 0.0
        raw_score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        label = html.escape(f"{index}: [{span.role}] {span.text}")
        if len(label) > 110:
            label = label[:107] + "..."
        bar_w = chart_w * score
        elements.append(f'<text x="{margin}" y="{y}" font-size="13" fill="#172033">{label}</text>')
        elements.append(
            f'<rect x="{margin + label_w}" y="{y - 15}" width="{bar_w:.2f}" height="16" rx="3" fill="{_input_color(score)}"/>'
        )
        elements.append(
            f'<text x="{margin + label_w + chart_w + 10}" y="{y - 2}" font-size="12" fill="#536176">{raw_score:.4f}</text>'
        )
        y += row_h
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#f7f8fb"/>\n'
        f'<rect x="14" y="14" width="{width - 28}" height="{height - 28}" rx="8" fill="#ffffff" stroke="#dfe4ee"/>\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(svg, encoding="utf-8")
    return save_path
