import html
import textwrap
from pathlib import Path

import numpy as np

from visualization.text_attribution_visualization import _input_attribution_scores


WIDTH = 1320
CARD_X = 16
CONTENT_X = 34
CONTENT_RIGHT = WIDTH - 46
LINE_H = 23
CHIP_H = 27
CHIP_GAP_X = 7
CHIP_GAP_Y = 7


def _esc(value):
    return html.escape(str(value), quote=True)


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


def _text_block(elements, title, text, y, max_lines=None):
    elements.append(
        f'<text x="{CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">{_esc(title)}</text>'
    )
    y += 28
    lines = _wrap_lines(text)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    for line in lines:
        elements.append(
            f'<text x="{CONTENT_X}" y="{y}" font-size="15" fill="#172033">{_esc(line)}</text>'
        )
        y += LINE_H
    return y + 38


def _chip_label(text):
    label = _plain_text(text).replace("\n", "\\n").replace("\t", "\\t")
    return label if label else " "


def _chip_width(label):
    return max(30, min(360, 18 + len(label) * 8.12))


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


def _span_scores(saved_json_file, span_count):
    return _input_attribution_scores(saved_json_file, span_count)


def _chip_flow(elements, items, y, color_fn, title_prefix):
    x = CONTENT_X
    for item in items:
        label = _chip_label(item["label"])
        w = _chip_width(label)
        if x + w > CONTENT_RIGHT:
            x = CONTENT_X
            y += CHIP_H + CHIP_GAP_Y
        title = _esc(f"{title_prefix} {item['title']} {item['raw_score']:.4f}")
        fill = color_fn(item["norm_score"])
        elements.append(
            f'<g><title>{title}</title>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{CHIP_H}" rx="5" fill="{fill}" stroke="rgba(23,32,51,0.14)"/>'
            f'<text x="{x + 9}" y="{y + 18}" font-size="14" fill="#172033" xml:space="preserve">{_esc(label[:42])}</text>'
            "</g>"
        )
        x += w + CHIP_GAP_X
    return y + CHIP_H + 44


def _input_items(saved_json_file, role):
    spans = saved_json_file.get("spans", [])
    raw_scores, norm_scores = _span_scores(saved_json_file, len(spans))
    items = []
    for index, span in enumerate(spans):
        if span.get("role") != role:
            continue
        items.append(
            {
                "label": span.get("text", ""),
                "title": "input attribution score",
                "raw_score": float(raw_scores[index]),
                "norm_score": float(norm_scores[index]),
            }
        )
    return items


def _output_items(token_input_influence):
    rows = token_input_influence.get("rows", [])
    return [
        {
            "label": row.get("token", ""),
            "title": (
                f"token {row.get('output_token_index', '')} input influence; "
                f"{row.get('relative_source', '')}"
            ),
            "raw_score": float(row.get("input_influence_score", 0.0)),
            "norm_score": float(row.get("normalized_input_influence", 0.0)),
        }
        for row in rows
    ]


def _input_legend(elements, y):
    elements.append(
        f'<text x="{CONTENT_X}" y="{y}" font-size="16" font-weight="700" fill="#172033">Input attribution bar</text>'
    )
    elements.append(
        '<linearGradient id="inputAttribution" x1="0%" y1="0%" x2="100%">'
        '<stop offset="0%" stop-color="rgb(255,255,255)"/>'
        '<stop offset="100%" stop-color="rgb(234,88,12)"/>'
        "</linearGradient>"
    )
    elements.append(
        f'<text x="{CONTENT_X + 180}" y="{y + 22}" font-size="13" fill="#536176">lower input attribution</text>'
    )
    elements.append(
        f'<rect x="{CONTENT_X + 180}" y="{y - 13}" width="380" height="14" rx="7" fill="url(#inputAttribution)" stroke="rgba(23,32,51,0.14)"/>'
    )
    elements.append(
        f'<text x="{CONTENT_X + 578}" y="{y + 22}" font-size="13" fill="#536176">higher input attribution</text>'
    )
    return y + 54


def _output_legend(elements, y):
    elements.append(
        f'<text x="{CONTENT_X}" y="{y}" font-size="16" font-weight="700" fill="#172033">Output saliency bar</text>'
    )
    elements.append(
        '<linearGradient id="outputInfluence" x1="0%" y1="0%" x2="100%">'
        '<stop offset="0%" stop-color="rgb(49,130,206)"/>'
        '<stop offset="50%" stop-color="rgb(248,250,252)"/>'
        '<stop offset="100%" stop-color="rgb(185,28,28)"/>'
        "</linearGradient>"
    )
    elements.append(
        f'<text x="{CONTENT_X + 180}" y="{y + 22}" font-size="13" fill="#536176">generated-prefix / LM-prior related</text>'
    )
    elements.append(
        f'<rect x="{CONTENT_X + 180}" y="{y - 13}" width="380" height="14" rx="7" fill="url(#outputInfluence)"/>'
    )
    elements.append(
        f'<text x="{CONTENT_X + 578}" y="{y + 22}" font-size="13" fill="#536176">input-text related</text>'
    )
    return y + 54


def save_combined_text_attribution_report(saved_json_file, token_input_influence, save_path):
    elements = [
        f'<text x="{CONTENT_X}" y="34" font-size="26" font-weight="750" fill="#172033">Text Attribution Report</text>',
        f'<text x="{CONTENT_X}" y="62" font-size="14" fill="#536176">Input-level attribution over system/user prompt; output-level attribution uses AUC(insertion curve) - curve minimum per generated token.</text>',
    ]

    y = 96
    y = _text_block(elements, "System Prompt", saved_json_file.get("system_prompt", ""), y)
    y = _text_block(elements, "User Prompt", saved_json_file.get("user_prompt", ""), y)
    y = _text_block(elements, "Generated Output", saved_json_file.get("output_text", ""), y, max_lines=18)

    system_items = _input_items(saved_json_file, "system")
    if system_items:
        elements.append(
            f'<text x="{CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: System Prompt</text>'
        )
        y = _chip_flow(elements, system_items, y + 9, _input_color, "system")

    user_items = _input_items(saved_json_file, "user")
    if user_items:
        elements.append(
            f'<text x="{CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: User Prompt</text>'
        )
        y = _chip_flow(elements, user_items, y + 9, _input_color, "user")

    if system_items or user_items:
        y = _input_legend(elements, y + 12)

    output_items = _output_items(token_input_influence)
    if output_items:
        elements.append(
            f'<text x="{CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Output-Level Attribution: Generated Output Tokens</text>'
        )
        y = _chip_flow(elements, output_items, y + 9, _output_color, "token")
        y = _output_legend(elements, y + 12)

    height = int(y + 34)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">\n'
        '<rect width="100%" height="100%" fill="#f7f8fb"/>\n'
        f'<rect x="{CARD_X}" y="16" width="{WIDTH - 32}" height="{height - 32}" rx="8" fill="#ffffff" stroke="#dfe4ee"/>\n'
        + "".join(elements)
        + "\n</svg>\n"
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(svg, encoding="utf-8")
    return save_path
