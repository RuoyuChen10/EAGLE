import html
import textwrap
from pathlib import Path

import numpy as np


REPORT_WIDTH = 1320
REPORT_CARD_X = 16
REPORT_CONTENT_X = 34
REPORT_CONTENT_RIGHT = REPORT_WIDTH - 46
REPORT_LINE_H = 23
REPORT_CHIP_H = 27
REPORT_CHIP_GAP_X = 7
REPORT_CHIP_GAP_Y = 7


def normalize_scores(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    span = float(values.max() - values.min())
    if span > 1e-8:
        return (values - values.min()) / span
    if float(np.abs(values).max()) > 1e-8:
        return np.ones_like(values)
    return np.zeros_like(values)


def input_color(score):
    score = float(np.clip(score, 0.0, 1.0))
    r = int(255 * (1 - score) + 234 * score)
    g = int(255 * (1 - score) + 88 * score)
    b = int(255 * (1 - score) + 12 * score)
    return f"rgb({r},{g},{b})"


def plain_text(value):
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def wrap_lines(text, width=146):
    lines = []
    for raw_line in plain_text(text).split("\n"):
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


def svg_text_block(elements, title, text, y, max_lines=None):
    elements.append(
        f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">{html.escape(title)}</text>'
    )
    y += 28
    lines = wrap_lines(text)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    for line in lines:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="15" fill="#172033">{html.escape(line)}</text>'
        )
        y += REPORT_LINE_H
    return y + 38


def chip_label(text):
    label = plain_text(text).replace("\n", "\\n").replace("\t", "\\t")
    return label if label else " "


def chip_width(label):
    return max(30, min(360, 18 + len(label) * 8.12))


def svg_chip_flow(elements, items, y, title_prefix):
    x = REPORT_CONTENT_X
    for item in items:
        label = chip_label(item["label"])
        width = chip_width(label)
        if x + width > REPORT_CONTENT_RIGHT:
            x = REPORT_CONTENT_X
            y += REPORT_CHIP_H + REPORT_CHIP_GAP_Y
        title = html.escape(f"{title_prefix} input attribution score {item['raw_score']:.4f}")
        fill = input_color(item["norm_score"])
        elements.append(
            f'<g><title>{title}</title>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{REPORT_CHIP_H}" rx="5" fill="{fill}" stroke="rgba(23,32,51,0.14)"/>'
            f'<text x="{x + 9}" y="{y + 18}" font-size="14" fill="#172033" xml:space="preserve">{html.escape(label[:42])}</text>'
            "</g>"
        )
        x += width + REPORT_CHIP_GAP_X
    return y + REPORT_CHIP_H + 44


def svg_input_legend(elements, y):
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


def span_items(result, role):
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


def save_input_attribution_report_svg(result, save_path, subtitle):
    elements = [
        f'<text x="{REPORT_CONTENT_X}" y="34" font-size="26" font-weight="750" fill="#172033">Text Attribution Report</text>',
        f'<text x="{REPORT_CONTENT_X}" y="62" font-size="14" fill="#536176">{html.escape(subtitle)}</text>',
    ]

    y = 96
    y = svg_text_block(elements, "System Prompt", result.get("system_prompt", ""), y)
    y = svg_text_block(elements, "User Prompt", result.get("user_prompt", ""), y)
    y = svg_text_block(elements, "Generated Output", result.get("output_text", ""), y, max_lines=18)

    system_items = span_items(result, "system")
    if system_items:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: System Prompt</text>'
        )
        y = svg_chip_flow(elements, system_items, y + 9, "system")

    user_items = span_items(result, "user")
    if user_items:
        elements.append(
            f'<text x="{REPORT_CONTENT_X}" y="{y}" font-size="20" font-weight="700" fill="#172033">Input-Level Attribution: User Prompt</text>'
        )
        y = svg_chip_flow(elements, user_items, y + 9, "user")

    if system_items or user_items:
        y = svg_input_legend(elements, y + 12)

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
