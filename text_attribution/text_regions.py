from dataclasses import asdict, dataclass
import re
import unicodedata

import numpy as np


@dataclass
class TextSpan:
    role: str
    text: str
    char_start: int
    char_end: int
    token_indices: list

    def to_dict(self):
        return asdict(self)


def _is_cjk(char):
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _char_group(char):
    if char.isspace():
        return "space"
    if _is_cjk(char):
        return "cjk"
    category = unicodedata.category(char)
    if category[0] in {"L", "N"} or char == "_":
        return "word"
    return "punct"


def split_readable_spans(text):
    """Split text into word/CJK-run/punctuation spans, skipping whitespace."""
    spans = []
    index = 0
    while index < len(text):
        group = _char_group(text[index])
        if group == "space":
            index += 1
            continue

        start = index
        index += 1
        if group in {"word", "cjk"}:
            while index < len(text) and _char_group(text[index]) == group:
                index += 1

        spans.append((text[start:index], start, index))
    return spans


def _trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return text[start:end], start, end


def split_sentence_spans(text):
    """Split text into sentence-like spans, skipping surrounding whitespace."""
    spans = []
    sentence_start = None
    sentence_end_chars = set(".!?;。！？；")
    closing_chars = set("\"')]}”’》」』）】")
    index = 0

    while index < len(text):
        char = text[index]
        if sentence_start is None:
            if char.isspace():
                index += 1
                continue
            sentence_start = index

        if char == "\n":
            span = _trim_span(text, sentence_start, index)
            if span is not None:
                spans.append(span)
            sentence_start = None
        elif char in sentence_end_chars:
            end = index + 1
            while end < len(text) and text[end] in closing_chars:
                end += 1
            span = _trim_span(text, sentence_start, end)
            if span is not None:
                spans.append(span)
            sentence_start = None
            index = end - 1

        index += 1

    if sentence_start is not None:
        span = _trim_span(text, sentence_start, len(text))
        if span is not None:
            spans.append(span)

    return spans


def split_message_span(text):
    span = _trim_span(text, 0, len(text))
    return [span] if span is not None else []


def split_text_spans(text, granularity="readable"):
    if granularity == "readable":
        return split_readable_spans(text)
    if granularity == "sentence":
        return split_sentence_spans(text)
    if granularity == "message":
        return split_message_span(text)
    raise ValueError(f"Unsupported text span granularity: {granularity}")


def _apply_chat_template(tokenizer, messages, enable_thinking=False):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _role_content(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def _locate_message_ranges(rendered_text, messages):
    ranges = []
    cursor = 0
    for message in messages:
        role = message.get("role", "")
        if role not in {"system", "user"}:
            continue

        content = _role_content(message)
        if not content:
            continue

        start = rendered_text.find(content, cursor)
        if start < 0:
            start = rendered_text.find(content)
        if start < 0:
            raise ValueError(
                f"Could not locate {role!r} content in the rendered chat template."
            )
        end = start + len(content)
        ranges.append((role, content, start, end))
        cursor = end
    return ranges


def _token_offsets(tokenizer, rendered_text):
    encoded = tokenizer(
        rendered_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    return encoded["input_ids"], encoded["offset_mapping"]


def _tokens_for_char_range(offsets, start, end):
    token_indices = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= start or token_start >= end:
            continue
        if token_start == token_end:
            continue
        token_indices.append(token_index)
    return token_indices


def build_text_span_masks(tokenizer, messages, enable_thinking=False, granularity="readable"):
    """Build token-span masks for system/user content in a chat prompt.

    Returns:
        rendered_text: Full chat-template text.
        input_ids: Token ids for the rendered text, without adding extra tokens.
        masks: np.ndarray shaped [num_spans, seq_len].
        spans: List[TextSpan] with role/text/character/token metadata.
    """
    rendered_text = _apply_chat_template(tokenizer, messages, enable_thinking)
    input_ids, offsets = _token_offsets(tokenizer, rendered_text)
    seq_len = len(input_ids)

    spans = []
    masks = []
    for role, content, role_start, _ in _locate_message_ranges(rendered_text, messages):
        for span_text, local_start, local_end in split_text_spans(content, granularity):
            char_start = role_start + local_start
            char_end = role_start + local_end
            token_indices = _tokens_for_char_range(offsets, char_start, char_end)
            if not token_indices:
                continue

            mask = np.zeros(seq_len, dtype=np.float32)
            mask[token_indices] = 1.0
            masks.append(mask)
            spans.append(
                TextSpan(
                    role=role,
                    text=span_text,
                    char_start=char_start,
                    char_end=char_end,
                    token_indices=token_indices,
                )
            )

    if masks:
        mask_array = np.stack(masks, axis=0)
    else:
        mask_array = np.zeros((0, seq_len), dtype=np.float32)

    return rendered_text, input_ids, mask_array, spans


def decode_tokens(tokenizer, token_ids):
    return [
        tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()
