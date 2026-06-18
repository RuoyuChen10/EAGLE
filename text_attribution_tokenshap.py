import argparse
import os
from pathlib import Path

from baselines.tokenshap_text import (
    TokenSHAPTextAttributor,
    build_tokenshap_result,
    save_tokenshap_attribution_json,
    save_tokenshap_span_scores_csv,
    save_tokenshap_text_attribution_html,
    save_tokenshap_text_saliency_svg,
)
from text_attribution import QwenTextAdaptor, build_text_span_masks


os.environ.setdefault("HF_HOME", "./model_checkpoint/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(description="TokenSHAP black-box response-similarity baseline.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--system-prompt",
        default="You are a precise assistant. Answer with a concise explanation.",
    )
    parser.add_argument(
        "--user-prompt",
        default="Explain why batch inference can make perturbation-based attribution faster.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--input-granularity",
        choices=["sentence", "message", "readable"],
        default="sentence",
    )
    parser.add_argument("--sampling-ratio", type=float, default=0.0)
    parser.add_argument("--max-combinations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./tokenshap_text_attribution_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]

    qwen = QwenTextAdaptor.from_pretrained(args.model_name)
    rendered_text, input_ids, _, spans = build_text_span_masks(
        qwen.tokenizer,
        messages,
        enable_thinking=False,
        granularity=args.input_granularity,
    )
    generation = qwen.generate_from_rendered_chat(
        rendered_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    attributor = TokenSHAPTextAttributor(
        qwen_adaptor=qwen,
        messages=messages,
        spans=spans,
        sampling_ratio=args.sampling_ratio,
        max_combinations=args.max_combinations,
        random_seed=args.random_seed,
        max_new_tokens=args.max_new_tokens,
    )
    attribution = attributor.attribute()
    result = build_tokenshap_result(
        attribution=attribution,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        rendered_prompt=rendered_text,
        input_granularity=args.input_granularity,
        input_ids=input_ids,
        spans=spans,
        generation=generation,
    )

    json_path = output_dir / "tokenshap_text_attribution_result.json"
    csv_path = output_dir / "tokenshap_span_scores.csv"
    html_path = output_dir / "tokenshap_text_attribution_visualization.html"
    svg_path = output_dir / "tokenshap_text_saliency_map.svg"
    save_tokenshap_attribution_json(result, json_path)
    save_tokenshap_span_scores_csv(result, csv_path)
    save_tokenshap_text_attribution_html(result, html_path)
    save_tokenshap_text_saliency_svg(result, svg_path)

    print(generation["output_text"])
    print(f"Saved TokenSHAP attribution JSON to {json_path}")
    print(f"Saved TokenSHAP span CSV to {csv_path}")
    print(f"Saved TokenSHAP input visualization HTML to {html_path}")
    print(f"Saved TokenSHAP input saliency SVG to {svg_path}")


if __name__ == "__main__":
    main()
