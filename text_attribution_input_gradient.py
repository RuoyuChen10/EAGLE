import argparse
import os
from pathlib import Path

from baselines.gradient_text import (
    InputXGradientTextAttributor,
    build_gradient_result,
    save_gradient_attribution_json,
    save_gradient_span_scores_csv,
    save_gradient_text_attribution_html,
    save_gradient_text_saliency_svg,
)
from text_attribution import QwenTextAdaptor, build_text_span_masks


os.environ.setdefault("HF_HOME", "./model_checkpoint/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(description="Input x Gradient baseline attribution for pure LLM prompts.")
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
    parser.add_argument(
        "--target-token-limit",
        type=int,
        default=None,
        help="Optionally attribute only the first N generated tokens.",
    )
    parser.add_argument(
        "--target-score",
        choices=["logit", "logprob"],
        default="logit",
        help="Differentiate the target token logit or log-probability.",
    )
    parser.add_argument(
        "--span-reduction",
        choices=["sum", "mean", "max"],
        default="sum",
        help="How token scores inside a text span are aggregated.",
    )
    parser.add_argument("--output-dir", default="./input_gradient_text_attribution_outputs")
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
    selected_token_indices = list(range(len(generation["generated_answer_ids"])))
    if args.target_token_limit is not None:
        selected_token_indices = selected_token_indices[: args.target_token_limit]
    if not selected_token_indices:
        raise ValueError("No generated tokens were selected for attribution.")

    attributor = InputXGradientTextAttributor(
        qwen.model,
        qwen.tokenizer,
        target_score=args.target_score,
        span_reduction=args.span_reduction,
    )
    attribution = attributor.attribute(
        input_ids=input_ids,
        generated_answer_ids=generation["generated_answer_ids"],
        spans=spans,
        selected_output_token_indices=selected_token_indices,
    )
    result = build_gradient_result(
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

    json_path = output_dir / "input_gradient_text_attribution_result.json"
    csv_path = output_dir / "input_gradient_span_scores.csv"
    html_path = output_dir / "input_gradient_text_attribution_visualization.html"
    svg_path = output_dir / "input_gradient_text_saliency_map.svg"
    save_gradient_attribution_json(result, json_path)
    save_gradient_span_scores_csv(result, csv_path)
    save_gradient_text_attribution_html(result, html_path)
    save_gradient_text_saliency_svg(result, svg_path)

    print(generation["output_text"])
    print(f"Saved Input x Gradient attribution JSON to {json_path}")
    print(f"Saved Input x Gradient span CSV to {csv_path}")
    print(f"Saved Input x Gradient input visualization HTML to {html_path}")
    print(f"Saved Input x Gradient input saliency SVG to {svg_path}")


if __name__ == "__main__":
    main()
