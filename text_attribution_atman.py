import argparse
import os
from pathlib import Path

from baselines.atman_text import (
    AtManTextAttributor,
    build_atman_result,
    save_atman_attribution_json,
    save_atman_span_scores_csv,
    save_atman_text_attribution_html,
    save_atman_text_saliency_svg,
)
from text_attribution import QwenTextAdaptor, build_text_span_masks


os.environ.setdefault("HF_HOME", "./model_checkpoint/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(description="AtMan-style attention manipulation baseline for pure LLM prompts.")
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
        "--suppression-factor",
        type=float,
        default=0.1,
        help="Attention key suppression factor for the selected input span.",
    )
    parser.add_argument(
        "--score-reduction",
        choices=["sum", "mean"],
        default="sum",
        help="How selected target-token logprobs are reduced into one score.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="./atman_text_attribution_outputs")
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
        raise ValueError("No generated tokens were selected for AtMan attribution.")

    attributor = AtManTextAttributor(
        qwen.model,
        qwen.tokenizer,
        suppression_factor=args.suppression_factor,
        score_reduction=args.score_reduction,
        batch_size=args.batch_size,
    )
    attribution = attributor.attribute(
        input_ids=input_ids,
        generated_answer_ids=generation["generated_answer_ids"],
        spans=spans,
        selected_output_token_indices=selected_token_indices,
    )
    result = build_atman_result(
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

    json_path = output_dir / "atman_text_attribution_result.json"
    csv_path = output_dir / "atman_span_scores.csv"
    html_path = output_dir / "atman_text_attribution_visualization.html"
    svg_path = output_dir / "atman_text_saliency_map.svg"
    save_atman_attribution_json(result, json_path)
    save_atman_span_scores_csv(result, csv_path)
    save_atman_text_attribution_html(result, html_path)
    save_atman_text_saliency_svg(result, svg_path)

    print(generation["output_text"])
    print(f"Saved AtMan attribution JSON to {json_path}")
    print(f"Saved AtMan span CSV to {csv_path}")
    print(f"Saved AtMan input visualization HTML to {html_path}")
    print(f"Saved AtMan input saliency SVG to {svg_path}")


if __name__ == "__main__":
    main()
