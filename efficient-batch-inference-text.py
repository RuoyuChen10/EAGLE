import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from text_attribution import (
    EfficientLLMSubModularExplanationText,
    QwenTextAdaptor,
    build_text_span_masks,
)
from visualization.text_attribution_visualization import save_text_attribution_html, save_text_saliency_svg
from visualization.combined_text_attribution_report import save_combined_text_attribution_report
from text_attribution.output_token_input_influence import (
    compute_output_token_input_influence,
    save_output_token_input_influence_csv,
    save_output_token_input_influence_html,
    save_output_token_input_influence_json,
    save_output_token_input_influence_svg,
)


os.environ.setdefault("HF_HOME", "./model_checkpoint/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(description="Efficient batch attribution for pure LLM prompts.")
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-scope", type=int, default=8)
    parser.add_argument("--pending-samples", type=int, default=4)
    parser.add_argument("--update-step", type=int, default=10)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument(
        "--mask-strategy",
        choices=["replace", "attention"],
        default="replace",
        help="replace uses pad/eos token replacement and keeps attention_mask=1.",
    )
    parser.add_argument(
        "--input-granularity",
        choices=["sentence", "message", "readable"],
        default="sentence",
        help="Input attribution region granularity: sentence is sparse; message groups each role; readable keeps the old word/CJK/punctuation spans.",
    )
    parser.add_argument(
        "--target-token-limit",
        type=int,
        default=None,
        help="Optionally explain only the first N generated tokens.",
    )
    parser.add_argument("--output-dir", default="./text_attribution_outputs")
    return parser.parse_args()


def tensor_to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.user_prompt},
    ]

    qwen = QwenTextAdaptor.from_pretrained(args.model_name)
    rendered_text, input_ids, V_set, spans = build_text_span_masks(
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
    qwen.set_targets(generation["generated_answer_ids"], selected_token_indices)

    explainer = EfficientLLMSubModularExplanationText(
        qwen,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        search_scope=args.search_scope,
        pending_samples=args.pending_samples,
        update_step=args.update_step,
        batch_size=args.batch_size,
        mask_strategy=args.mask_strategy,
    )

    S_set, saved_json_file = explainer(input_ids, V_set)
    output_tokens = generation["output_tokens"]
    selected_tokens = [output_tokens[index] for index in selected_token_indices]

    saved_json_file.update(
        {
            "model_name": args.model_name,
            "system_prompt": args.system_prompt,
            "user_prompt": args.user_prompt,
            "rendered_prompt": rendered_text,
            "input_granularity": args.input_granularity,
            "input_ids": tensor_to_list(input_ids),
            "spans": [span.to_dict() for span in spans],
            "ordered_masks": tensor_to_list(S_set),
            "generated_answer_ids": tensor_to_list(generation["generated_answer_ids"]),
            "output_text": generation["output_text"],
            "output_tokens": output_tokens,
            "selected_interpretation_token_id": selected_token_indices,
            "selected_interpretation_token_word_id": tensor_to_list(
                qwen.selected_interpretation_token_word_id
            ),
            "selected_interpretation_tokens": selected_tokens,
        }
    )

    json_path = output_dir / "text_attribution_result.json"
    json_path.write_text(
        json.dumps(saved_json_file, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "insertion_deletion_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "region_area", "insertion_score", "deletion_score", "smdl_score"])
        rows = zip(
            saved_json_file.get("region_area", []),
            saved_json_file.get("insertion_score", []),
            saved_json_file.get("deletion_score", []),
            saved_json_file.get("smdl_score", []),
        )
        for step, (region_area, insertion_score, deletion_score, smdl_score) in enumerate(rows):
            writer.writerow([step, region_area, insertion_score, deletion_score, smdl_score])

    html_path = output_dir / "text_attribution_visualization.html"
    save_text_attribution_html(
        saved_json_file=saved_json_file,
        spans=spans,
        output_tokens=output_tokens,
        output_text=generation["output_text"],
        model_name=args.model_name,
        save_path=html_path,
    )

    saliency_svg_path = output_dir / "text_saliency_map.svg"
    save_text_saliency_svg(
        saved_json_file=saved_json_file,
        spans=spans,
        save_path=saliency_svg_path,
    )

    token_input_influence = compute_output_token_input_influence(saved_json_file)
    token_input_influence_json_path = output_dir / "output_token_input_influence.json"
    token_input_influence_csv_path = output_dir / "output_token_input_influence.csv"
    token_input_influence_html_path = output_dir / "output_token_input_influence.html"
    token_input_influence_svg_path = output_dir / "output_token_input_influence.svg"
    save_output_token_input_influence_json(token_input_influence, token_input_influence_json_path)
    save_output_token_input_influence_csv(token_input_influence, token_input_influence_csv_path)
    save_output_token_input_influence_html(token_input_influence, token_input_influence_html_path)
    save_output_token_input_influence_svg(token_input_influence, token_input_influence_svg_path)

    combined_report_path = output_dir / "combined_attribution_report.svg"
    save_combined_text_attribution_report(
        saved_json_file=saved_json_file,
        token_input_influence=token_input_influence,
        save_path=combined_report_path,
    )

    print(generation["output_text"])
    print(f"Saved attribution JSON to {json_path}")
    print(f"Saved insertion/deletion CSV to {csv_path}")
    print(f"Saved visualization HTML to {html_path}")
    print(f"Saved saliency map SVG to {saliency_svg_path}")
    print(f"Saved output token input influence JSON to {token_input_influence_json_path}")
    print(f"Saved output token input influence CSV to {token_input_influence_csv_path}")
    print(f"Saved output token input influence HTML to {token_input_influence_html_path}")
    print(f"Saved output token input influence SVG to {token_input_influence_svg_path}")
    print(f"Saved combined attribution report SVG to {combined_report_path}")


if __name__ == "__main__":
    main()
