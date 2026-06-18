import argparse
import json
import os
from pathlib import Path

from text_attribution import QwenTextAdaptor
from text_attribution.input_attribution_auc import (
    compute_existing_input_attribution_auc,
    compute_input_attribution_auc,
    save_input_attribution_auc_csv,
    save_input_attribution_auc_json,
    save_input_attribution_auc_summary,
    save_input_attribution_token_auc_csv,
)


os.environ.setdefault("HF_HOME", "./model_checkpoint/hf_cache")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate text input attribution with insertion/deletion AUC.")
    parser.add_argument("--outputs-root", default="./text_attribution_outputs")
    parser.add_argument("--result-glob", default="*/*attribution_result.json")
    parser.add_argument("--score-field", default="span_scores")
    parser.add_argument("--score-type", choices=["probability", "logit"], default="probability")
    parser.add_argument("--target-token-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _output_stem(args):
    stem = "input_attribution_auc"
    if args.score_type != "probability" or args.target_token_limit is not None:
        score_label = "logit_softmax" if args.score_type == "logit" else args.score_type
        stem = f"input_attribution_{score_label}"
        if args.target_token_limit is not None:
            stem += f"_top{args.target_token_limit}"
        stem += "_auc"
    return stem


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    result_paths = sorted(outputs_root.glob(args.result_glob))
    if not result_paths:
        raise FileNotFoundError(f"No result files matched {outputs_root / args.result_glob}.")

    first_result = _load_json(result_paths[0])
    model_name = first_result.get("model_name", "Qwen/Qwen3-8B")
    qwen = QwenTextAdaptor.from_pretrained(model_name)

    summary_rows = []
    output_stem = _output_stem(args)
    for result_path in result_paths:
        result = _load_json(result_path)
        if result.get("model_name", model_name) != model_name:
            raise ValueError(
                f"All results in one run must use the same model. "
                f"{result_path} uses {result.get('model_name')} but loaded {model_name}."
            )
        print(f"Evaluating {result_path}")
        if args.score_field in result:
            auc_result = compute_input_attribution_auc(
                qwen,
                result,
                score_field=args.score_field,
                batch_size=args.batch_size,
                score_type=args.score_type,
                target_token_limit=args.target_token_limit,
            )
        elif args.score_type == "probability" and args.target_token_limit is None:
            auc_result = compute_existing_input_attribution_auc(result)
        else:
            ordered_span_indices = result.get("selected_region_indices")
            if ordered_span_indices is None:
                raise ValueError(
                    f"{result_path} does not contain {args.score_field} or selected_region_indices."
                )
            auc_result = compute_input_attribution_auc(
                qwen,
                result,
                score_field="selected_region_indices",
                batch_size=args.batch_size,
                score_type=args.score_type,
                target_token_limit=args.target_token_limit,
                ordered_span_indices=[int(index) for index in ordered_span_indices],
            )
        json_path = result_path.parent / f"{output_stem}.json"
        csv_path = result_path.parent / f"{output_stem}.csv"
        token_csv_path = result_path.parent / f"{output_stem}_tokens.csv"
        save_input_attribution_auc_json(auc_result, json_path)
        save_input_attribution_auc_csv(auc_result, csv_path)
        save_input_attribution_token_auc_csv(auc_result, token_csv_path)
        summary_rows.append(
            {
                "method": auc_result["method"],
                "result_path": str(result_path),
                "num_spans": auc_result["num_spans"],
                "num_output_tokens": auc_result["num_output_tokens"],
                "insertion_auc": auc_result["insertion_auc"],
                "deletion_auc": auc_result["deletion_auc"],
                "insertion_start": auc_result["insertion_score"][0],
                "insertion_end": auc_result["insertion_score"][-1],
                "deletion_start": auc_result["deletion_score"][0],
                "deletion_end": auc_result["deletion_score"][-1],
            }
        )

    summary_path = outputs_root / f"{output_stem}_summary.csv"
    save_input_attribution_auc_summary(summary_rows, summary_path)
    print(f"Saved summary to {summary_path}")
    for row in summary_rows:
        print(
            f"{row['method']}: insertion_auc={row['insertion_auc']:.6f}, "
            f"deletion_auc={row['deletion_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
