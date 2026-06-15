import numpy as np
import torch
from tqdm import tqdm


class EfficientLLMSubModularExplanationText(object):
    """Black-box submodular attribution over text token-span masks."""

    def __init__(
        self,
        model,
        lambda1=1.0,
        lambda2=1.0,
        search_scope=10,
        pending_samples=8,
        update_step=5,
        batch_size=None,
        mask_strategy="replace",
        mask_token_id=None,
    ):
        self.LLM = model
        self.device = self.LLM.device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.search_scope = search_scope
        self.pending_samples = pending_samples
        self.update_step = update_step
        self.batch_size = batch_size
        self.mask_strategy = mask_strategy
        self.mask_token_id = mask_token_id if mask_token_id is not None else self.LLM.mask_token_id

    def save_file_init(self):
        self.saved_json_file = {
            "insertion_score": [],
            "deletion_score": [],
            "smdl_score": [],
            "insertion_word_score": [],
            "deletion_word_score": [],
            "region_area": [],
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "mask_strategy": self.mask_strategy,
            "insertion_score_oppose": [],
            "deletion_score_oppose": [],
            "insertion_word_score_oppose": [],
            "deletion_word_score_oppose": [],
            "region_area_oppose": [],
            "selected_region_indices": [],
            "selected_region_indices_oppose": [],
        }

    def _to_float_masks(self, masks):
        if isinstance(masks, torch.Tensor):
            return masks.to(self.device, dtype=torch.float32)
        return torch.from_numpy(np.asarray(masks)).to(self.device, dtype=torch.float32)

    def _visible_to_inputs(self, visible_masks):
        visible_masks = visible_masks.clamp(0, 1)
        source_ids = self.source_input_ids.unsqueeze(0).expand(visible_masks.shape[0], -1)

        if self.mask_strategy == "attention":
            input_ids = source_ids
            attention_mask = visible_masks.to(torch.long)
        else:
            mask_ids = torch.full_like(source_ids, int(self.mask_token_id))
            input_ids = torch.where(visible_masks.bool(), source_ids, mask_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask

    def LLM_inference_batch(self, visible_masks):
        if visible_masks.shape[0] == 0:
            return torch.empty(0, device=self.device)

        batch_size = self.batch_size or visible_masks.shape[0]
        results = []
        for start in range(0, visible_masks.shape[0], batch_size):
            end = min(start + batch_size, visible_masks.shape[0])
            input_ids, attention_mask = self._visible_to_inputs(visible_masks[start:end])
            output_logits = self.LLM(input_ids, attention_mask=attention_mask)
            if output_logits.dim() == 1:
                output_logits = output_logits.unsqueeze(0)
            results.append(output_logits)
        return torch.cat(results, dim=0)

    def _paired_inference(self, insertion_visible, deletion_visible):
        paired_masks = torch.cat([insertion_visible, deletion_visible], dim=0)
        paired_scores = self.LLM_inference_batch(paired_masks).to(torch.float32)
        split_at = insertion_visible.shape[0]
        return paired_scores[:split_at], paired_scores[split_at:]

    def save_positive_file(self, insertion_score, deletion_score, smdl_score, region_index):
        self.saved_json_file["insertion_score"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score"].append(insertion_score.cpu().numpy().tolist())
        self.saved_json_file["deletion_score"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score"].append(deletion_score.cpu().numpy().tolist())
        self.saved_json_file["smdl_score"].append(smdl_score.cpu().item())
        self.saved_json_file["selected_region_indices"].append(int(region_index))

    def save_negative_file(self, insertion_score, deletion_score, region_index):
        self.saved_json_file["insertion_score_oppose"].append(insertion_score.mean().cpu().numpy().item())
        self.saved_json_file["insertion_word_score_oppose"].append(insertion_score.cpu().numpy().tolist())
        self.saved_json_file["deletion_score_oppose"].append(deletion_score.mean().cpu().numpy().item())
        self.saved_json_file["deletion_word_score_oppose"].append(deletion_score.cpu().numpy().tolist())
        self.saved_json_file["selected_region_indices_oppose"].append(int(region_index))

    def _candidate_tensors(self, candidates):
        masks = self._to_float_masks([candidate["mask"] for candidate in candidates])
        indices = [candidate["index"] for candidate in candidates]
        return masks, indices

    def evaluation_maximum_sample(self, S_set, S_set_opposite):
        candidates = self.V_set
        if len(S_set) and self.update_count % self.update_step != 0:
            candidates = candidates[: self.search_scope]

        candidate_masks, candidate_indices = self._candidate_tensors(candidates)
        selected_candidate_masks = (candidate_masks + self.refer_baseline.unsqueeze(0)).clamp(0, 1)
        insertion_visible = (self.fixed_visible_mask.unsqueeze(0) + selected_candidate_masks).clamp(0, 1)
        deletion_visible = (1 - selected_candidate_masks).clamp(0, 1)

        with torch.no_grad():
            insertion_scores, deletion_scores = self._paired_inference(
                insertion_visible, deletion_visible
            )
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
            smdl_scores = smdl_scores.mean(-1)
            arg_max_index = int(smdl_scores.argmax().cpu().item())

            if len(S_set) == 0 or self.update_count % self.update_step == 0:
                sorted_indices = torch.argsort(smdl_scores, descending=True).cpu().tolist()
                self.V_set = [candidates[i] for i in sorted_indices] + self.V_set[len(candidates) :]
                chosen_local_index = 0
                score_index = sorted_indices[0]
            else:
                chosen_local_index = arg_max_index
                score_index = arg_max_index

            chosen = self.V_set[chosen_local_index]
            chosen_mask = self._to_float_masks(chosen["mask"])
            S_set.append(chosen["mask"])
            self.refer_baseline = (self.refer_baseline + chosen_mask).clamp(0, 1)
            del self.V_set[chosen_local_index]

            self.save_positive_file(
                insertion_scores[score_index],
                deletion_scores[score_index],
                smdl_scores[score_index],
                chosen["index"],
            )
            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )

            if len(S_set_opposite) == 0 and self.V_set:
                remaining_candidate_ids = {id(item) for item in self.V_set}
                remaining_score_indices = [
                    index for index, item in enumerate(candidates) if id(item) in remaining_candidate_ids
                ]
                if remaining_score_indices:
                    remaining_scores = smdl_scores[remaining_score_indices]
                    arg_min_position = int(remaining_scores.argmin().cpu().item())
                    arg_min_index = remaining_score_indices[arg_min_position]
                    weakest = candidates[arg_min_index]
                else:
                    arg_min_index = score_index
                    weakest = self.V_set[-1]

                weakest_list_index = next(
                    index for index, item in enumerate(self.V_set) if id(item) == id(weakest)
                )
                del self.V_set[weakest_list_index]
                weakest_mask = self._to_float_masks(weakest["mask"])
                S_set_opposite.append(weakest["mask"])
                self.refer_baseline_opposite = (self.refer_baseline_opposite - weakest_mask).clamp(0, 1)
                self.save_negative_file(
                    deletion_scores[arg_min_index],
                    insertion_scores[arg_min_index],
                    weakest["index"],
                )
                self.saved_json_file["region_area_oppose"].append(
                    (
                        (self.fixed_visible_mask.sum() + self.refer_baseline_opposite.sum())
                        / self.total_visible_area
                    )
                    .cpu()
                    .item()
                )
                return S_set, S_set_opposite

            if len(self.V_set) > self.pending_samples:
                oppose_candidates = self.V_set[-self.pending_samples :]
                oppose_masks, oppose_indices = self._candidate_tensors(oppose_candidates)
                remaining_masks = (
                    self.refer_baseline_opposite.unsqueeze(0) - oppose_masks
                ).clamp(0, 1)
                insertion_visible = (
                    self.fixed_visible_mask.unsqueeze(0) + remaining_masks
                ).clamp(0, 1)
                deletion_visible = (1 - remaining_masks).clamp(0, 1)
                insertion_scores, deletion_scores = self._paired_inference(
                    insertion_visible, deletion_visible
                )
                oppose_smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (
                    1 - deletion_scores
                )
                arg_max_index_oppose = int(oppose_smdl_scores.mean(-1).argmax().cpu().item())
                chosen_oppose = oppose_candidates[arg_max_index_oppose]
                chosen_oppose_mask = self._to_float_masks(chosen_oppose["mask"])
                S_set_opposite.append(chosen_oppose["mask"])
                self.refer_baseline_opposite = (
                    self.refer_baseline_opposite - chosen_oppose_mask
                ).clamp(0, 1)
                del self.V_set[-self.pending_samples + arg_max_index_oppose]
                self.save_negative_file(
                    insertion_scores[arg_max_index_oppose],
                    deletion_scores[arg_max_index_oppose],
                    oppose_indices[arg_max_index_oppose],
                )
                self.saved_json_file["region_area_oppose"].append(
                    (
                        (self.fixed_visible_mask.sum() + self.refer_baseline_opposite.sum())
                        / self.total_visible_area
                    )
                    .cpu()
                    .item()
                )

        return S_set, S_set_opposite

    def get_merge_set(self):
        S_set = []
        S_set_opposite = []
        self.refer_baseline = torch.zeros_like(self.explainable_mask, device=self.device)
        self.refer_baseline_opposite = self.explainable_mask.clone()

        baseline_visible = self.fixed_visible_mask.unsqueeze(0)
        original_visible = torch.ones_like(self.fixed_visible_mask).unsqueeze(0)
        scores = self.LLM_inference_batch(torch.cat([baseline_visible, original_visible], dim=0)).to(
            torch.float32
        )
        self.saved_json_file["baseline_score"] = scores[0].cpu().numpy().tolist()
        self.saved_json_file["org_score"] = scores[1].cpu().numpy().tolist()

        self.update_count = 0
        for _ in tqdm(range(self.saved_json_file["sub-region_number"])):
            if len(self.V_set) == 1:
                last = self.V_set.pop(0)
                S_set.append(last["mask"])
                self.saved_json_file["selected_region_indices"].append(int(last["index"]))
                break
            if len(self.V_set) == 0:
                break
            S_set, S_set_opposite = self.evaluation_maximum_sample(S_set, S_set_opposite)
            self.update_count += 1

        self.saved_json_file["insertion_score"] = (
            self.saved_json_file["insertion_score"]
            + self.saved_json_file["insertion_score_oppose"][::-1]
            + [scores[1].cpu().mean().item()]
        )
        self.saved_json_file["deletion_score"] = (
            self.saved_json_file["deletion_score"]
            + self.saved_json_file["deletion_score_oppose"][::-1]
            + [scores[0].cpu().mean().item()]
        )
        self.saved_json_file["insertion_word_score"] = (
            self.saved_json_file["insertion_word_score"]
            + self.saved_json_file["insertion_word_score_oppose"][::-1]
            + [scores[1].cpu().numpy().tolist()]
        )
        self.saved_json_file["deletion_word_score"] = (
            self.saved_json_file["deletion_word_score"]
            + self.saved_json_file["deletion_word_score_oppose"][::-1]
            + [scores[0].cpu().numpy().tolist()]
        )
        self.saved_json_file["region_area"] = (
            self.saved_json_file["region_area"]
            + self.saved_json_file["region_area_oppose"][::-1]
            + [1.0]
        )
        self.saved_json_file["selected_region_indices"] = (
            self.saved_json_file["selected_region_indices"]
            + self.saved_json_file["selected_region_indices_oppose"][::-1]
        )

        ordered_sets = S_set + S_set_opposite[::-1]
        merged_set = np.asarray(ordered_sets, dtype=np.float32)
        self.saved_json_file["smdl_score"] = (
            np.array(self.saved_json_file["insertion_score"])
            + 1
            - np.array(self.saved_json_file["deletion_score"])
        ).tolist()
        return merged_set

    def __call__(self, source_input_ids, V_set):
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)

        self.source_input_ids = torch.as_tensor(source_input_ids, device=self.device, dtype=torch.long)
        self.V_set = [
            {"index": index, "mask": np.asarray(mask, dtype=np.float32)}
            for index, mask in enumerate(V_set)
        ]
        if len(self.V_set) == 0:
            raise ValueError("V_set is empty; no text spans were found for attribution.")

        V_set_tensor = self._to_float_masks([item["mask"] for item in self.V_set])
        self.explainable_mask = V_set_tensor.max(dim=0).values.clamp(0, 1)
        self.fixed_visible_mask = (1 - self.explainable_mask).clamp(0, 1)
        self.region_area = self.explainable_mask.sum().clamp_min(1.0)
        self.total_visible_area = torch.tensor(
            float(self.source_input_ids.numel()), device=self.device, dtype=torch.float32
        ).clamp_min(1.0)

        return self.get_merge_set(), self.saved_json_file
