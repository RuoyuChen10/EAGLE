import numpy as np
from tqdm import tqdm
import torch
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from .submodular_vision import MLLMSubModularExplanationVision

class EfficientMLLMVideoSubModularExplanationVision(MLLMSubModularExplanationVision):
    """
    Black-box explanation of multimodal large language
    model (MLLM) based on submodular subset selection.

    NOTE (Video version):
    - source image is a video tensor/array with shape (N, H, W, 3)
      (or a list of N frames each (H, W, 3))
    - each element in V_set is a binary mask with shape (N, H, W, 1)
    """
    def __init__(self,
                 model,
                 preproccessing_function=None,
                 lambda1=1.0,
                 lambda2=1.0,
                 search_scope=10,
                 pending_samples=8,
                 update_step=5,
                 # batch_size=4,
                 ):
        super(EfficientMLLMVideoSubModularExplanationVision, self).__init__(
            model=model,
            preproccessing_function=preproccessing_function,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        self.device = self.MLLM.device

        self.search_scope = search_scope
        self.update_step = update_step
        self.pending_samples = pending_samples

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
            "insertion_score_oppose": [],
            "deletion_score_oppose": [],
            "insertion_word_score_oppose": [],
            "deletion_word_score_oppose": [],
            "region_area_oppose": [],
        }

    def save_positive_file(self, insertion_score, deletion_score, smdl_score):
        self.saved_json_file["insertion_score"].append(
            insertion_score.mean().cpu().numpy().item()
        )
        self.saved_json_file["insertion_word_score"].append(
            insertion_score.cpu().numpy().tolist()
        )

        self.saved_json_file["deletion_score"].append(
            deletion_score.mean().cpu().numpy().item()
        )
        self.saved_json_file["deletion_word_score"].append(
            deletion_score.cpu().numpy().tolist()
        )

        self.saved_json_file["smdl_score"].append(smdl_score.cpu().item())

    def save_negative_file(self, insertion_score, deletion_score):
        self.saved_json_file["insertion_score_oppose"].append(
            insertion_score.mean().cpu().numpy().item()
        )
        self.saved_json_file["insertion_word_score_oppose"].append(
            insertion_score.cpu().numpy().tolist()
        )

        self.saved_json_file["deletion_score_oppose"].append(
            deletion_score.mean().cpu().numpy().item()
        )
        self.saved_json_file["deletion_word_score_oppose"].append(
            deletion_score.cpu().numpy().tolist()
        )

    def evaluation_maximun_sample(self, S_set, S_set_opposite):
        # V_set: list of (N,H,W,1) -> tensor (B,N,H,W,1)
        V_set_tensor = torch.from_numpy(np.array(self.V_set)).float().to(self.device)

        # refer_baseline: (N,H,W,1)
        alpha_batch = V_set_tensor + self.refer_baseline.unsqueeze(0)  # (B,N,H,W,1)
        # keep last channel and expand to RGB
        alpha_batch = alpha_batch.repeat(1, 1, 1, 1, 3)  # (B,N,H,W,3)

        if len(S_set) == 0 or self.update_count % self.update_step == 0:
            # Positive samples search
            source_tensor = self.source_tensor.unsqueeze(0).expand(
                alpha_batch.shape[0], -1, -1, -1, -1
            )  # (B,N,H,W,3)
        else:
            alpha_batch = alpha_batch[:self.search_scope]
            # Positive samples search with scope
            source_tensor = self.source_tensor.unsqueeze(0).expand(
                alpha_batch.shape[0], -1, -1, -1, -1
            )  # (B,N,H,W,3)

        batch_input_images = alpha_batch * source_tensor
        batch_input_images_reverse = (1 - alpha_batch) * source_tensor

        with torch.no_grad():
            # Insertion
            insertion_scores = self.MLLM_inference_batch_images(batch_input_images).to(torch.float32)

            # Deletion
            deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse).to(torch.float32)

            # Overall submodular score
            smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
            smdl_scores = smdl_scores.mean(-1)  # mean over token-dim (as your original code)
            arg_max_index = smdl_scores.argmax().cpu().item()

            if len(S_set) == 0 or self.update_count % self.update_step == 0:
                indices = torch.argsort(smdl_scores, descending=True)
                sorted_V = [self.V_set[i] for i in indices]
                self.V_set = sorted_V

                # Update baseline with best
                S_set.append(self.V_set[0])
                self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[0]).float().to(self.device)
                del self.V_set[0]
            else:
                # Update baseline with argmax in scoped candidates
                S_set.append(self.V_set[arg_max_index])
                self.refer_baseline = self.refer_baseline + torch.from_numpy(self.V_set[arg_max_index]).float().to(self.device)
                del self.V_set[arg_max_index]

            # Save intermediate results (keep your original indexing behavior)
            self.save_positive_file(
                insertion_scores[arg_max_index],
                deletion_scores[arg_max_index],
                smdl_scores[arg_max_index]
            )

            self.saved_json_file["region_area"].append(
                (self.refer_baseline.sum() / self.region_area).cpu().item()
            )

            # Negative samples search
            if len(S_set_opposite) == 0:
                arg_min_index = smdl_scores.argmin().cpu().item()

                self.save_negative_file(
                    deletion_scores[arg_min_index],
                    insertion_scores[arg_min_index]
                )

                S_set_opposite.append(self.V_set[-1])
                self.refer_baseline_opposite = self.refer_baseline_opposite - torch.from_numpy(self.V_set[-1]).float().to(self.device)
                del self.V_set[-1]

                self.saved_json_file["region_area_oppose"].append(
                    (self.refer_baseline_opposite.sum() / self.region_area).cpu().item()
                )

                return S_set, S_set_opposite

            elif len(self.V_set) > self.pending_samples:
                oppose_set_condidates = self.V_set[-self.pending_samples:]
                oppose_set_tensor = torch.from_numpy(np.array(oppose_set_condidates)).float().to(self.device)  # (B,N,H,W,1)

                alpha_batch_opposite = self.refer_baseline_opposite.unsqueeze(0) - oppose_set_tensor  # (B,N,H,W,1)
                alpha_batch_opposite = alpha_batch_opposite.repeat(1, 1, 1, 1, 3)  # (B,N,H,W,3)

                source_tensor = self.source_tensor.unsqueeze(0).expand(
                    alpha_batch_opposite.shape[0], -1, -1, -1, -1
                )  # (B,N,H,W,3)

                batch_input_images = alpha_batch_opposite * source_tensor
                batch_input_images_reverse = (1 - alpha_batch_opposite) * source_tensor

                # Insertion
                insertion_scores = self.MLLM_inference_batch_images(batch_input_images).to(torch.float32)
                # Deletion
                deletion_scores = self.MLLM_inference_batch_images(batch_input_images_reverse).to(torch.float32)

                oppose_smdl_scores = self.lambda1 * insertion_scores + self.lambda2 * (1 - deletion_scores)
                arg_max_index_oppose = oppose_smdl_scores.mean(-1).argmax().cpu().item()

                self.save_negative_file(
                    insertion_scores[arg_max_index_oppose],
                    deletion_scores[arg_max_index_oppose]
                )

                S_set_opposite.append(oppose_set_condidates[arg_max_index_oppose])
                self.refer_baseline_opposite = self.refer_baseline_opposite - torch.from_numpy(
                    oppose_set_condidates[arg_max_index_oppose]
                ).float().to(self.device)

                del self.V_set[-self.pending_samples + arg_max_index_oppose]

                self.saved_json_file["region_area_oppose"].append(
                    (self.refer_baseline_opposite.sum() / self.region_area).cpu().item()
                )

        return S_set, S_set_opposite

    def get_merge_set(self):
        # define subsets
        S_set = []
        S_set_opposite = []

        # refer_baseline: (N,H,W,1)
        self.refer_baseline = torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)
        self.refer_baseline_opposite = 1 - torch.zeros_like(torch.from_numpy(self.V_set[0]).float(), device=self.device)

        # sub_images: (2,N,H,W,3) via broadcast on last dim (1 -> 3)
        sub_images = torch.stack([
            self.source_tensor * self.refer_baseline,
            self.source_tensor * self.refer_baseline_opposite
        ])
        scores = self.MLLM_inference_batch_images(sub_images).to(torch.float32)

        self.saved_json_file["org_score"] = scores[1].cpu().numpy().tolist()
        self.saved_json_file["baseline_score"] = scores[0].cpu().numpy().tolist()

        self.update_count = 0
        for _ in tqdm(range(self.saved_json_file["sub-region_number"])):
            if len(self.V_set) == 1:
                S_set = S_set + self.V_set
                break
            S_set, S_set_opposite = self.evaluation_maximun_sample(S_set, S_set_opposite)
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
            + [1.]
        )

        # concatenate along first axis: (K,N,H,W,1)
        S_set = np.concatenate((S_set, S_set_opposite[::-1]))

        return S_set

    def __call__(self, image, V_set):
        """
        Args:
            image:
              - np.ndarray of shape (N,H,W,3), OR
              - list of N frames each (H,W,3)
            V_set:
              - list, each element is (N,H,W,1) binary mask
        """
        self.save_file_init()
        self.saved_json_file["sub-region_number"] = len(V_set)

        # accept list-of-frames or already-stacked video
        if isinstance(image, list):
            self.source_image = np.stack(image, axis=0)  # (N,H,W,3)
        else:
            self.source_image = image  # expect (N,H,W,3)

        if self.source_image.ndim != 4 or self.source_image.shape[-1] != 3:
            raise ValueError(f"Expected image shape (N,H,W,3), got {self.source_image.shape}")

        self.source_tensor = torch.from_numpy(self.source_image).float().to(self.device)

        N, self.h, self.w, _ = self.source_image.shape
        # video-level area for ratio computation
        self.region_area = N * self.h * self.w

        # store V_set (list of numpy masks)
        self.V_set = V_set.copy()

        Submodular_Subset = self.get_merge_set()
        self.saved_json_file["smdl_score"] = (np.array(self.saved_json_file["insertion_score"]) + 1 - np.array(self.saved_json_file["deletion_score"])).tolist()

        return Submodular_Subset, self.saved_json_file