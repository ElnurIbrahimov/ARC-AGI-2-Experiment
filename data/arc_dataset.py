"""
ARCDataset — PyTorch Dataset for ARC-AGI-2 tasks.

Loads tasks from JSON files in the standard ARC format and encodes them
for the hybrid Mamba-2 + Transformer model with 2D position IDs.
"""

from typing import Dict, List, Optional, Tuple
import json
import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.grid_tokenizer import GridTokenizer
from data.grid_utils import normalize_grid
from data.augmentation import augment_task

logger = logging.getLogger(__name__)


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC-AGI-2.

    Loads tasks from JSON files in the ARC-AGI-2 format:
    {
        "train": [{"input": [[...]], "output": [[...]]}, ...],
        "test": [{"input": [[...]], "output": [[...]]}]
    }

    Each item returns the task encoded for the model:
    - token_ids: full task token sequence
    - row_ids, col_ids: 2D position IDs
    - target_tokens: target grid tokens
    - task_id: string identifier
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        tokenizer: Optional[GridTokenizer] = None,
        augment: bool = False,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            data_dir: path to ARC-AGI-2 data root (contains training/, evaluation/ dirs)
            split: 'training' or 'evaluation'
            tokenizer: GridTokenizer instance (created if None)
            augment: apply data augmentation (dihedral + color permutations)
            max_seq_len: maximum token sequence length
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer or GridTokenizer(max_seq_len=max_seq_len)
        self.augment = augment
        self.max_seq_len = max_seq_len

        # Load all tasks
        self.tasks: List[Dict] = []
        self.task_ids: List[str] = []
        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load all task JSON files from the split directory."""
        split_dir = os.path.join(self.data_dir, self.split)

        if not os.path.isdir(split_dir):
            logger.warning(
                f"Data directory not found: {split_dir}. "
                f"Dataset will be empty (useful for testing without data)."
            )
            return

        json_files = sorted([
            f for f in os.listdir(split_dir)
            if f.endswith(".json")
        ])

        if not json_files:
            logger.warning(f"No JSON files found in {split_dir}")
            return

        for fname in json_files:
            fpath = os.path.join(split_dir, fname)
            try:
                with open(fpath, "r") as f:
                    task_data = json.load(f)

                task_id = os.path.splitext(fname)[0]

                # Validate structure
                if "train" not in task_data:
                    logger.warning(f"Skipping {fname}: missing 'train' key")
                    continue
                if "test" not in task_data:
                    logger.warning(f"Skipping {fname}: missing 'test' key")
                    continue

                # Each test example becomes one dataset item
                for test_idx, test_pair in enumerate(task_data["test"]):
                    item = {
                        "train": task_data["train"],
                        "test_input": test_pair["input"],
                        "test_output": test_pair.get("output"),  # None in eval
                        "task_id": task_id if len(task_data["test"]) == 1
                                   else f"{task_id}_{test_idx}",
                    }
                    self.tasks.append(item)
                    self.task_ids.append(item["task_id"])

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping {fname}: {e}")

        logger.info(f"Loaded {len(self.tasks)} task items from {split_dir}")

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns dict with:
        - token_ids: (max_seq_len,)
        - row_ids: (max_seq_len,)
        - col_ids: (max_seq_len,)
        - target_tokens: (target_len,)
        - task_id: str
        - demo_inputs: list of np arrays (for refinement loop)
        - demo_outputs: list of np arrays
        - test_input: np array
        - test_output: np array (may be zeros if not available)
        """
        task = self.tasks[idx]

        # Extract grids
        demo_inputs = [normalize_grid(p["input"]) for p in task["train"]]
        demo_outputs = [normalize_grid(p["output"]) for p in task["train"]]
        test_input = normalize_grid(task["test_input"])
        test_output = (
            normalize_grid(task["test_output"])
            if task["test_output"] is not None
            else np.zeros((1, 1), dtype=int)
        )

        # Augmentation
        if self.augment:
            augmented = augment_task(
                demo_inputs + [test_input],
                demo_outputs + [test_output],
                include_dihedral=True,
                include_color_perm=True,
                n_color_perms=2,
            )
            # Pick a random augmentation (index 0 is the original)
            aug_idx = np.random.randint(0, len(augmented))
            aug_inputs, aug_outputs = augmented[aug_idx]

            # Split back into demos + test
            demo_inputs = aug_inputs[:-1]
            demo_outputs = aug_outputs[:-1]
            test_input = aug_inputs[-1]
            test_output = aug_outputs[-1]

        # Encode task
        encoded = self.tokenizer.encode_task(demo_inputs, demo_outputs, test_input)
        encoded = self.tokenizer.pad_to_length(encoded, self.max_seq_len)

        # Encode target
        target_encoded = self.tokenizer.encode_target(test_output)

        return {
            "token_ids": encoded["token_ids"],
            "row_ids": encoded["row_ids"],
            "col_ids": encoded["col_ids"],
            "target_tokens": target_encoded["token_ids"],
            "task_id": task["task_id"],
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "test_input": test_input,
            "test_output": test_output,
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for DataLoader.

        Pads all sequences to the max length in the batch.
        Stacks tensors, keeps lists for variable-size grids.
        """
        # token_ids, row_ids, col_ids are already padded to max_seq_len
        token_ids = torch.stack([item["token_ids"] for item in batch])
        row_ids = torch.stack([item["row_ids"] for item in batch])
        col_ids = torch.stack([item["col_ids"] for item in batch])

        # target_tokens vary in length — pad to max in batch
        target_lens = [item["target_tokens"].shape[0] for item in batch]
        max_target_len = max(target_lens)
        padded_targets = []
        for item in batch:
            t = item["target_tokens"]
            if t.shape[0] < max_target_len:
                pad_len = max_target_len - t.shape[0]
                t = torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
            padded_targets.append(t)
        target_tokens = torch.stack(padded_targets)

        # Lists of variable-size data
        task_ids = [item["task_id"] for item in batch]
        demo_inputs = [item["demo_inputs"] for item in batch]
        demo_outputs = [item["demo_outputs"] for item in batch]
        test_inputs = [item["test_input"] for item in batch]
        test_outputs = [item["test_output"] for item in batch]

        return {
            "token_ids": token_ids,
            "row_ids": row_ids,
            "col_ids": col_ids,
            "target_tokens": target_tokens,
            "target_lens": torch.tensor(target_lens, dtype=torch.long),
            "task_ids": task_ids,
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "test_inputs": test_inputs,
            "test_outputs": test_outputs,
        }
