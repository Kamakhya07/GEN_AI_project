import json
import random
from typing import List, Dict, Any, Tuple
from collections import Counter

class SplitManager:
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def create_source_level_splits(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Splits patch records strictly by source_image_id to prevent any data leakage.
        """
        # Group records by source_image_id
        images_to_records: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            src_id = r["source_image_id"]
            if src_id not in images_to_records:
                images_to_records[src_id] = []
            images_to_records[src_id].append(r)

        unique_image_ids = sorted(list(images_to_records.keys()))
        rng = random.Random(self.seed)
        rng.shuffle(unique_image_ids)

        n_total = len(unique_image_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_img_ids = set(unique_image_ids[:n_train])
        val_img_ids = set(unique_image_ids[n_train:n_train + n_val])
        test_img_ids = set(unique_image_ids[n_train + n_val:])

        # Verify zero leakage across image IDs
        assert len(train_img_ids.intersection(val_img_ids)) == 0, "Train-Val leakage detected!"
        assert len(train_img_ids.intersection(test_img_ids)) == 0, "Train-Test leakage detected!"
        assert len(val_img_ids.intersection(test_img_ids)) == 0, "Val-Test leakage detected!"

        splits = {
            "train": [],
            "val": [],
            "test": []
        }

        for img_id, patches in images_to_records.items():
            if img_id in train_img_ids:
                for p in patches:
                    p_copy = dict(p)
                    p_copy["split"] = "train"
                    splits["train"].append(p_copy)
            elif img_id in val_img_ids:
                for p in patches:
                    p_copy = dict(p)
                    p_copy["split"] = "val"
                    splits["val"].append(p_copy)
            else:
                for p in patches:
                    p_copy = dict(p)
                    p_copy["split"] = "test"
                    splits["test"].append(p_copy)

        return splits

    def compute_class_distribution(self, splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
        """
        Computes per-split class counts and percentages.
        """
        summary = {}
        for split_name, patch_list in splits.items():
            counter = Counter(p["class_name"] for p in patch_list)
            summary[split_name] = dict(counter)
        return summary

    def save_manifest(self, splits: Dict[str, List[Dict[str, Any]]], manifest_path: str, class_counts_path: str):
        """
        Saves split manifest and class counts report as JSON.
        """
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        summary = self.compute_class_distribution(splits)
        with open(class_counts_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Split manifest saved to: {manifest_path}")
        print(f"Class count report saved to: {class_counts_path}")
        return summary
