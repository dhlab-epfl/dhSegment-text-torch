from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from dh_segment_torch.data.datasets import Dataset
from dh_segment_torch.data.transforms.albumentation import Compose
from dh_segment_torch.data.transforms.assign_labels import Assign

from dh_segment_text.data.text_dataset import (
    TextDataset,
    load_sample,
    sample_to_tensor,
    load_data_from_csv,
    load_data_from_csv_list,
    load_data_from_folder,
)


class ImageTextDataset(TextDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[Union[str, Path]] = None,
        image_base_dir: Optional[Union[str, Path]] = None,
        label_base_dir: Optional[Union[str, Path]] = None,
        text_base_dir: Optional[Union[str, Path]] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
        repeat_dataset: int = 1,
    ):
        super().__init__(
            data,
            base_dir,
            image_base_dir,
            label_base_dir,
            text_base_dir,
            repeat_dataset,
        )
        self.compose = compose
        if self.compose:
            self.compose.add_targets({"label": "mask", "embeddings_map": "mask"})
        self.assign_transform = assign_transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "image": self.data.image.iloc[idx],
            "label": self.data.label.iloc[idx],
            "embeddings": self.data.embeddings.iloc[idx],
            "embeddings_map": self.data.embeddings_map.iloc[idx],
        }

        sample = load_sample(sample)

        if self.compose:
            sample = self.compose(**sample)
        if self.assign_transform:
            label = sample["label"]
            sample.update({"label": self.assign_transform.apply(label)})
        return sample_to_tensor(sample)

    @classmethod
    def from_csv(
        cls,
        csv_filename: Union[str, Path],
        base_dir: Union[str, Path] = None,
        image_base_dir: Union[str, Path] = None,
        label_base_dir: Union[str, Path] = None,
        text_base_dir: Optional[Union[str, Path]] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
        repeat_dataset: int = 1,
    ):
        data = load_data_from_csv(str(csv_filename))
        return cls(
            data,
            base_dir,
            image_base_dir,
            label_base_dir,
            text_base_dir,
            compose,
            assign_transform,
            repeat_dataset,
        )

    @classmethod
    def from_csv_list(
        cls,
        csv_list: List[Union[str, Path]],
        base_dir: Union[str, Path] = None,
        image_base_dir: Union[str, Path] = None,
        label_base_dir: Union[str, Path] = None,
        text_base_dir: Optional[Union[str, Path]] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
        repeat_dataset: int = 1,
    ):
        data = load_data_from_csv_list([str(csv) for csv in csv_list])
        return cls(
            data,
            base_dir,
            image_base_dir,
            label_base_dir,
            text_base_dir,
            compose,
            assign_transform,
            repeat_dataset,
        )

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        compose: Compose = None,
        assign_transform: Assign = None,
        repeat_dataset: int = 1,
    ):
        data = load_data_from_folder(str(folder))
        return cls(
            data,
            compose=compose,
            assign_transform=assign_transform,
            repeat_dataset=repeat_dataset,
        )


Dataset.register("image_text_dataframe")(ImageTextDataset)
Dataset.register("image_text_csv", "from_csv")(ImageTextDataset)
Dataset.register("image_text_csv_list", "from_csv_list")(ImageTextDataset)
Dataset.register("image_text_folder", "from_folder")(ImageTextDataset)
