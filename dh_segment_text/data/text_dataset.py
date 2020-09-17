import logging
import mimetypes
import os
from abc import ABC
from glob import glob
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import cv2
import pandas as pd
import numpy as np
import torch
from scipy import sparse
from torchvision.transforms.functional import to_tensor

from dh_segment_torch.data.datasets import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[Union[str, Path]] = None,
        image_base_dir: Optional[Union[str, Path]] = None,
        label_base_dir: Optional[Union[str, Path]] = None,
        text_base_dir: Optional[Union[str, Path]] = None,
        repeat_dataset: int = 1,
    ):

        if base_dir:
            base_dir = str(base_dir)
            if text_base_dir:
                logger.warning("Base dir and text base dir were set, ignoring base dir")
            else:
                text_base_dir = base_dir

        if text_base_dir is not None:
            text_base_dir = str(text_base_dir)
            data["embeddings"] = data["embeddings"].apply(
                lambda path: os.path.join(text_base_dir, path)
            )
            data["embeddings_map"] = data["embeddings_map"].apply(
                lambda path: os.path.join(text_base_dir, path)
            )

        super().__init__(data, base_dir, image_base_dir, label_base_dir, repeat_dataset)

    def check_filenames_exist(self):
        super().check_filenames_exist()
        for embeddings_filename in list(self.data.embeddings.values):
            if not os.path.exists(embeddings_filename):
                raise FileNotFoundError(embeddings_filename)
        for embeddings_map_filename in list(self.data.embeddings_map.values):
            if not os.path.exists(embeddings_map_filename):
                raise FileNotFoundError(embeddings_map_filename)


def load_data_from_csv(csv_filename: str):
    return pd.read_csv(
        csv_filename,
        header=None,
        names=["image", "label", "embeddings", "embeddings_map"],
    )


def load_data_from_csv_list(csv_list: List[str]):
    list_dataframes = list()
    for csv_file in csv_list:
        assert os.path.isfile(csv_file), f"{csv_file} does not exist"
        list_dataframes.append(
            pd.read_csv(
                csv_file,
                header=None,
                names=["image", "label", "embeddings", "embeddings_map"],
            )
        )

    return pd.concat(list_dataframes, axis=0)


def load_data_from_folder(folder: str):
    image_dir = os.path.join(folder, "images")
    labels_dir = os.path.join(folder, "labels")
    text_dir = os.path.join(folder, "text")
    check_dirs_exist(image_dir, labels_dir, text_dir)
    input_data = compose_input_data(image_dir, labels_dir, text_dir)
    return pd.DataFrame(
        data=input_data, columns=["image", "label", "embeddings", "embeddings_map"]
    )


def load_sample(sample: dict) -> dict:
    """
    Loads the image, the label image, the embeddings and the embeddings map. Returns the updated dictionary.

    :param sample: dictionary containing at least ``image``, ``label``, ``embeddings`` and ``embeddings_map`` keys.
    """
    image_filename, label_filename = sample["image"], sample["label"]
    embeddings_filename, embeddings_map_filename = (
        sample["embeddings"],
        sample["embeddings_map"],
    )

    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label image
    label_image = cv2.imread(label_filename)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    embeddings = np.load(embeddings_filename).astype(np.float32)

    embeddings_map = (
        sparse.load_npz(embeddings_map_filename).todense().astype(np.int32)
    )

    sample.update(
        {
            "image": image,
            "label": label_image,
            "embeddings": embeddings,
            "embeddings_map": embeddings_map,
        }
    )
    return sample


def sample_to_tensor(sample: Dict[str, Any]):
    image, label = sample["image"], sample["label"]
    embeddings, embeddings_map = sample['embeddings'], sample['embeddings_map']

    # If we have multilabel, we need to transpose
    if label.ndim == 3:
        label = label.transpose((2, 0, 1))

    sample.update(
        {
            "image": to_tensor(image),
            "label": torch.from_numpy(label),
            "shape": torch.tensor(image.shape[:2]),
            "embeddings": torch.from_numpy(embeddings),
            "embeddings_map": torch.from_numpy(embeddings_map)
        }
    )
    return sample


def get_image_exts():
    image_exts = [
        ext for ext, app in mimetypes.types_map.items() if app.startswith("image")
    ]
    image_exts = image_exts + [ext.upper() for ext in image_exts]
    return image_exts


def check_dirs_exist(image_dir: str, labels_dir: str, text_dir: str):
    assert os.path.isdir(image_dir), f"Dataset creation: {image_dir} not found."
    assert os.path.isdir(labels_dir), f"Dataset creation: {labels_dir} not found."
    assert os.path.isdir(text_dir), f"Dataset creation: {text_dir} not found."


def compose_input_data(image_dir: str, labels_dir: str, text_dir):
    image_extensions = get_image_exts()
    image_files = list()
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, f"*{ext}"), recursive=True))
    input_data = list()
    for image_filename in image_files:
        basename = ".".join(os.path.basename(image_filename).split(".")[:-1])
        label_filename_candidates = glob(os.path.join(labels_dir, basename + ".*"))

        if len(label_filename_candidates) == 0:
            logging.error(
                f"Did not found the corresponding label image of {image_filename} "
                f"in {labels_dir} directory"
            )
            continue
        elif len(label_filename_candidates) > 1:
            logging.warning(
                f"Found more than 1 label match for {image_filename}. "
                f"Taking the first one {label_filename_candidates[0]}"
            )

        label_filename = label_filename_candidates[0]

        embeddings_filename_candidates = glob(
            os.path.join(text_dir, basename + "_embeddings.npy")
        )

        if len(embeddings_filename_candidates) == 0:
            logging.error(
                f"Did not found the corresponding embeddings of {image_filename} "
                f"in {text_dir} directory"
            )
            continue
        elif len(embeddings_filename_candidates) > 1:
            logging.warning(
                f"Found more than 1 label match for {image_filename}. "
                f"Taking the first one {embeddings_filename_candidates[0]}"
            )

        embeddings_filename = embeddings_filename_candidates[0]

        embeddings_map_filename_candidates = glob(
            os.path.join(text_dir, basename + "_map.npz")
        )

        if len(embeddings_map_filename_candidates) == 0:
            logging.error(
                f"Did not found the corresponding embeddings of {image_filename} "
                f"in {text_dir} directory"
            )
            continue
        elif len(embeddings_map_filename_candidates) > 1:
            logging.warning(
                f"Found more than 1 label match for {image_filename}. "
                f"Taking the first one {embeddings_map_filename_candidates[0]}"
            )

        embeddings_map_filename = embeddings_map_filename_candidates[0]

        input_data.append(
            (
                image_filename,
                label_filename,
                embeddings_filename,
                embeddings_map_filename,
            )
        )
    return input_data
