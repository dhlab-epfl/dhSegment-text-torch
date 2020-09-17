from types import ModuleType
from typing import Callable, Optional, List, Tuple

import numpy as np
import torch
from dh_segment_torch.data import DataLoader, compute_paddings
from torch.nn import functional as F
from torch.utils import data


def compute_embeddings_paddings(
    num_embeddings: np.array,
) -> List[Tuple[int, int, int, int]]:
    max_num = num_embeddings.max()
    padding = max_num - num_embeddings

    return [(0, 0, 0, pad) for pad in padding]


def collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    if not all(["shape" in x for x in examples]):
        for example in examples:
            example["shape"] = torch.tensor(example["image"].shape[1:])

    heights = np.array([x["shape"][0] for x in examples])
    widths = np.array([x["shape"][1] for x in examples])
    image_paddings = compute_paddings(heights, widths)

    num_embeddings = np.array([x["embeddings"].shape[0] for x in examples])
    embeddings_paddings = compute_embeddings_paddings(num_embeddings)

    images = []
    masks = []
    all_embeddings = []
    all_embeddings_map = []
    shapes_out = []

    for example, image_padding, embeddings_padding in zip(
        examples, image_paddings, embeddings_paddings
    ):
        image, shape = example["image"], example["shape"]
        embeddings, embeddings_map = example["embeddings"], example["embeddings_map"]
        embeddings_map = (
            F.interpolate(
                embeddings_map.unsqueeze(0).unsqueeze(0).to(float),
                image.shape[1:],
                mode="nearest",
            )
            .squeeze(0)
            .squeeze(0)
            .to(int)
        )  # TODO check if better to float-int cast or ceil/round

        images.append(F.pad(image, image_padding))
        all_embeddings.append(F.pad(embeddings, embeddings_padding))
        all_embeddings_map.append(F.pad(embeddings_map, image_padding))
        shapes_out.append(shape)

        if "label" in example:
            label = example["label"]
            masks.append(F.pad(label, image_padding))

    if len(masks) > 0:
        return {
            "input": torch.stack(images, dim=0),
            "embeddings": torch.stack(all_embeddings, dim=0),
            "embeddings_map": torch.stack(all_embeddings_map, dim=0),
            "target": torch.stack(masks, dim=0),
            "shapes": torch.stack(shapes_out, dim=0),
        }
    else:
        return {
            "input": torch.stack(images, dim=0),
            "embeddings": torch.stack(all_embeddings, dim=0),
            "embeddings_map": torch.stack(all_embeddings_map, dim=0),
            "shapes": torch.stack(shapes_out, dim=0),
        }


@DataLoader.register("text_data_loader")
class TextDataLoader(DataLoader):
    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[data.Sampler] = None,
        batch_sampler: Optional[data.Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = collate_fn,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Optional[ModuleType] = None,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )
