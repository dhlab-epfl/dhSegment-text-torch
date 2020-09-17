import torch
from dh_segment_torch.config import Registrable


class EmbeddingsEncoder(torch.nn.Module, Registrable):
    default_implementation = "no_op"

    def __init__(self, target_embeddings_size: int):
        super().__init__()
        if target_embeddings_size <= 0:
            raise ValueError(
                f"Embeddings size should be at least 1, got {target_embeddings_size}"
            )
        self.target_embeddings_size = target_embeddings_size


    def forward(
        self,
        embeddings: torch.Tensor,
        embeddings_map: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Embeddings encoder should implement a forward method"
        )
