import torch

import torch.nn.functional as F

from dh_segment_text.models.embeddings_encoder.embeddings_encoder import (
    EmbeddingsEncoder,
)


@EmbeddingsEncoder.register("no_op")
class NoOpEncoder(EmbeddingsEncoder):
    def forward(
        self,
        embeddings: torch.Tensor,
        embeddings_map: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        embeddings_map = F.interpolate(embeddings_map.unsqueeze(1).to(float), target_shape, mode="nearest").to(int).squeeze(1)
        batch_size = embeddings.shape[0]
        res = embeddings[torch.arange(batch_size).unsqueeze(1).unsqueeze(2), embeddings_map].permute(0,3,1,2)
        return res



