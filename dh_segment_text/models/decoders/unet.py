from typing import List, Optional

import torch
from dh_segment_torch.models import UnetDecoder, Decoder
from dh_segment_torch.nn.activations import Activation
from dh_segment_torch.nn.normalizations.normalization import Normalization

from dh_segment_text.models.embeddings_encoder.embeddings_encoder import (
    EmbeddingsEncoder,
)
from dh_segment_text.models.text_module import TextModule
from dh_segment_text.models.utils import conv2d_extra_params

@Decoder.register("text_unet")
class TextUnetDecoder(UnetDecoder, TextModule):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        embeddings_encoder: EmbeddingsEncoder = None,
        embeddings_level: int = -1,
        use_deconvolutions: bool = False,
        max_channels: Optional[int] = None,
        normalization: Normalization = None,
        activation: Activation = None,
    ):
        super().__init__(
            encoder_channels,
            decoder_channels,
            num_classes,
            use_deconvolutions,
            max_channels,
            normalization,
            activation,
        )

        self.embeddings_level = embeddings_level

        self.embeddings_encoder = embeddings_encoder

        if embeddings_level > -1:
            if embeddings_level >= len(self.level_ops):
                raise ValueError(
                    f"Embeddings level {embeddings_level} is larger than maximum embedding level {len(self.level_ops)}"
                )
            conv2d_to_patch = self.level_ops[embeddings_level]["decoder_conv"][0].conv2d

            in_channels = conv2d_to_patch.in_channels
            extra_params = conv2d_extra_params(conv2d_to_patch)
            self.level_ops[embeddings_level]["decoder_conv"][0] = torch.nn.Conv2d(
                in_channels + embeddings_encoder.target_embeddings_size, **extra_params
            )

    def forward(self, *features_maps, embeddings, embeddings_map) -> torch.tensor:

        features_maps = list(reversed(features_maps))
        x = features_maps[0]
        x = self.reduce_output_encoder(x)

        for index, (x_skip, level_op) in enumerate(
            zip(features_maps[1:], self.level_ops)
        ):
            if self.embeddings_level == index:
                embeddings_map_reduced = self.embeddings_encoder(
                    embeddings, embeddings_map, x.shape[-2:]
                )
                x = torch.cat([x, embeddings_map_reduced], dim=1)
            x_skip = level_op["reduce_dim"](x_skip)
            x = level_op["up_concat"](x, x_skip)
            x = level_op["decoder_conv"](x)
        x = self.logits(x)
        return x
