from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from dh_segment_torch.models import ResNetEncoder, Encoder
from dh_segment_torch.nn import Normalization

from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from pretrainedmodels import pretrained_settings as pretraining
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock


from dh_segment_text.models.embeddings_encoder.embeddings_encoder import (
    EmbeddingsEncoder,
)
from dh_segment_text.models.text_module import TextModule
from dh_segment_text.models.utils import conv2d_extra_params


class TextResnetEncoder(ResNetEncoder, TextModule):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        output_dims: List[int],
        embeddings_encoder: EmbeddingsEncoder = None,
        embeddings_level: int = -1,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = None,
        pretrained: bool = False,
        progress: bool = False,
    ):
        super().__init__(
            block,
            layers,
            output_dims,
            replace_stride_with_dilation,
            normalization,
            blocks,
            pretrained_settings,
            pretrained,
            progress,
        )

        self.embeddings_level = embeddings_level

        self.embeddings_encoder = embeddings_encoder

        if embeddings_level > -1:
            if embeddings_level >= min(5, self.blocks + 1):
                raise ValueError(
                    f"Embeddings level {embeddings_level} is larger than maximum embedding level {min(5, self.blocks + 1)}"
                )
            if embeddings_level == 0:
                conv2d_to_patch = self.conv1
            else:
                conv2d_to_patch = self.__getattr__(f'layer{embeddings_level}')[0].conv1
            in_channels = conv2d_to_patch.in_channels
            extra_params = conv2d_extra_params(conv2d_to_patch)

            patched_conv = torch.nn.Conv2d(
                in_channels=in_channels + embeddings_encoder.target_embeddings_size, **extra_params
            )

            if embeddings_level == 0:
                self.conv1 = patched_conv
            else:
                layer = self.__getattr__(f'layer{embeddings_level}')[0]
                setattr(layer, 'conv1', patched_conv)
                if hasattr(layer, 'downsample'):

                    down_layer = getattr(layer, 'downsample')
                    down_conv = down_layer[0]
                    in_channels = down_conv.in_channels
                    extra_params = conv2d_extra_params(down_conv)

                    down_layer[0] = torch.nn.Conv2d(
                        in_channels=in_channels + embeddings_encoder.target_embeddings_size, **extra_params
                    )

                    setattr(layer, 'downsample', down_layer)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x, embeddings, embeddings_map = x[0], x[1], x[2]
        x = super().normalize_if_pretrained(x)

        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        feature_maps = [x]
        for index, layer in enumerate(layers[: self.blocks + 1]):
            if self.embeddings_level == index:
                if self.embeddings_level == 1:
                    x = self.maxpool(x)
                    layer = self.layer1
                embeddings_map_reduced = self.embeddings_encoder(
                    embeddings, embeddings_map, x.shape[-2:]
                )
                x = torch.cat([x, embeddings_map_reduced], dim=1)

            x = layer(x)
            feature_maps.append(x)

        return feature_maps

    @classmethod
    def resnet18(
            cls,
            embeddings_encoder: EmbeddingsEncoder = None,
            embeddings_level: int = -1,
            replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
            normalization: Normalization = None,
            blocks: int = 4,
            pretrained_settings: Optional[
                Dict[str, Union[str, int, float, List[Union[int, float]]]]
            ] = pretraining["resnet18"]["imagenet"],
            pretrained: bool = True,
            progress: bool = False,
    ):
        return cls(
            embeddings_encoder=embeddings_encoder,
            embeddings_level=embeddings_level,
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            output_dims=[3, 64, 64, 128, 256, 512][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet34(
            cls,
            embeddings_encoder: EmbeddingsEncoder = None,
            embeddings_level: int = -1,
            replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
            normalization: Normalization = None,
            blocks: int = 4,
            pretrained_settings: Optional[
                Dict[str, Union[str, int, float, List[Union[int, float]]]]
            ] = pretraining["resnet34"]["imagenet"],
            pretrained: bool = True,
            progress: bool = False,
    ):
        return cls(
            embeddings_encoder=embeddings_encoder,
            embeddings_level=embeddings_level,
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 64, 128, 256, 512][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet50(
            cls,
            embeddings_encoder: EmbeddingsEncoder = None,
            embeddings_level: int = -1,
            replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
            normalization: Normalization = None,
            blocks: int = 4,
            pretrained_settings: Optional[
                Dict[str, Union[str, int, float, List[Union[int, float]]]]
            ] = pretraining["resnet50"]["imagenet"],
            pretrained: bool = True,
            progress: bool = False,
    ):
        return cls(
            embeddings_encoder=embeddings_encoder,
            embeddings_level=embeddings_level,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )


Encoder.register("text_resnet18", "resnet18")(TextResnetEncoder)
Encoder.register("text_resnet34", "resnet34")(TextResnetEncoder)
Encoder.register("text_resnet50", "resnet50")(TextResnetEncoder)
