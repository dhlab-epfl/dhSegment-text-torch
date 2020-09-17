from typing import Dict, Optional, Union, List, Set, Tuple

import torch
from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.metrics.metric import Metric, MetricType
from dh_segment_torch.models import Model
from dh_segment_torch.models.decoders.decoder import Decoder
from dh_segment_torch.models.encoders.encoder import Encoder
from dh_segment_torch.nn.loss.losses import Loss, BCEWithLogitsLoss, CrossEntropyLoss

from dh_segment_text.models.text_module import TextModule


class TextSegmentationModel(Model):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        loss: Loss = BCEWithLogitsLoss(),
        metrics: Dict[str, Metric] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.metrics: Dict[str, Metric] = metrics if metrics else {}

    def forward(
        self,
        input: torch.Tensor,
        embeddings: torch.tensor,
        embeddings_map: torch.tensor,
        target: Optional[torch.Tensor] = None,
        shapes: Optional[torch.Tensor] = None,
        track_metrics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        res = {}
        if isinstance(self.encoder, TextModule):
            features_maps = self.encoder([input, embeddings, embeddings_map])
        else:
            features_maps = self.encoder(input)

        if isinstance(self.decoder, TextModule):
            logits = self.decoder(*features_maps, embeddings=embeddings, embeddings_map=embeddings_map)
        else:
            logits = self.decoder(*features_maps)

        res["logits"] = logits
        if target is not None:
            loss = self.loss(logits, target, shapes)
            res["loss"] = loss
            if track_metrics:
                self.update_metrics(target, logits, shapes)
        return res

    def update_metrics(
        self,
        target: torch.Tensor,
        logits: torch.Tensor,
        shapes: Optional[torch.Tensor] = None,
    ):
        for metric in self.metrics.values():
            metric(target, logits, shapes)

    def get_metric(self, metric: str, reset: bool = False) -> MetricType:
        return self.metrics[metric].get_metric_value(reset)

    def get_metrics(self, reset: bool = False) -> Dict[str, MetricType]:
        return {
            metric_str: metric.get_metric_value(reset)
            for metric_str, metric in self.metrics.items()
        }

    def get_available_metrics(self) -> Set[str]:
        return set(self.metrics.keys())

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    @classmethod
    def from_partial(
        cls,
        encoder: Encoder,
        decoder: Lazy[Decoder],
        num_classes: int,
        loss: Optional[Lazy[Loss]] = None,
        metrics: Optional[
            Union[
                Dict[str, Lazy[Metric]],
                List[Union[Tuple[str, Lazy[Metric]], Lazy[Metric]]],
                Lazy[Metric],
            ]
        ] = None,
        multilabel: bool = False,
        classes_labels: Optional[List[str]] = None,
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        decoder = decoder.construct(
            encoder_channels=encoder.output_dims, num_classes=num_classes
        )
        if metrics is None:
            metrics = {}

        metric_names = None
        if isinstance(metrics, Lazy):
            metrics_list = [metrics]
            metric_names = [None]
        elif isinstance(metrics, Dict):
            metrics_list = list(metrics.values())
            metric_names = metrics.keys()
        elif isinstance(metrics, List):
            metrics_list = []
            metric_names = []
            for metric in metrics:
                if isinstance(metric, Tuple):
                    if len(metric) != 2 or not isinstance(metric[0], str):
                        raise ValueError(
                            "Expected metric tuple to be of size 2 with a first item a string"
                        )
                    metric_names.append(metric[0])
                    metrics_list.append(metric[1])
                else:
                    metric_names.append(None)
                    metrics_list.append(metric)
        else:
            raise ValueError(
                "Expected metrics to be either a metric, a dict of metrics or a list of metrics"
            )

        metrics_built = [
            metric.construct(
                num_classes=num_classes,
                ignore_padding=ignore_padding,
                multilabel=multilabel,
                classes_labels=classes_labels,
                margin=margin,
            )
            for metric in metrics_list
        ]
        assert len(metric_names) == len(metrics_built)
        metric_names = [
            Metric.get_type(type(metric)) if metric_name is None else metric_name
            for metric_name, metric in zip(metric_names, metrics_built)
        ]

        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "Expected each metric to have an unique name,"
                f"got {len(metric_names) - len(set(metric_names))} duplicate(s)."
            )

        metrics = dict(zip(metric_names, metrics_built))

        if loss:
            loss = loss.construct(ignore_padding=ignore_padding, margin=margin)
        else:
            if multilabel:
                loss = BCEWithLogitsLoss(ignore_padding=ignore_padding, margin=margin)
            else:
                loss = CrossEntropyLoss(ignore_padding=ignore_padding, margin=margin)

        return cls(encoder, decoder, loss, metrics)

    @classmethod
    def from_color_labels(
        cls,
        encoder: Encoder,
        decoder: Lazy[Decoder],
        color_labels: ColorLabels,
        loss: Optional[Lazy[Loss]] = None,
        metrics: Optional[
            Union[
                Dict[str, Lazy[Metric]],
                List[Union[Tuple[str, Lazy[Metric]], Lazy[Metric]]],
                Lazy[Metric],
            ]
        ] = None,
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        return cls.from_partial(
            encoder,
            decoder,
            color_labels.num_classes,
            loss,
            metrics,
            color_labels.multilabel,
            color_labels.labels,
            ignore_padding,
            margin,
        )


Model.register("text_segmentation_model", "from_partial")(TextSegmentationModel)
Model.register("text_segmentation_model_color_labels", "from_color_labels")(
    TextSegmentationModel
)
