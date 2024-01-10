from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from seal.modules.loss import Loss
import numpy as np
import logging

logger = logging.getLogger(__name__)

@Loss.register("multi-label-bce")
class MultiLabelBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._loss_values = []

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        # all the labels are -100 -> unlabeled data (no pseudo_labeling, you can delete this condition by editing CE Loss to ignore some tokens or value)
        # y_hat/labels is empty means in select_confidence(thresholding method), all the samples are filtered out
        if labels.sum() <= -100 * labels.numel() or y_hat.numel() == 0 or labels.numel() == 0:    
            return torch.zeros((1,1) , requires_grad=True).to(y_hat.device)
        loss = self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype)).sum(
            dim=-1
        )  # (batch, 1,)
        self._loss_values.append(float(torch.mean(loss)))

        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {}

        if self._loss_values:
            metrics = {"cross_entropy_loss": np.mean(self._loss_values)}

        if reset:
            self._loss_values = []

        return metrics


@Loss.register("multi-label-bce-unreduced")
class MultiLabelBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        return self.loss_fn(
            y_hat, labels.to(dtype=y_hat.dtype)
        )  # (batch, label_size,)
