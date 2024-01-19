from typing import List, Tuple, Union, Dict, Any, Optional, cast
from seal.modules.loss import Loss
from seal.modules.loss.inference_net_loss import (
    MarginBasedLoss,
)
from allennlp.common.checks import ConfigurationError
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.logging import (
    LoggingMixin,
    LoggedValue,
    LoggedScalarScalar,
    LoggedScalarScalarSample,
    LoggedNPArrayNPArraySample,
)
import torch

@Loss.register("reinforce")
class REINFORCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        self._loss_values = []

        self._inference_score_values = []
        if self.score_nn is None:
            raise ConfigurationError("score_nn cannot be None for REINFORCE Loss")

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],  # (batch, num_samples)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        with torch.no_grad():
            ratio = self._get_predicted_score(
                x, labels, y_hat, y_hat_extra, buffer, **kwargs
            )
        
        loss = self.compute_ce_loss(x, labels, y_hat, y_hat_extra, buffer)

        loss = ratio * loss

        return loss

    def _get_predicted_score(
        self,
        x: Any,
        labels: Optional[torch.Tensor],
        y_hat: torch.Tensor,
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        # labels shape (batch, 1, ...)
        # y_hat shape (batch, num_samples, ...)
        self.score_nn = cast(
            ScoreNN, self.score_nn
        )  # purely for typing, no runtime effect

        # score_nn always expects y to be normalized
        # do the normalization based on the task
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        # label_score = self.score_nn(
        #     x, labels.float(), buffer, **kwargs
        # )  # (batch, num_samples)

        self._inference_score_values.append(float(torch.mean(predicted_score)))
    
        # ratio = predicted_score / label_score
        ratio = predicted_score

        return ratio

    def compute_ce_loss(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        if labels.sum() <= -100 * labels.numel() or y_hat.numel() == 0 or labels.numel() == 0:    
            return torch.zeros((1,1) , requires_grad=True).to(y_hat.device)
        
        loss = self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype)).sum(
            dim=-1
        )  # (batch, 1,)
        self._loss_values.append(float(torch.mean(loss)))

        return loss

