import contextlib
import copy
import warnings
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Any,
    Optional,
    Callable,
    Generator,
    overload,
    Iterator,
)

import numpy as np
import torch
from allennlp.common.lazy import Lazy
from allennlp.training.optimizers import Optimizer
from seal.common import ModelMode
from seal.modules.loss import Loss
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.sampler import Sampler
from seal.modules.score_nn import ScoreNN
from seal.modules.task_nn import (
    TaskNN,
    CostAugmentedLayer,
)


@Sampler.register("inference-network")
class InferenceNetSampler(Sampler):
    def parameters_with_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        yield from self.inference_nn.parameters()
        if self.cost_augmented_layer is not None:
            yield from self.cost_augmented_layer.parameters()

    def __init__(
        self,
        loss_fn: Loss,
        inference_nn: TaskNN,
        score_nn: ScoreNN,
        cost_augmented_layer: Optional[CostAugmentedLayer] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        task_thresholding: Dict = {"use_th":False},
        use_self_training:bool = False,
        task_soft_label: bool = False,
        task_label_smoothing: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        assert ScoreNN is not None
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )
        self.inference_nn = inference_nn
        self.cost_augmented_layer = cost_augmented_layer
        self.loss_fn = loss_fn

        self.logging_children.append(self.loss_fn)

        # Task net parameters for each method
        self.thresholding = task_thresholding
        self.use_self_training = use_self_training
        self.soft_label = task_soft_label
        self.use_ls = task_label_smoothing.get('use_ls', False)
        self.alpha = task_label_smoothing.get('alpha', 0.0)

    @property
    def is_normalized(self) -> bool:
        """Whether the sampler produces normalized or unnormalized samples"""

        return False

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return y

    def forward(
        self,
        x: Any,
        labels: Optional[
            torch.Tensor
        ],  #: If given will have shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        y_hat, y_cost_aug = self._get_values(
            x, labels, buffer
        )  # (batch_size, 1, ...) Unnormalized

        if labels is not None:
            # compute loss for logging.
            loss = self.loss_fn(
                x,
                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                y_hat,
                y_cost_aug,
                buffer,
            )
        else:
            loss = None

        return self.normalize(y_hat), self.normalize(y_cost_aug), loss

    def _get_values(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        y_inf: torch.Tensor = self.inference_nn(x, buffer).unsqueeze(
            1
        )  # (batch_size, 1, ...) unormalized
        # inference_nn is TaskNN so it will output tensor of shape (batch, ...)
        # hence the unsqueeze

        if self.cost_augmented_layer is not None and labels is not None:
            y_cost_aug = self.cost_augmented_layer(
                torch.cat(
                    (
                        y_inf.squeeze(1),
                        labels.to(dtype=y_inf.dtype),
                    ),
                    dim=-1,
                ),
                buffer,
            ).unsqueeze(
                1
            )  # (batch_size,1, ...)
        else:
            y_cost_aug = None

        return y_inf, y_cost_aug

    def self_training(
        self,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        use_self_training:bool,
        soft_label:bool,
        use_ls: bool,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def select_confident(
        self,
        x: Any,
        y_hat: torch.Tensor,
        buffer: Dict,
        thresholding:Dict,
        **kwargs: Any,
    ) -> None:
        if not thresholding['use_th']:
            return x, y_hat, buffer
        else:
            max_confidence = y_hat.max(dim=-1)[0].mean(dim=-1)
            
            if thresholding.get('method') == "local_mean":
                filtered_batch = (max_confidence >= max_confidence.mean()).nonzero(as_tuple=True)[0]
            
            elif thresholding.get('method') == "local_median":
                filtered_batch = (max_confidence >= max_confidence.median()).nonzero(as_tuple=True)[0]
            
            else: # self.thresholding.method == "score"
                max_confidence = buffer.get(buffer.get('score_name'))
                threshold = thresholding.get('score_conf').get('threshold')
                if not isinstance(threshold, torch.Tensor): 
                    threshold = torch.tensor([threshold], device=max_confidence.device)
                else:
                    threshold = threshold.to(max_confidence.device)
            filtered_batch = (max_confidence >= threshold).nonzero(as_tuple=True)[0].unique()
        
            y_hat_conf = y_hat[filtered_batch]
            
            x_conf = self.reconstruct_inputs(x, filtered_batch)
    
            buffer_conf = copy.deepcopy(buffer)
            for key in buffer.keys():
                value = buffer.get(key)
                if type(value) == torch.Tensor:
                    buffer_conf[key] = value[filtered_batch]
                elif type(value) == list:
                    buffer_conf[key] = [value[i.item()] for i in filtered_batch]
                else:
                    continue
            return x_conf, y_hat_conf, buffer_conf
    
    def reconstruct_inputs(
        self,
        x,
        filtered_batch,
    ):
        raise NotImplementedError



InferenceNetSampler.register("inference-network-unnormalized")(
    InferenceNetSampler
)

Sampler.register("inference-network-unnormalized")(InferenceNetSampler)
