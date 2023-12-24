import copy
from typing import List, Tuple, Union, Dict, Any, Optional, overload
from overrides import overrides
from seal.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


@Sampler.register("multi-label-inference-net-normalized")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized",
)
class MultiLabelNormalized(InferenceNetSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if y is not None:
            return torch.sigmoid(y)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False
    
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
            x_, labels_, y_hat_, buffer_ = x.clone(), labels.clone(), y_hat.clone(), buffer
            if "unlabeled" in buffer.get('task'):
                x_, y_hat_, buffer_ = self.select_confident(
                    x_, y_hat_, buffer_, self.thresholding,
                )
                labels_ = self.self_training(
                    labels_, y_hat_, buffer_,
                    self.use_self_training, self.soft_label, self.use_ls
                )

            loss = self.loss_fn(
                x_, labels_.unsqueeze(1), y_hat_, y_cost_aug, buffer_,
            ) # labels_.unsqueeze(1) (batch, num_samples or 1, ...)
        else:
            loss = None

        y_hat, y_hat_aug, loss = self.normalize(y_hat), self.normalize(y_cost_aug), loss
        
        buffer["prob"] = y_hat.max(dim=-1)[0].mean(dim=-1)
        
        return y_hat, y_hat_aug, loss
    
    @overrides
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
        if not use_self_training:
            return labels
        else:
            with torch.no_grad():
                if y_hat.numel() == 0:
                    return labels
            
                if soft_label:
                    return self.normalize(y_hat).squeeze(1)

                assert buffer.get("meta") is not None
                assert len(y_hat.shape) == 3
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                pseudo_labels = torch.where(y_hat >= 0.5, 1.0, 0.0).to(device)
                
                if use_ls: 
                    pseudo_labels = pseudo_labels.float()
                    assert pseudo_labels.size() == y_hat.size()
                    pseudo_labels = pseudo_labels*(1- self.alpha) + y_hat*self.alpha  
            
            return pseudo_labels.squeeze(1)

    @overrides
    def reconstruct_inputs(self, x, filtered_batch):
        x_conf = copy.deepcopy(x)
        try:
            for key, value in x['x'].items():
                x_conf['x'][key] = value[filtered_batch]
        except:
            x_conf = x[filtered_batch]  
        return x_conf  



@Sampler.register("multi-label-inference-net-normalized-or-sampled")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized-or-sampled"
)
class MultiLabelNormalizedOrSampled(InferenceNetSampler):
    """
    Samples during training and normalizes during evaluation.
    """

    def __init__(
        self, num_samples: int = 1, keep_probs: bool = True, **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.keep_probs = keep_probs
        self.num_samples = num_samples if not keep_probs else num_samples - 1

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is not None:
            if self._mode == "sample":
                return self.generate_samples(y)
            else:  # inference
                return torch.sigmoid(y)
        else:
            return None

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        assert (
            y.dim() == 3
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        p = torch.sigmoid(y).squeeze(1)  # (batch, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=p).sample(  # type: ignore
                [self.num_samples]  # (num_samples, batch, num_labels)
            ),
            0,
            1,
        )  # (batch, num_samples, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, p.unsqueeze(1)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True


@Sampler.register("multi-label-inference-net-normalized-or-continuous-sampled")
@InferenceNetSampler.register(
    "multi-label-inference-net-normalized-or-continuous-sampled"
)
class MultiLabelNormalizedOrContinuousSampled(MultiLabelNormalizedOrSampled):
    """
    Samples during training and normalizes during evaluation.

    The samples are themselves probability distributions instead of hard samples. We
    do this by adding gaussian noise in the logit space (before taking sigmoid).
    """

    def __init__(self, std: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.std = std

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        assert (
            y.dim() == 3
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape (batch, 1, ...)"
        # add gaussian noise
        # y.shape == (batch, 1, num_labels)
        samples = torch.sigmoid(
            torch.normal(
                y.expand(
                    -1, self.num_samples, -1
                ),  # (batch, num_samples, num_labels)
                std=self.std,
            )
        )  # (batch, num_samples, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, torch.sigmoid(y)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples
