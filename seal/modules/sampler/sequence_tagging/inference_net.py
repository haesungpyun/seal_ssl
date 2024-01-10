import copy
from typing import List, Tuple, Union, Dict, Any, Optional, overload
from overrides import overrides
from allennlp.nn.util import (
    get_lengths_from_binary_sequence_mask,
)
from seal.modules.sampler import (
    Sampler,
    SamplerModifier,
    InferenceNetSampler,
)
import torch
from torch.nn import functional as F
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal.modules.sequence_tagging_task_nn import (
    SequenceTaggingTaskNN,
)

@Sampler.register("sequence-tagging-inference-net-normalized-custom")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized-custom",
)
class SequenceTaggingNormalizedCustom(InferenceNetSampler):
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
            return torch.softmax(y,dim=-1)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False
    
    @overrides
    def forward(
        self, 
        x: Any, 
        labels: Optional[
            torch.Tensor
        ],
        buffer: Dict,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        y_hat, y_cost_aug = self._get_values(
            x, labels, buffer
        )  # (batch_size, 1, ...) Unnormalized
        
        if labels is not None:    
            x_, labels_, y_hat_, buffer_ = x, labels, y_hat, buffer
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

                mask = buffer.get("mask")
                assert mask is not None
                assert buffer.get("meta") is not None
                assert len(y_hat.shape) == 4
                
                (
                    _, _, _, wordpiece_ids,
                ) = self.constrained_decode(
                    y_hat.squeeze(1), mask, buffer.get("wordpiece_offsets")
                )
                
                sequence_lengths = get_lengths_from_binary_sequence_mask(
                    mask
                ).data.tolist()  # list with seq len for each instance in the batch.

                new_ids = []
                for (ids, seq_len) in zip(wordpiece_ids, sequence_lengths):
                    ids += [0 for _ in range(seq_len, labels.size(1))]
                    ids = F.one_hot(torch.tensor(ids, dtype=torch.int64, device=y_hat.device), self.num_tags)
                    new_ids.append(ids)
                
                pseudo_labels = torch.stack(new_ids).to(y_hat.device)
                
                if use_ls: 
                    pseudo_labels = pseudo_labels.float()
                    pseudo_labels = pseudo_labels*(1- self.alpha) + y_hat.squeeze(1)*self.alpha            

            return pseudo_labels
    
    @overrides
    def reconstruct_inputs(self, x, filtered_batch):
        x_conf = copy.deepcopy(x)
        for key, value in x['tokens'].items():
            x_conf['tokens'][key] = value[filtered_batch]
        return x_conf

@Sampler.register("sequence-tagging-inference-net-normalized")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized",
)
class SequenceTaggingNormalized(InferenceNetSampler):
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
            return torch.softmax(y,dim=-1)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False


@Sampler.register("sequence-tagging-inference-net-normalized-or-sampled")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized-or-sampled"
)
class SequenceTaggingNormalizedOrSampled(InferenceNetSampler):
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
                return torch.softmax(y,dim=-1)
        else:
            return None

    def generate_samples(self, y: torch.Tensor) -> torch.Tensor:
        '''
        This function provides discrete samples drawn from the probability y.
        Called by self.normalize() which is callend in the forward() of sampler.
        In the forward, when this function is called, y has (batch, 1, ...)
        '''
        assert (
            y.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        p = torch.softmax(y,dim=-1).squeeze(1)   # (batch, seq_len, num_labels)
        samples = torch.transpose(
            torch.distributions.categorical.Categorical(probs=p).sample(  # type: ignore, <-- logits=y is also possible.
                [self.num_samples]  # (num_samples, batch, seq_len)
            ),
            0,
            1,
        )  # (# batch, num_samples, seq_len)
        samples = torch.nn.functional.one_hot(samples,y.shape[-1]) # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, p.unsqueeze(1)), dim=1 #  p: (batch, 1, seq_len, num_labels)
            )  # (batch, num_samples+1, seq_len, num_labels)

        return samples

    @property
    def different_training_and_eval(self) -> bool:
        return True

    @property
    def is_normalized(self) -> bool:
        return True


@Sampler.register("sequence-tagging-inference-net-normalized-or-continuous-sampled")
@InferenceNetSampler.register(
    "sequence-tagging-inference-net-normalized-or-continuous-sampled"
)
class SequenceTaggingNormalizedOrContinuousSampled(SequenceTaggingNormalizedOrSampled):
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
            y.dim() == 4
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        assert (
            y.shape[1] == 1
        ), "Output of inference_net should be of shape  (batch, 1, seq_len, num_labels)"
        # add gaussian noise
        # y.shape == (batch, seq_len, num_labels)
        samples = torch.softmax(
            torch.normal(
                y.expand( # (batch, 1, seq_len, num_labels)
                    -1, self.num_samples, -1, -1
                ),  # (batch, num_samples, seq_len, num_labels)
                std=self.std,
            ),
            dim=-1
        )  # (batch, num_samples, seq_len, num_labels)

        if self.keep_probs:
            samples = torch.cat(
                (samples, torch.softmax(y, dim=-1)), dim=1
            )  # (batch, num_samples+1, num_labels)

        return samples
