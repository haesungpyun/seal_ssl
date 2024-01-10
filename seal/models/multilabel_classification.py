import copy
import logging
from typing import List, Tuple, Union, Dict, Any, Optional
from overrides import overrides

import torch
from allennlp.models import Model

from seal.metrics import (
    MultilabelClassificationF1,
    MultilabelClassificationMeanAvgPrecision,
    MultilabelClassificationMicroAvgPrecision,
    MultilabelClassificationRelaxedF1,
    MultilabelClassificationAvgRank,
    MultilabelClassificationMeanReciprocalRank,
    MultilabelClassificationNormalizedDiscountedCumulativeGain,
    MultilabelClassificationRankBiasedOverlap,
)
from seal.training.callbacks.write_read_scores import ThresholdingCallback
from .base import ScoreBasedLearningModel
from ..modules.oracle_value_function.manhatten_distance import (
    ManhattanDistanceValueFunction,
)

logger = logging.getLogger(__name__)


@Model.register(
    "multi-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@Model.register(
    "multi-label-classification", constructor="from_partial_objects"
)
@ScoreBasedLearningModel.register(
    "multi-label-classification-with-infnet",
    constructor="from_partial_objects_with_shared_tasknn",
)
@ScoreBasedLearningModel.register(
    "multi-label-classification", constructor="from_partial_objects"
)
class MultilabelClassification(ScoreBasedLearningModel):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # metrics
        self.f1 = MultilabelClassificationF1()
        self.map = MultilabelClassificationMeanAvgPrecision()

        # self.micro_map = MultilabelClassificationMicroAvgPrecision()

        self.relaxed_f1 = MultilabelClassificationRelaxedF1()

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """For mlc we add a dim at 1"""

        return labels.unsqueeze(1)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        """Remove the dim at 1"""

        return y.squeeze(1)

    @overrides
    def make_unlabel_y(self, label)->torch.Tensor:
        assert len(label.shape) == 1
        return label.unsqueeze(-1).expand(-1, self.vocab.get_vocab_size('labels'))

    @torch.no_grad()
    @overrides
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:

        self.map(y_hat, labels)

        # self.micro_map(y_hat, labels)

        if not self.inference_module.is_normalized:
            y_hat_n = torch.sigmoid(y_hat)
        else:
            y_hat_n = y_hat

        self.relaxed_f1(y_hat_n, labels)
        self.f1(y_hat_n, labels)

        # save scores to storage in validation session
        if not self.training:
            ThresholdingCallback(self.serialization_dir).save_to_storage(
                score_conf=self.thresholding.get('score_conf'),
                buffer=buffer,
                predictions=y_hat.tolist(),
                ground_truth=labels.tolist(),
                exact_match=False
            )

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "MAP": self.map.get_metric(reset),
            "fixed_f1": self.f1.get_metric(reset),
            # "micro_map": self.micro_map.get_metric(reset),
            "relaxed_f1": self.relaxed_f1.get_metric(reset),
        }

        return metrics

    # construct args for multi-task forward considering unlabeled data and labeled data
    # dataset reader should pass data_type in metadata
    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        kwargs["buffer"] = self.initialize_buffer(**kwargs)

        metadata = kwargs.get("meta")
        kwargs['buffer']['meta'] = metadata     

        task = kwargs.get("task") 
        # Check labeled data
        if task is None:  # if not multi-task
            if metadata[0].get("data_type") is None: # if unlabeled data, should pass data_type from data_reader
                task = ["labeled"] * len(metadata)
            else:
                task = []
                for meta in metadata:
                    task.append(meta.get('data_type'))
                
        kwargs['buffer']['task'] = task     
        
        if "unlabeled" in task:
            try:
                kwargs['labels'] = torch.stack(kwargs.pop('labels')).squeeze()
            except:
                kwargs['labels'] = kwargs.pop('labels')

        score_name = self.thresholding.get("score_conf", {'score_name':None}).get('score_name')
        kwargs["buffer"]["score_name"] = score_name
        # this is for the case when we want to analysis the score/prob distribution of the unlabeled data
        kwargs["buffer"]["score"] = torch.FloatTensor((float('-inf'),)).repeat(len(task))
        kwargs["buffer"]["prob"] = torch.FloatTensor((float('-inf'),)).repeat(len(task))

        return kwargs
    
    @overrides
    def pseudo_labeling(
        self,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        use_pseudo_labeling:bool,
        soft_label:bool,
        use_ls: bool,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not use_pseudo_labeling:
            return labels
        else:
            with torch.no_grad():
                if y_hat.numel() == 0:
                    return labels
            
                if soft_label:
                    return y_hat.squeeze(1)

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

@Model.register(
    "multi-label-classification-with-scorenn-evaluation",
    constructor="from_partial_objects",
)
@Model.register(
    "multi-label-classification-with-infnet-and-scorenn-evaluation",
    constructor="from_partial_objects_with_shared_tasknn",
)
class MultilabelClassificationWithScoreNNEvaluation(MultilabelClassification):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if self.evaluation_module is None:
            raise ValueError(
                "Evaluation Module can not be none for this model type."
            )

        self.micro_map = MultilabelClassificationMicroAvgPrecision()
        self.average_rank = MultilabelClassificationAvgRank()
        self.mrr = MultilabelClassificationMeanReciprocalRank()
        self.ndcg = (
            MultilabelClassificationNormalizedDiscountedCumulativeGain()
        )
        self.rbo = MultilabelClassificationRankBiasedOverlap()
        self.tasknn_samples_gbi_f1 = MultilabelClassificationF1()
        self.random_samples_gbi_f1 = MultilabelClassificationF1()

    @torch.no_grad()
    def calculate_metrics(  # type: ignore
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        super().calculate_metrics(x, labels, y_hat, buffer, results)
        self.micro_map(y_hat, labels)

        if not self.inference_module.is_normalized:
            y_hat_n = torch.sigmoid(y_hat)
        else:
            y_hat_n = y_hat

        tasknn_samples = self.get_samples(y_hat_n, labels=labels)
        sample_scores = self.score_nn(
            x, tasknn_samples, buffer
        )  # (batch, num_samples+1)
        true_scores = self.oracle_value_function(
            self.unsqueeze_labels(labels), tasknn_samples
        )

        if isinstance(
            self.oracle_value_function, ManhattanDistanceValueFunction
        ):  # invert the scores for l1 ovf
            true_scores = true_scores - true_scores.amin(dim=1, keepdim=True)

        sample_labels = torch.zeros_like(
            sample_scores
        )  # (batch, num_samples+1)
        sample_labels[:, 0] = 1  # set true label index to 1

        # calculate score_nn metrics
        self.average_rank(sample_scores, sample_labels)
        self.mrr(sample_scores, sample_labels)
        self.ndcg(sample_scores, true_scores)
        self.rbo(sample_scores, true_scores)

        random_samples = self.get_samples(y_hat_n, random=True)

        # call evaluation_module on distribution and random samples
        tasknn_gbi_samples, _, loss_ = self.evaluation_module(
            x, labels, buffer, init_samples=tasknn_samples[:, 1:, :], index=0
        )
        self.tasknn_samples_gbi_f1(self.squeeze_y(tasknn_gbi_samples), labels)
        random_gbi_samples, _, loss_ = self.evaluation_module(
            x, labels, buffer, init_samples=random_samples, index=1
        )
        self.random_samples_gbi_f1(self.squeeze_y(random_gbi_samples), labels)

    def get_true_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_true_metrics(reset=reset)
        eval_metrics = {
            "tasknn_samples_gbi_fixed_f1": self.tasknn_samples_gbi_f1.get_metric(
                reset
            ),
            "random_samples_gbi_fixed_f1": self.random_samples_gbi_f1.get_metric(
                reset
            ),
            "micro_map": self.micro_map.get_metric(reset),
            "average_rank": self.average_rank.get_metric(reset),
            "MRR": self.mrr.get_metric(reset),
            "NDCG": self.ndcg.get_metric(reset),
            "RBO": self.rbo.get_metric(reset),
        }

        metrics.update(eval_metrics)

        return metrics

    def get_samples(self, p, random=False, labels=None):
        num_samples = self.num_eval_samples

        if random:
            samples = torch.transpose(
                torch.randint(
                    low=0,
                    high=2,
                    size=(num_samples,) + p.shape,
                    dtype=p.dtype,
                    device=p.device,
                ),
                0,
                1,
            )
        else:
            distribution = torch.distributions.Bernoulli(probs=p)
            samples = torch.transpose(
                distribution.sample([num_samples]), 0, 1
            )  # (batch, num_samples, num_labels)

        # stack labels on top of samples

        if labels is not None:
            samples = torch.hstack([self.unsqueeze_labels(labels), samples])

        return samples