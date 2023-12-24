from typing import List, Tuple, Union, Dict, Any, Optional
from typing_extensions import override
from .score_nn import ScoreNN
import torch


@ScoreNN.register("multi-label-classification")
class MultilabelClassificationScoreNN(ScoreNN):
    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        label_scores = self.task_nn(
            x, buffer
        )  # unormalized logit of shape (batch, num_labels)
        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy

    @override
    def forward(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples or 1, ...).
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None
        local_score = self.compute_local_score(x, y, buffer, **kwargs)

        if local_score is not None:
            score = local_score

        global_score = self.compute_global_score(y, buffer, **kwargs)  # type: ignore

        if global_score is not None:
            if score is not None:
                score = score + global_score
            else:
                score = global_score

        buffer["score"] = (local_score + global_score)

        return score  # (batch, num_samples)
    

