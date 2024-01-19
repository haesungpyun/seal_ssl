from typing import List, Tuple, Union, Dict, Any, Optional
from seal.modules.loss import REINFORCELoss, Loss
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("multi-label-reinforce")
class MultiLabelREINFORCEcoreLoss(REINFORCELoss):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)