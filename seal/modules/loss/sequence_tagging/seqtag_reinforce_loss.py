from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from seal.modules.loss import Loss, REINFORCELoss
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y,dim=-1)

@Loss.register("sequence-tagging-reinforce")
class SequenceTaggingREINFORCELoss(REINFORCELoss):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
    
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)
