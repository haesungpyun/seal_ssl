import copy
import logging
from typing import Any, Optional, Dict, List, Tuple
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from seal.metrics import SegIoU
from seal.modules.loss import Loss
from seal.modules.sampler import Sampler
from seal.training.callbacks.write_read_scores import ThresholdingCallback
from .base import ScoreBasedLearningModel

logger = logging.getLogger(__name__)


@Model.register("weizmann-horse-seg", constructor="from_partial_objects")
@Model.register("seal-weizmann-horse-seg", constructor="from_partial_objects_with_shared_tasknn")
class WeizmannHorseSegModel(ScoreBasedLearningModel):

    def __init__(
        self,
        vocab: Vocabulary,
        sampler: Sampler,
        loss_fn: Loss,
        **kwargs: Any,
    ):
        super().__init__(vocab, sampler, loss_fn, **kwargs)
        self.instantiate_metrics()

    def instantiate_metrics(self) -> None:
        self._seg_iou = SegIoU()

    @overrides
    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(-4) # (b, n=1, c=1, h, w)

    @overrides
    def squeeze_y(self, y: torch.Tensor) -> torch.Tensor:
        return y

    @overrides
    def construct_args_for_forward(self, **kwargs: Any) -> Dict:
        _forward_args = {}
        _forward_args["buffer"] = self.initialize_buffer(**kwargs)
        _forward_args["x"] = kwargs.pop("image")
        _forward_args["labels"] = kwargs.pop("mask")

        if kwargs.get("meta") is None:
            metadata = {}
        else:
            metadata = kwargs.get('meta')
        _forward_args['buffer']['meta'] = metadata     

        task = kwargs.get("task") 
        # Check labeled data
        if task is None:  # if not multi-task
            try:
                metadata[0]["data_type"] # if unlabeled data, should pass data_type from data_reader
                task = []
                for meta in metadata:
                    task.append(meta.get('data_type'))
            except:
               task = ["labeled"] * len(_forward_args["x"])
                
        _forward_args['buffer']['task'] = task   

        if "unlabeled" in task:
            kwargs['labels'] = torch.stack(kwargs.pop('labels')).squeeze()
        
        score_name = self.thresholding.get("score_conf", {'score_name':None}).get('score_name')
        _forward_args["buffer"]["score_name"] = score_name
        _forward_args["buffer"]["score"] = torch.FloatTensor((float('-inf'),)).repeat(len(task))
        _forward_args["buffer"]["prob"] = torch.FloatTensor((float('-inf'),)).repeat(len(task))



        return {**_forward_args, **kwargs}

    @overrides
    def calculate_metrics(
        self,
        x: Any,
        labels: torch.Tensor, # (b, )
        y_hat: torch.Tensor,
        buffer: Dict,
        results: Dict,
        **kwargs: Any,
    ) -> None:
        self._seg_iou(y_hat.detach(), labels.long())

        if not self.training:
            ThresholdingCallback(self.serialization_dir).save_to_storage(
                score_conf=self.thresholding.get('score_conf'),
                buffer=buffer,
                predictions=torch.argmax(y_hat, dim=-3, keepdim=True).tolist(),
                ground_truth=labels.tolist()
            )

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"seg_iou": self._seg_iou.get_metric(reset=reset)}

    @overrides
    def pseudo_labeling(
        self,
        x: Any,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
        buffer: Dict,
        thresholding:Dict,
        use_ls: bool,
        **kwargs: Any,
    ) -> torch.Tensor:
        if thresholding['use_th']:
            x_conf, y_hat_conf, buffer_conf = self.select_confident(x, y_hat, buffer, thresholding)
            if y_hat_conf.numel() == 0:
                return x_conf, labels, y_hat_conf, buffer_conf
        else:
            x_conf, y_hat_conf, buffer_conf = x, y_hat, buffer

        assert buffer_conf.get("meta") is not None
        assert len(y_hat_conf.shape) == 4
        
        pseudo_labels = torch.argmax(y_hat_conf, dim=-3, keepdims=True)
        
        if use_ls: 
            pseudo_labels = pseudo_labels.float()
            pseudo_labels = pseudo_labels*(1- self.alpha) + y_hat_conf.squeeze(1)*self.alpha            
        
        return x_conf, pseudo_labels, y_hat_conf, buffer_conf
    
    @overrides
    def select_confident(
        self,
        x: Any,
        y_hat: torch.Tensor,
        buffer: Dict,
        thresholding: Dict,
        **kwargs: Any,
    ) -> None:
        assert len(y_hat.shape) == 4
        
        max_confidence = y_hat.max(dim=-3)[0].mean(dim=(-1,-2))
        
        if thresholding.get('method') == "local_mean":
            filtered_batch = (max_confidence >= max_confidence.mean()).nonzero(as_tuple=True)[0]
        
        elif thresholding.get('method') == "local_median":
            filtered_batch = (max_confidence >= max_confidence.median()).nonzero(as_tuple=True)[0]
        
        else: # self.thresholding.method == "score"
            max_confidence = buffer.get(buffer.get('score_name'))
            threshold = thresholding.get('score_conf').get('threshold')
            if type(threshold) == int: 
                threshold = torch.tensor([threshold], device=max_confidence.device)
            else:
                threshold = threshold.to(max_confidence.device)
            filtered_batch = (max_confidence >= threshold).nonzero(as_tuple=True)[0].unique()
        
        y_hat_conf = y_hat[filtered_batch]

        x_conf = x[filtered_batch]
        
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
