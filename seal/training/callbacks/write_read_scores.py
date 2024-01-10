import copy
import json
import logging
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import allennlp.nn.util as nn_util
import numpy as np
from allennlp.data.tokenizers import Token
import torch
from allennlp.common.tqdm import Tqdm
from allennlp.data import TensorDict
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.multitask_scheduler import _chunked_iterator
from allennlp.data.fields import (
    LabelField, MetadataField,
    SequenceLabelField, TensorField, TextField
)
from allennlp.data.instance import Instance
from allennlp.evaluation.serializers.serializers import SimpleSerializer
from allennlp.models.archival import load_archive
from allennlp.training.callbacks import TrainerCallback
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from pyparsing import Generator
from sklearn.linear_model import LogisticRegression

from seal.common import ModelMode
from seal.dataset_readers.custom_multitask_scheduler import custom_roundrobin

logger = logging.getLogger(__name__)

@TrainerCallback.register("thresholding")
class ThresholdingCallback(TrainerCallback):
    storage = defaultdict(list)
    storage['epoch'] = 0
    '''
        cls.storage = {
            'epoch': int,
            'cut_type': str,   # ['all', 'quantile', 'discrim']
            'score_name':str,   # ['score', 'prob']
            'score': List[torch.Tensor],
            'prob': List[torch.Tensor],
            'score_pos': List[torch.Tensor],
            'score_neg': List[torch.Tensor],
            'prob_pos': List[torch.Tensor],
            'prob_neg': List[torch.Tensor]
        }
    '''
    def __init__(self, serialization_dir: str) -> None:
        self.serialization_dir = serialization_dir
        self.trainer: Optional['GradientDescentTrainer'] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def on_start(self, trainer: GradientDescentTrainer, is_primary: bool = True, **kwargs) -> None:
        if ThresholdingCallback(self.serialization_dir).storage.get('threshold') is not None:
            try:
                threshold_dict = trainer.model.thresholding.as_dict()
            except:
                threshold_dict = trainer.model.thresholding

            threshold = ThresholdingCallback(self.serialization_dir).storage['threshold']
            # import pdb
            # pdb.set_trace()
            # threshold_dict['score_conf']['threshold'] = threshold.to(self.device)
            try:
                threshold_dict['score_conf']['threshold'] = threshold
            except:
                return
            logger.info(f"\tRestored Threshold:\t{threshold}\n")
        if trainer.filter_unlab['use_fu']:
            self.validate_save_unlabeled(trainer)
            
    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[TensorDict],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool = True,
        batch_grad_norm: float or None = None,
        **kwargs
    ) -> None:

        if is_training:
            store_dict = defaultdict(list)
            
            dir_path = os.path.join(self.serialization_dir, './scores/')
            if (not os.path.isfile(dir_path)) and (not os.path.exists(dir_path)):
                os.makedirs(dir_path)     
            
            for output in batch_outputs:
                if output.get('score') is None or output.get('prob') is None:
                    continue
                prob = output.get('prob')[output.get('prob') > float('-inf')]
                score = output.get('score')[output.get('score') > float('-inf')]
                store_dict['prob'].extend(prob)
                store_dict['score'].extend(score)
            
            store_dict['prob'] = torch.stack(store_dict['prob'])
            store_dict['score'] = torch.stack(store_dict['score'])
                
            for key in store_dict.keys():
                file_path = os.path.join(dir_path, f'./epoch_{epoch}_training_{key}.txt')
                with open(file_path, 'ab') as f:
                    f.write(b'\n')
                    np.savetxt(f, store_dict[key].cpu().numpy())
                                
    def on_epoch(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs
    ) -> Any:
        
        storage = ThresholdingCallback.storage
        storage['epoch'] = epoch
        self.trainer = trainer        
        try:
            threshold_dict = trainer.model.thresholding.as_dict()
        except:
            threshold_dict = trainer.model.thresholding
        use_scaling = self.trainer.model.loss_fn.loss_scaling['use_scaling']
        scaler_type = self.trainer.model.loss_fn.loss_scaling.get('scaler_type')
        
        if threshold_dict.get('method') != 'score':
            logger.info('!!!! Skip Mode! Do Not Calculate threshold !!!!')
            self.write_reset_score()
            return False
        
        score_name = threshold_dict['score_conf']['score_name']
        cut_type = threshold_dict['score_conf']['cut_type']
        
        storage['threshold'] = threshold_dict['score_conf']['threshold']
        
        if cut_type == 'discrim' :
            logger.info('!!! Calculate Threshold from pos, neg dist using Logistic Regression !!!')
           
            pos_list = storage[f'{score_name}_pos']
            neg_list = storage[f'{score_name}_neg']

            if len(pos_list) == 0 or len(neg_list) == 0:
                logger.info(f"!!!!!!!!!! Empty score in given pos lenth  {len(pos_list)}, neg lenth {len(neg_list)} !!!!!!!!!!")
                self.write_reset_score()
                return False
            
            pos, neg = self.sample_list_to_tensor(pos_list, neg_list, num_sample=None)
                            
            x = torch.cat((pos, neg))
            y = torch.cat((torch.zeros(pos.shape[0]), torch.ones(neg.shape[0])))    # 0 -> pos // 1 -> neg
        
            classifier = LogisticRegression()
            classifier.fit(x.cpu(),y.cpu())
            
            b, w = classifier.intercept_, classifier.coef_
            decision_boundary = torch.from_numpy(-b/w).to(self.device)
            threshold_dict['score_conf']['threshold'] = decision_boundary
            
            logger.info(f'\tThreshold(decision boundary):\t{decision_boundary}\n')
                   
            if use_scaling:
                params = [0, 1, 0]
                prob_pos_list = storage[f'prob_pos']
                prob_pos = self.list_to_tensor(prob_pos_list)
                if scaler_type == 'robust':
                    params = [prob_pos.quantile(0.5), prob_pos.quantile(0.75), prob_pos.quantile(0.25)]
                elif scaler_type == 'standard':
                    params = [prob_pos.mean(), prob_pos.std(), 0]
                elif scaler_type == 'min_max':
                    params = [prob_pos.min(), prob_pos.max(), prob_pos.min()]
                self.trainer.model.loss_fn.loss_scaling['params'] = params
                
            # Erase scores from previous epochs for refinement
            self.write_reset_score()
            storage['threshold'] = decision_boundary
            
            return True

        else:
            logger.info('Calculate Threshold from Scores using Quantile')
            
            pos_list = storage[f'{score_name}_pos']

            if len(pos_list) == 0:
                logger.info(f"!!!!!!!!!! Empty score in given pos lenth  {len(pos_list)}  !!!!!!!!!!")
                self.write_reset_score()
                return False
            
            pos = torch.stack(pos_list).to(self.device)
                       
            if pos.numel() == 0:
                self.write_reset_score()
                return False
            
            threshold = pos.quantile(threshold_dict['score_conf']['quantile'])
            threshold_dict['score_conf']['threshold'] = threshold
            
            logger.info(f"\tThreshold(quantile={threshold_dict['score_conf']['quantile']}):\t{threshold}\n")
            
            if use_scaling:
                prob_pos_list = storage[f'prob_pos']
                prob_pos = self.list_to_tensor(prob_pos_list)
                if scaler_type == 'robust':
                    params = [prob_pos.quantile(0.5), prob_pos.quantile(0.75), prob_pos.quantile(0.25)]
                elif scaler_type == 'standard':
                    params = [prob_pos.mean(), prob_pos.std(), 0]
                elif scaler_type == 'min_max':
                    params = [prob_pos.max(), ]
                elif scaler_type == 'mad':
                    params = [prob_pos.quantile(0.5), torch.abs(prob_pos-prob_pos.quantile(0.5)).quantile(0.5), 0]
                self.trainer.model.loss_fn.loss_scaling['params'] = params
            
            # Erase scores from previous epochs for refinement
            self.write_reset_score()
            storage['threshold'] = threshold
            
            return True
    
    def write_reset_score(
        self,
        is_training=False,
        reset = True,
        **kwargs:Any         
    ):  
        '''
        cls.storage = {
            'epoch': int,
            'cut_type': str,   # ['all', 'quantile', 'discrim']
            'score_name':str,   # ['score', 'prob']
            'score': List[torch.Tensor],
            'prob': List[torch.Tensor],
            'score_pos': List[torch.Tensor],
            'score_neg': List[torch.Tensor],
            'prob_pos': List[torch.Tensor],
            'prob_neg': List[torch.Tensor]
        }
        '''
        storage = ThresholdingCallback.storage

        dir_path = os.path.join(self.serialization_dir, './scores/')
        if (not os.path.isfile(dir_path)) and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)           
                            
        # ep = storage['epoch']
        # for key in storage:
        #     value = storage[key]
        #     if type(value) == list:
        #         file_path = os.path.join(dir_path, f'epoch_{ep}_valid_{key}.pt')
        #         value = self.list_to_tensor(value)
        #         torch.save(value, file_path)
        #         if reset: 
        #             storage[key] = []
        #     else:
        #         file_path = os.path.join(dir_path, f'epoch_{ep}_valid_{key}.pt')
        #         torch.save(value, file_path)

    def write_validation_score(
        self,
        is_training=False,
        reset = True,
        **kwargs:Any         
    ):  
        '''
        cls.storage = {
            'epoch': int,
            'cut_type': str,   # ['all', 'quantile', 'discrim']
            'score_name':str,   # ['score', 'prob']
            'score': List[torch.Tensor],
            'prob': List[torch.Tensor],
            'score_pos': List[torch.Tensor],
            'score_neg': List[torch.Tensor],
            'prob_pos': List[torch.Tensor],
            'prob_neg': List[torch.Tensor]
        }
        '''
        storage = ThresholdingCallback.storage

        dir_path = self.serialization_dir
        
        if (not os.path.isfile(dir_path)) and (not os.path.exists(dir_path)):
            os.makedirs(dir_path)           
                                      
        file_path = os.path.join(dir_path, f'valid_dist.pt')
        torch.save(storage['score_dist'], file_path)
        storage['score_dist'] = []
        
    def list_to_tensor(
        self,
        lists:list
    ):  
        assert type(lists)==list
        if len(lists) == 0:
            return torch.zeros(0, 1).to(self.device)
        elif type(lists[0]) != torch.Tensor:
            return lists
        else:
            return torch.stack(lists).to(self.device)
    
    def sample_list_to_tensor(
        self,
        pos_list,
        neg_list,
        num_sample=None,
    ):
        logger.info('Convert Pos/Neg List to Tensor')      
       
        pos_samples = torch.stack(pos_list).to(self.device)
        neg_samples = torch.stack(neg_list).to(self.device)
        
        # pos_q3, pos_q1 = pos_samples.quantile(0.75), pos_samples.quantile(0.25)
        # pos_iqr = pos_q3 - pos_q1
        
        # neg_q3, neg_q1 = neg_samples.quantile(0.75), neg_samples.quantile(0.25)
        # neg_iqr = neg_q3 - neg_q1
        
        # pos_samples = pos_samples[(pos_q1 - 1.5*pos_iqr <= pos_samples) & (pos_samples <= pos_q3 + 1.5*pos_iqr), None]
        # neg_samples = neg_samples[(neg_q1 - 1.5*neg_iqr <= neg_samples) & (neg_samples <= neg_q3 + 1.5*neg_iqr), None]
        
        if num_sample is None:
            num_sample = min(len(pos_samples), len(neg_samples))
        
        pos_samples = random.sample(list(pos_samples), num_sample)
        neg_samples = random.sample(list(neg_samples), num_sample)
        
        return torch.stack(pos_samples).to(self.device),\
            torch.stack(neg_samples).to(self.device)
        
    @classmethod
    def save_to_storage(
        cls,
        buffer: Dict,
        score_conf:Dict = None,
        predictions: List[List[Union[str, int]]] = None,
        ground_truth:List[List[Union[str, int]]] = None,
        **kwargs:Any        
    ):  
        assert len(predictions) == len(ground_truth)
        assert type(predictions) == type(ground_truth) == list
        
        if score_conf is not None:
            cls.storage['cut_type'] = score_conf.get('cut_type', 'discrim')
            cls.storage['score_name'] = score_conf.get('score_name', 'score')
        
        score = buffer.get('score')
        prob = buffer.get('prob')
        
        cls.storage['score'].extend(score)
        cls.storage['prob'].extend(prob)

        cls.storage['score_dist'].extend(buffer.get('score_dist', []))

        exact_match = kwargs.get('exact_match', True)

        pos_idx, neg_idx = [], []
        for i, (pred, gold) in enumerate(zip(predictions, ground_truth)):
            if exact_match:
                if pred==gold:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            else:
                true_positive = sum([1 if (pred[i] == gold[i]) and (pred[i] == 1) else 0 for i in range(len(pred))])
                false_negative = sum([1 if (pred[i] != gold[i]) and (gold[i] == 1) else 0 for i in range(len(pred))])
                false_positive = sum([1 if (pred[i] != gold[i]) and (pred[i] == 1) else 0 for i in range(len(pred))])
                precision = true_positive / (true_positive+false_positive if (true_positive+false_positive) != 0 else 1)
                recall = true_positive / (true_positive+false_negative if (true_positive+false_negative) != 0 else 1)
                f1 = (2 * precision * recall) / (precision+recall if (precision+recall) != 0 else 1)
                if f1 > 0.8:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i) 
        pos_idx, neg_idx = torch.LongTensor(pos_idx), torch.LongTensor(neg_idx)
        
        
        score_pos, score_neg = score[pos_idx,], score[neg_idx,]
        prob_pos, prob_neg = prob[pos_idx,], prob[neg_idx,]
        
        cls.storage['score_pos'].extend(score_pos)
        cls.storage['score_neg'].extend(score_neg)
        cls.storage['prob_pos'].extend(prob_pos)
        cls.storage['prob_neg'].extend(prob_neg)
        # cls.storage['pos_words'].extend([buffer['meta'][i]['words'] for i in pos_idx])
        # cls.storage['neg_words'].extend([buffer['meta'][i]['words'] for i in neg_idx])


    def state_dict(self) -> Dict[str, Any]:
        return {
            'threshold': ThresholdingCallback(self.serialization_dir).storage['threshold']
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ThresholdingCallback(self.serialization_dir).storage['threshold'] = state_dict['threshold']
        
    def validate_save_unlabeled(self, trainer):
        
        archive_file="./model.tar.gz"
        weights_file=f"/{trainer.filter_unlab['model_to_run']}/best.th"

        directory = f"{trainer._serialization_dir}/filtered_unlabeled"
        pos_predictions_path = f"{directory}/prediction_pos.jsonl"
        neg_predictions_path = f"{directory}/prediction_neg.jsonl"
        
        if not os.path.exists(directory):
            os.mkdir(directory)
        else:
            return True    
        
        f = open(pos_predictions_path, 'w')
        f.close()
        f = open(neg_predictions_path, 'w')
        f.close()
        
        pos_predictions_file = Path(pos_predictions_path).open("w", encoding="utf-8")
        neg_predictions_file = Path(neg_predictions_path).open("w", encoding="utf-8")
        
        archive = load_archive(
            archive_file=archive_file,
            weights_file=weights_file,
            cuda_device=-1 if trainer.data_loader.cuda_device == torch.device('cpu') else 0,
            overrides='{\"model.inference_module.type":"sequence-tagging-inference-net-normalized-custom"}'
        )
        model = archive.model
        model.eval()
            
        unlabeled_instance = defaultdict(Generator)
        unlabeled_instance['unlabeled'] = trainer.data_loader._get_instances_for_epoch()['unlabeled']
        
        chunked_iterators= {
            dataset: _chunked_iterator(iterator,trainer._validation_data_loader.batch_sampler.batch_size, True)
            for dataset, iterator in unlabeled_instance.items()
        }
         
        batch_generator = iter(
            nn_util.move_to_device(
                Batch(instances).as_tensor_dict(),
                -1 if trainer.data_loader.cuda_device is None else trainer.data_loader.cuda_device,
            )
            for instances in custom_roundrobin(chunked_iterators)
        )
        
        len_data_loader = len(trainer.data_loader._loaders['unlabeled']) / trainer._validation_data_loader.batch_sampler.batch_size
        num_training_batches = math.ceil(len_data_loader)
        batch_generator_tqdm = Tqdm.tqdm(batch_generator, total=num_training_batches)
        
        batch_serializer = SimpleSerializer()
        postprocessor_fn_name = "make_output_human_readable"
        model_postprocess_function = getattr(model, postprocessor_fn_name, None)
        
        with torch.no_grad():
            model.eval()
            loss_count = 0
            total_loss = 0.0
            total_weight = 0.0
            
            for batch in batch_generator_tqdm:
                output_dict = model(**batch, mode=ModelMode.UPDATE_TASK_NN)
                loss = output_dict.get("loss")
                
                metrics = model.get_metrics()
                
                if loss is not None:
                    loss_count += 1                
                    weight = 1.0
                    total_weight += weight
                    total_loss += loss.item() * weight
                    metrics["loss"] = total_loss / total_weight
                description = (
                    ", ".join(
                        [
                            "%s: %.2f" % (name, value)
                            for name, value in metrics.items()
                            if not name.startswith("_")
                        ]
                    )
                    + " ||"
                )
                batch_generator_tqdm.set_description(description, refresh=False)
                
                (
                    word_tags, _, _, _,
                ) = model.constrained_decode(
                    output_dict['y_pred'],
                    batch['tokens']['tokens']['mask'],
                    [x["offsets"] for x in batch['metadata']]
                )
                
                pos_id = (output_dict['y_pred'].max(-1)[0].mean(-1) >= 0.99).nonzero(as_tuple=True)[0].unique()
                neg_id = (output_dict['y_pred'].max(-1)[0].mean(-1) < 0.99).nonzero(as_tuple=True)[0].unique()
                
                assert len([i for i in pos_id if i in neg_id]) == 0
                assert len(pos_id) + len(neg_id) == len(output_dict['y_pred'])
                
                pids, nids = [], []
                for idx in torch.arange(output_dict['y_pred'].shape[0]):
                    # tmp_tokens = defaultdict(dict)
                    # for key in tokens['tokens'].keys():
                    #     value = tokens['tokens'].get(key)
                    #     tmp_tokens['tokens'][key] = value[idx][:seq_len]
                    
                    tmp_batch = defaultdict(dict)
                    tmp_batch['metadata'] = batch['metadata'][idx]
                    # for key in batch.keys():
                    #     value = batch.get(key)
                    #     if type(value) == torch.Tensor:
                    #         tmp_batch[key] = value[idx][:seq_len]
                    #     elif type(value) == list:
                    #         tmp_batch[key] = value[idx]
                    #     else:
                    #         continue
                
                    tmp_batch['metadata']['pseudo_tags'] = word_tags[idx]
                    # tmp_batch['tokens'] = tmp_tokens
                    
                    if idx in pos_id:
                        pos_predictions_file.write(
                            batch_serializer(
                                tmp_batch,
                                {},
                                trainer.data_loader,
                                output_postprocess_function=model_postprocess_function,
                            )
                            + "\n"
                        )
                        pids.append(idx)
                    else:
                        neg_predictions_file.write(
                            batch_serializer(
                                tmp_batch,
                                {},
                                trainer.data_loader,
                                output_postprocess_function=model_postprocess_function,
                            )
                            + "\n"
                        )
                        nids.append(idx)
                        
                assert pids == [i.item() for i in pos_id] and nids == [i.item() for i in neg_id]
                assert len(pids) == len(pos_id) and len(nids) == len(neg_id)
                
            pos_predictions_file.close()
            neg_predictions_file.close()  
           
        def make_instance(line):
            tags = line['metadata']['pseudo_tags']
            verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
            tokens = [Token(t) for t in line['metadata']['words']]
            return trainer.data_loader.readers['unlabeled'].text_to_instance(tokens, verb_indicator, tags, False)
                   
        def get_instances():    
            instances = []
            with open(pos_predictions_path, encoding='utf-8', mode='r') as f:
                for line in f:
                    instance = make_instance(json.loads(line))
                    instance.index_fields(trainer.data_loader._loaders['unlabeled']._vocab)
                    instances.append(instance)
            return instances
                    
        trainer.data_loader._loaders['unlabeled']._instances = get_instances()