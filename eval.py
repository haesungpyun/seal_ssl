import json
import shutil
import sys
import json
from json import JSONDecodeError
from pathlib import Path
from copy import deepcopy
import os

from allennlp.common.util import prepare_environment
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.evaluation import Evaluator
from allennlp.common.util import import_module_and_submodules

from pathlib import Path
import torch

from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, int_to_device
from allennlp.nn import util as nn_util
from allennlp.data import DataLoader
from allennlp.evaluation.serializers.serializers import Serializer, SimpleSerializer

dir_path = './result'
for model in os.listdir(dir_path):
    model_to_run = os.path.join(dir_path, model)
    for data in ['unlabeled', 'labeled']:
        print(model_to_run)
        print(data)
        if not os.path.isdir(model_to_run) or not os.path.exists(model_to_run + "/metrics.json"):
            continue
        if not model_to_run.endswith('mlc_cal500_s_nodistil_t_nodistil_unwei0.1_selfwei0.1_l8_u8'):
            continue
        # raise ValueError
        directory = f"{model_to_run}/{data}/"
        serialization_dir = os.path.join(os.getcwd(), f'./{directory}/')
        if (not os.path.isfile(serialization_dir)) and (not os.path.exists(serialization_dir)):
            os.makedirs(serialization_dir)       
            os.makedirs(os.path.join(serialization_dir, "./scores"))

        archive_file=f"./{model_to_run}"
        weights_file=f"./{model_to_run}/best.th"
        # input_file="./data/conll-2012-da/v12/data/development"
        cuda_device=0

        import_module_and_submodules('seal')

        archive = load_archive(
                archive_file=archive_file,
                weights_file=weights_file,
                cuda_device=cuda_device,
        )

        config = deepcopy(archive.config)

        prepare_environment(config)

        model = archive.model
        model.eval()

        evaluator_params = config.pop("evaluation", {})
        evaluator_params["cuda_device"] = -1
        evaluator = Evaluator.from_params(evaluator_params)

        dataset_reader = archive.validation_dataset_reader

        input_file = config.as_dict()['data_loader']['data_path'][data]

        try:
            # Try reading it as a list of JSON objects first. Some readers require
            # that kind of input.
            evaluation_data_path_list = json.loads(f"[{input_file}]")
        except JSONDecodeError:
            evaluation_data_path_list = input_file.split(",")

        metrics_output_file = f"./{directory}/metric.json"
        predictions_output_file = f"./{directory}/prediction.json"

        f = open(metrics_output_file, 'w')
        f.close()

        f = open(predictions_output_file, 'w')
        f.close()

        output_file_list = [
            p.parent.joinpath(f"{p.stem}.outputs") for p in map(Path, evaluation_data_path_list)
        ]
        predictions_output_file_list = [
            p.parent.joinpath(f"{p.stem}.preds") for p in map(Path, evaluation_data_path_list)
        ]

        all_metrics = {}

        batch_serializer = SimpleSerializer()
        postprocessor_fn_name = "make_output_human_readable"

        for index, evaluation_data_path in enumerate(evaluation_data_path_list):
            
            if isinstance(evaluation_data_path, str):
                eval_file_name = Path(evaluation_data_path).stem
            else:
                eval_file_name = str(index)
            
            data_loader_params = config.get("validation_data_loader", None)

            if data_loader_params is None:
                data_loader_params = config.get("data_loader")
            data_loader = DataLoader.from_params(
                params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
            )

            data_loader.index_with(model.vocab)

            check_for_gpu(cuda_device)
            data_loader.set_target_device(int_to_device(cuda_device))
            metrics_output_file = Path(metrics_output_file) if metrics_output_file is not None else None
            
            if predictions_output_file is not None:
                predictions_file = Path(predictions_output_file).open("w", encoding="utf-8")
            else:
                predictions_file = None  # type: ignore

            model_postprocess_function = getattr(model, postprocessor_fn_name, None)

            with torch.no_grad():
                model.eval()

                iterator = iter(data_loader)
                generator_tqdm = Tqdm.tqdm(iterator)
                # Number of batches in instances.
                batch_count = 0
                # Number of batches where the model produces a loss.
                loss_count = 0
                # Cumulative weighted loss
                total_loss = 0.0
                # Cumulative weight across all batches.
                total_weight = 0.0

                for batch in generator_tqdm:
                    batch_count += 1
                    batch = nn_util.move_to_device(batch, cuda_device)
                    
                    from seal.common import ModelMode
                    output_dict = model(**batch, mode=ModelMode.UPDATE_TASK_NN)
                    loss = output_dict.get("loss")

                    metrics = model.get_metrics()

                    if loss is not None:
                        loss_count += 1
                    
                        weight = 1.0

                        total_weight += weight
                        total_loss += loss.item() * weight
                        # Report the average loss so far.
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
                    generator_tqdm.set_description(description, refresh=False)

                    # TODO(gabeorlanski): Add in postprocessing the batch for token
                    #  metrics
                    if predictions_file is not None:
                        predictions_file.write(
                            batch_serializer(
                                batch,
                                output_dict,
                                data_loader,
                                output_postprocess_function=model_postprocess_function,
                            )
                            + "\n"
                        )
                    # if batch_count == int(len(data_loader)*0.1):
                    #     break
                from seal.training.callbacks.write_read_scores import ThresholdingCallback
                
                ThresholdingCallback(serialization_dir).write_validation_score()
                
                if predictions_file is not None:
                    predictions_file.close()

                final_metrics = model.get_metrics(reset=True)
                if loss_count > 0:
                    # Sanity check
                    if loss_count != batch_count:
                        raise RuntimeError(
                            "The model you are trying to evaluate only sometimes produced a loss!"
                        )
                    final_metrics["loss"] = total_loss / total_weight

                if metrics_output_file is not None:
                    dump_metrics(str(metrics_output_file), final_metrics, log=True)


            for name, value in final_metrics.items():
                    if len(evaluation_data_path_list) > 1:
                        key = f"{eval_file_name}_"
                    else:
                        key = ""
                    all_metrics[f"{key}{name}"] = value
