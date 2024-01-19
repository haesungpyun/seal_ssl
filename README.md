# Run Project
    allennlp train [path-to-config] -s [save-folder-name] --include-package seal -f(delete and run)/r(resume) -o [override-setting]

> 여러 실험을 돌리기 위해서는 run_exps.sh 파일 수정 후   
>
    nohup bash run_exps.sh &

# 업데이트 노트

> * seal/dataset_readers/conlll_reader.py   
>   * 사용하지 않음

> * seal/dataset_readers/custom_multitask_scheduler.py   
>    * custom_roundrobin 구현
>    * label data와 unlabeled data를 multitask로 가지고 오며, 한 data 기준으로 전체 data를 모두 mini-batch로 나눌 때까지 다른 data는 cycle 돌며, 계속 batch 생성

> * seal/dataset_readers/multilabel_classification/aapd_reader.py
> * seal/dataset_readers/multilabel_classification/wos_reader.py  
>    * 각 데이터 셋에 맡는 dataset_reader 생성

>* seal/dataset_readers/multilabel_classification/arff_unlabeled_reader.py
>* seal/dataset_readers/multilabel_classification/blurb_genre_collection-unlabeled.py
>* seal/dataset_readers/multilabel_classification/nyt_reader-unlabeled.py
>    * 기존 labeled data를 unlabeled data로 만들기 위해 gt를 조작
>    * 각 데이터 별 dataset_reader 생성

> * seal/dataset_readers/multilabel_classification/conll_srl_reader.py
>    * _read 함수 파일 읽어오는 부분 수정

> * seal/dataset_readers/nyt_unlabeled.py
> * seal/dataset_readers/srl_unlabeled.py
>   * srl unlabeled data용 dataset_reader

> * seal/models/base.py
>   * 각 종 method를 위한 parameter 추가 
>   * unlabeled data의 label 처리를 위한 코드 추가
>   * pseudo_labeling 함수 추가

> * seal/models/multilabel_classification.py
> * seal/models/sequence_tagging.py
> * seal/models/weizmann_horse_seg.py
>   * conconstruct_args_for_forward 함수 추가
>   * pseudo_labeling 함수 overrides

> * seal/modules/loss/loss.py
>   * CombinationUnlabeledLoss 추가 
>       * unlabeled data에 대한 weight 및 multi-task loss 계산

> * seal/modules/loss/multilabel_classification/multilabel_classification_cross_entropy.py
> * seal/modules/loss/sequence_tagging/cross_entropy.py
> * seal/modules/loss/weizmann_horse_seg/tasknn_loss.py
>   * unlabeled data, pseudo label에 대한 behavior 코드 if 문 추가

> * seal/modules/loss/nce_loss.py
>   * nce loss scaling을 위한 코드 추가

> * seal/modules/loss/reinforce_loss.py
>   * REINFORCELoss 추가

> * seal/modules/multilabel_classification_score_nn.py
> * seal/modules/sampler/multilabel_classification/basic.py
>   * 각각 score와 probability를 buffer에 저장하기 위해 상속 풂

> * seal/modules/multilabel_classification_score_nn.py
>   * 각 종 method 위한 parameter 추가

> * seal/modules/sampler/multilabel_classification/inference_net.py
> * seal/modules/sampler/sequence_tagging/inference_net.py
> * seal/modules/sampler/weizmann_horse_seg/inference_net.py
>   * self-trianing 함수 추가

> * seal/modules/sequence_tagging_score_nn.py
> * seal/modules/weizmann_horse_seg_score_nn.py
>   * buffer에 score 저장하는 코드 추가

> * seal/training/callbacks/wandb_subcallbacks.py
>   * register되는 이름 변경

> * seal/training/callbacks/write_read_scores.py
>   * 매 batch 별, epoch 별 score, prob 저장하는 class 및 함수 구현

> * seal/training/trainer/gradient_descent_minimax_trainer.py
>   * pre_training parameter 추가
>   * time_limit 전 종료 코드 추가
>
