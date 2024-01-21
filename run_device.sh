#!/bin/bash
#!/usr/bin/env python

source ~/.bashrc
eval "$(conda shell.bash hook)"
# 가상 환경 활성화
conda activate seal_ad

# current directory 설정 serial directory 저장 및 config 불러오는 path 설정에 사용
cur_dir=$(pwd)
# CUDA_VISIBLE_DEVICES 에 사용되는 gpu 번호
cuda_num=0

# 실행 명령어
# base
# bash run_device.sh 1"serial_dir" 2"task" 3"data" 4"base" 5"base batch size" 
# bahs run_device.sh '' mlc bibtex base 32

# other training (batch size는 effective batch size 아닌 accumulate 되는 하나의 batch size를 기준으로 작성)
# bash run_device.sh 1"serial_dir" 2"task" 3"data" 4"'<score net training1 2 3>'" 5"ls alpha" 6"<task net training1 2 3>" 7"ls alpha"
# 8"unlab weight" 9"labels batch size" 10"unlabels batch size" 11"grad accum"
# bash run_device.sh '' mlc bibtex 'soft_label' '' 'label_smoothing' '0.3' '1.1' 4 32 4

# Score net의 loss 계산 방법 별 config (override용도)
declare -A score_training
# score hard label
score_training[hard_label]="'model.use_pseudo_labeling':true"
# score soft label
score_training[soft_label]="'model.use_pseudo_labeling':true,'model.soft_label':true"
# score ls
score_training[label_smoothing]="'model.use_pseudo_labeling':true,'model.label_smoothing.use_ls':true,'model.soft_label':false"
# score thres
score_training[thresholding]="'model.thresholding.use_th':true,'model.thresholding.method':'score',
'model.thresholding.score_conf.cut_type':'discrim','model.thresholding.score_name':'score','model.thresholding.score_conf.threshold':-100"

# Task net의 loss 계산 방법 별 config (override용도)
declare -A task_training
# task hard label
task_training[hard_label]="'model.inference_module.use_self_training':true"
# task soft label
task_training[soft_label]="'model.inference_module.use_self_training':true,'model.inference_module.task_soft_label':true"
# score ls
task_training[label_smoothing]="'model.inference_module.use_self_training':true,'model.inference_module.task_label_smoothing.use_ls':true"
# score thres
task_training[thresholding]="'model.inference_module.task_thresholding.use_th':true,'model.inference_module.task_thresholding.method':'score',
'model.inference_module.task_thresholding.score_conf.cut_type':'discrim','model.inference_module.task_thresholding.score_name':'score',
'model.inference_module.task_thresholding.score_conf.threshold':'-100"

# resume 되는 경우
if [ "$1" ]; then
    serial_dir=$1
    config="${cur_dir}/${serial_dir}/config.json"
    override="'trainer.cuda_device':0"
    option='r'  
    
    if [[ "$2" == *"multi"* ]]; then	data='multi'
	 # ['multi', 'nyt', 'onto', 'onto_multi']
    else								data='multi';	fi
else
    option='f'
    
	if [ "$2" ]; then   task=$2
    else                task='srl';   fi
    	# srl의 경우 data setting이 3가지로 실험함 각 config 파일을 통해서 data 분할한 방식 확인 가능
	if [ "$3" ]; then   data=$3     # datasetting ['multi', 'multi2', 'nyt']
    else                data='multi';   fi    

    if [ "$4" ]; then   training=$4
    else                training='base';    fi

    if [[ "$training" == *"base"* ]]; then
        if [ "$5" ]; then	
            serial_dir="${task}_${data}_base"
            config="${cur_dir}/configs_da/${task}/${data}/${training}.json" 
	    # 다른 method와 실험 setting을 동일하게 하기 위해서 base method 또한 accumulation 시킴
            overrides="{'trainer.num_gradient_accumulation_steps':2"
            lab_b=$(($5/2))
            serial_dir+="_l${5}"
            overrides+=",'data_loader.batch_size':${lab_b}"
        fi

    else
    	# config에 맞는 directory 이름 생성
        serial_dir="${task}_${data}_s"
	# 초기 config는 nodistil을 기준으로 함
        config="${cur_dir}/configs_da/${task}/${data}/nodistil.json"   

 	# Score net의 각 method에 대해 override 할 config 추가하기 위한 for loop
        overrides="{'trainer.cuda_device':0"
        for method in $training
        do  
            if [ "$method" == 'nodistil' ]; then
                serial_dir+="_nodistil"
                break
            fi
            overrides+=",${score_training[${method}]}"
            serial_dir+="_${method}"
        done

 	# label smoothing의 비율이 있다면 label smoothing config 추가
        if [ "$5" ]; then
            # 0.1 0.3 0.5 0.7 0.9
            alpha=$5
            overrides+=",'model.label_smoothing.alpha':${alpha}"
            serial_dir+="_alpha${alpha}"            
        fi
	
        if [ "$6" ]; then   
            training=$6
            serial_dir+="_t"
        else
            training='';    fi

	# Task net 각 method에 대한 override 할 config 추가하기 위한 for loop
        for method in $training
        do  
            if [ "$method" == 'nodistil' ]; then
                    serial_dir+="_nodistil"
                    break
            fi
	    if [ "$method" == 'reinforce' ]; then
                config="${cur_dir}/configs_da/${task}/${data}/reinforce.json"
                serial_dir+="_${method}"
                break
            fi
            overrides+=",${task_training[${method}]}"
            serial_dir+="_${method}"
        done

 	# label smoothing의 비율이 있다면 label smoothing config 추가
        if [ "$7" ]; then
            # 0.1 0.3 0.5 0.7 0.9
            alpha=$7
            overrides+=",'model.inference_module.task_label_smoothing.alpha':${alpha}"
            serial_dir+="_alpha${alpha}"            
        fi   

 	# unlabeled data에 대해서 weight를 주는데 해당 비율을 받음
        serial_unwei=""
        if [ "$8" ]; then
            # 0.1 0.25 0.5
            unlabwei=$8
            overrides+=",'model.loss_fn.loss_weights.unlabeled':${unlabwei}"
            serial_unwei+="_unwei${unlabwei}"
            if [ "$training" ]; then
                if [ "$training" == 'reinforce' ]; then
                    overrides+=",'model.inference_module.loss_fn.loss_weights.unlabeled':[${unlabwei}"]
                    serial_unwei+="_reinforce${unlabwei}"
                else
                    overrides+=",'model.inference_module.loss_fn.loss_weights.unlabeled':[${unlabwei},${unlabwei}]"
                    serial_unwei+="_selfwei${unlabwei}"
                fi
            else
                overrides+=",'model.inference_module.loss_fn.loss_weights.unlabeled':[${unlabwei},0]"
            fi
        else
            overrides+=",'model.inference_module.loss_fn.loss_weights.unlabeled':[0.1,0],'model.loss_fn.loss_weights.unlabeled':0.1"
            serial_unwei+="_unwei0.1"
        fi  

        serial_dir+=$serial_unwei

        shift
        shift

	# grad accumulation
        if [ "$9" ]; then
            accum=$9
            overrides+=",'trainer.num_gradient_accumulation_steps':${accum}"
	    # labeled data, unlabeled data가 번갈아 나오기 때문에 2로 나눠줌
     	    # accumulation이 2라는 것은 labeled batch, unlabeled batch 각 loss를 한 번에 update 하겠다는 것
	    # effective labeled & unlabeled batch 각 32로 하고 싶으나, memory에 8까지 밖에 안 올라가는 경우
	    # batch size 8로 setting & accumulation 8 (2 * 4)
            accum=$(($accum/2))
        else
            accum=1
            overrides+=",'trainer.num_gradient_accumulation_steps':2"
        fi

        if [ "$7" ]; then	
            # 4 8 16
	    # effective batch size 계산
            lab_b=$(($7*$accum))
            serial_dir+="_l${lab_b}"
            overrides+=",'data_loader.scheduler.batch_size.labeled':$7"
        fi
    
	    if [ "$8" ]; then    
            # 4 8 16 32 64
            unlab_b=$(($8*$accum))
            serial_dir+="_u${unlab_b}"
            overrides+=",'data_loader.scheduler.batch_size.unlabeled':$8"
        fi	    
    fi
fi

overrides+="}"

echo $config
echo $serial_dir
echo $option
echo $overrides
echo "$cuda_num allennlp train $config -s $serial_dir --include-package seal -$option -o $overrides"

if [ $option == 'f' ]; then
    CUDA_VISIBLE_DEVICES=$cuda_num allennlp train $config \
                -s $serial_dir \
                --include-package seal \
                -$option \
                -o $overrides
else
    CUDA_VISIBLE_DEVICES=$cuda_num allennlp train $config \
                -s $serial_dir \
                --include-package seal \
                -$option
fi

state=$?
echo $state
if [[ $state -eq 124 ]]; then # if [[ $state -eq 124 || $state -eq 1]] loop forever
    if [[ $serial_dir == *'_wo_pre' ]]; then
        exit
    fi
    new=$(nohup bash run_endless.sh $cuda_num $serial_dir )
    echo $new
elif [ $state -eq 0 ]; then
    new=$(nohup bash run_eval_da.sh $cuda_num $serial_dir)
    echo $new
fi
