#!/bin/bash
#!/usr/bin/env python

# source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate seal_ad
# cd ~/seal_ex

cur_dir="/home/haesung/seal-neurips-2022"
cuda_num=0

# base
# 1"serial_dir" 2"task" 3"data" 4"base" 5"base batch size" 

# other training
# 1"serial_dir" 2"task" 3"data" 4"'<score net training1 2 3>'" 5"ls alpha" 6"<task net training1 2 3>" 7"ls alpha"
# 8"unlab weight" 9"labels batch size" 10"unlabels batch size" 11"grad accum"
# bash run_device.sh '' mlc bibtex 'soft_label' '' 'label_smoothing' '0.3' '1.1' 4 32 4


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
    
	if [ "$3" ]; then   data=$3     # datasetting ['multi', 'multi2', 'nyt', 'onto', 'onto_multi']
    else                data='multi';   fi    

    if [ "$4" ]; then   training=$4
    else                training='base';    fi

    if [[ "$training" == *"base"* ]]; then
        echo "!!!!!!!!!!!! Base method !!!!!!!!!!!!"
        if [ "$5" ]; then	
            serial_dir="${task}_${data}_base"
            config="${cur_dir}/configs_da/${task}/${data}/${training}.json"    
            # 4 8 16
            overrides="{'trainer.num_gradient_accumulation_steps':2"
            lab_b=$(($5/2))
            serial_dir+="_l${5}"
            overrides+=",'data_loader.batch_size':${lab_b}"
        fi

    else
        serial_dir="${task}_${data}_s"
        config="${cur_dir}/configs_da/${task}/${data}/nodistil.json"   

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

        for method in $training
        do  
            if [ "$method" == 'nodistil' ]; then
                    serial_dir+="_nodistil"
                    break
            fi
            overrides+=",${task_training[${method}]}"
            serial_dir+="_${method}"
        done

        if [ "$7" ]; then
            # 0.1 0.3 0.5 0.7 0.9
            alpha=$7
            overrides+=",'model.inference_module.task_label_smoothing.alpha':${alpha}"
            serial_dir+="_alpha${alpha}"            
        fi   

        serial_unwei=""
        if [ "$8" ]; then
            # 0.1 0.25 0.5
            unlabwei=$8
            overrides+=",'model.loss_fn.loss_weights.unlabeled':${unlabwei}"
            serial_unwei+="_unwei${unlabwei}"
            if [ "$training" ]; then
                overrides+=",'model.inference_module.loss_fn.loss_weights.unlabeled':[${unlabwei},${unlabwei}]"
                serial_unwei+="_selfwei${unlabwei}"
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
        
        if [ "$9" ]; then
            # 1 2
            accum=$9
            overrides+=",'trainer.num_gradient_accumulation_steps':${accum}"
            accum=$(($accum/2))
        else
            accum=1
            overrides+=",'trainer.num_gradient_accumulation_steps':2"
        fi

        if [ "$7" ]; then	
            # 4 8 16
            lab_b=$(($7*$accum))
            serial_dir+="_l${lab_b}"
            overrides+=",'data_loader.scheduler.batch_size.labeled':${lab_b}"
        fi
    
	    if [ "$8" ]; then    
            # 4 8 16 32 64
            unlab_b=$(($8*$accum))
            serial_dir+="_u${unlab_b}"
            overrides+=",'data_loader.scheduler.batch_size.unlabeled':${unlab_b}"
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
