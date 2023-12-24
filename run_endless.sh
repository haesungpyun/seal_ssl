#!/bin/bash
#!/usr/bin/env python

#SBATCH --job-name=srl
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=128000MB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%j.out

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate seal_ad

if [ "$1" != '' ]; then
    serial_dir=$1
    config="${SLURM_SUBMIT_DIR}/${serial_dir}/config.json"
    override="'trainer.cuda_device':0"
    option='r'  
    
    if [[ "$1" == *"multi"* ]]; then   
        data='multi'
    else
        data='nyt'    # ['multi', 'nyt', 'onto', 'onto_multi']
    fi
else
    option='f'
    if [ "$2" ]; then   training=$2
    # ['base', 'nodistil', 'pre_nodistil', 'distil', 'pre_distil', 'thres', 'pre_thres', 'filterunlab', 'pre_filterunlab']
    else                training='base';   fi
    if [ "$3" ]; then   task=$3
    else                task='mlc';   fi
    if [ "$4" ]; then   data=$4
    else                data='bibtex';   fi    # ['multi', 'nyt', 'onto', 'onto_multi', 'bgc', 'bibtex']
    if [ "$5" ]; then   unlabwei=$5
    else                unlabwei='_un10';   fi # ["_un10", "_un25", "_un50"]
    
    if [[ $training == 'pre'* ]]; then
        override="'trainer.pre_training':4"
        suffix=${training#*'pre_'}
        echo $suffix
        if [[ $data == *'onto'* || $training == *'base'* ]]; then 
            unlabwei=''
            config="${SLURM_SUBMIT_DIR}/configs_da/${task}_${data}_${suffix}.json"
        else
            config="${SLURM_SUBMIT_DIR}/configs_da/${task}_${data}_${suffix}${unlabwei}.json"
        fi
    else
        override="'trainer.pre_training':0"
        echo $training
        if [[ $data == *'onto'* || $training == *'base'* ]]; then 
            unlabwei=''
            config="${SLURM_SUBMIT_DIR}/configs_da/${task}_${data}_${training}.json"
        else
            config="${SLURM_SUBMIT_DIR}/configs_da/${task}_${data}_${training}${unlabwei}.json"
        fi
    fi
    serial_dir="${task}_${data}_${training}${unlabwei}_ogdistil_multi_gpu"

fi

unlab="'data_loader.scheduler.batch_size.unlabeled':16"
lab="'data_loader.scheduler.batch_size.labeled':2"
accum="'trainer.num_gradient_accumulation_steps':1"
og_distil="'model.og_distil':true"
overrides="{${override},${unlab},${lab},${accum},${og_distil}}"
# overrides="{${override},${og_distil}}"
overrides="{${override}}"
echo $config
echo 'directory:' $serial_dir
echo $option
echo $overrides
echo "job_id_${SLURM_JOB_ID}"

if [ $option == 'f' ]; then
    output=$(timeout 710m srun allennlp train $config \
                                            -s $serial_dir \
                                            --include-package seal \
                                            -$option \
                                            -o $overrides
                                        )
else
    output=$(timeout 710m srun allennlp train $config \
                                            -s $serial_dir \
                                            --include-package seal \
                                            -$option \
                                        )
fi
# ./srl_multi_nodistil_un10_unlab16/config.json
# allennlp train ./configs_da/srl_multi_distil_un10.json -s tmp3 --include-package seal -f -o $overrides
# srun allennlp train ./srl_multi_nodistil_un10_unlab8/config.json -s srl_multi_from_scratch_un10_unlab8 --include-package seal -r --overrides '{"trainer.pre_training":0, "trainer.num_gradient_accumulation_steps":2}' \
# {"trainer":{"pre_training":4}}, {"data_loader":{"scheduler":{"batch_size":{"unlabeled":8}}}}
# "data_loader.scheduler.batch_size.unlabeled":8

state=$?
echo $state
if [[ $state -eq 124 ]]; then # if [[ $state -eq 124 || $state -eq 1]] loop forever
    if [[ $serial_dir == *'_wo_pre' ]]; then
        exit
    fi
    new=$(sbatch run_endless.sh $serial_dir)
    echo $new
elif [ $state -eq 0 ]; then
    new=$(sbatch run_eval_da.sh $serial_dir $data)
    echo $new
fi