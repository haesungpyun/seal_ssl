#!/bin/bash

#SBATCH --job-name=eval-srl
#SBATCH -p amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=128000MB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=./logs/%j-eval.out
#SBATCH --comment etc

export OMP_NUM_THREADS=1      

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 conda/pytorch_1.12.0

source ~/.bashrc
cds
cd seal_ex
eval "$(conda shell.bash hook)"
conda activate seal_ad
conda env list

if [ "$1" ]; then
    serialization_dir=$1
else
    serialization_dir="/srl_multi_pre_un10"
fi

if [[ "$1" == *"multi2"* ]]; then   
    data_file='multi'
    src="./data/conll-2012/v12/data/test/data/english/annotations/@(mz|wb)"
    trg="./data/conll-2012/v12/data/test/data/english/annotations/@(bc|bn|nw)"
    tot_unseen="./data/conll-2012/v12/data/test/data/english/annotations/@(tc|pt)"
    declare -a path=("${src}" "${trg}" "${tot_unseen}")
    declare -a name=("src" "trg" "tot-unseen")

elif [[ "$1" == *"multi"* ]]; then   
    data_file='multi'
    src="./data/conll-2012/v12/data/test/data/english/annotations/@(bc|bn)"
    trg="./data/conll-2012/v12/data/test/data/english/annotations/@(tc|nw|pt)"
    tot_unseen="./data/conll-2012/v12/data/test/data/english/annotations/@(mz|wb)"
    declare -a path=("${src}" "${trg}" "${tot_unseen}")
    declare -a name=("src" "trg" "tot-unseen")
    
elif [[ "$1" == *"nyt"* ]]; then 
    data_file='nyt'
    src="./data/conll-2012/v12/data/test/data/english/annotations/@(pt|wb|mz)"
    trg="./data/conll-2012/v12/data/test/data/english/annotations/@(nw|bn)"
    tot_unseen="./data/conll-2012/v12/data/test/data/english/annotations/@(tc|wb)"
    declare -a path=("${src}" "${trg}" "${tot_unseen}")
    declare -a name=("src" "trg" "tot-unseen")

fi
echo ""
echo "!!!!!"
echo "${path[*]}"
echo "${name[*]}"
echo "!!!!!"
echo ""
archive_file="${serialization_dir}/model.tar.gz"

echo $archive_file
echo $serialization_dir

# model_state=$(cd $serialization_dir && find ~+ -type f -name "model_state_*" && cd ../)
# model_state=${model_state#${serialization_dir}*}
# model_state=${model_state#*/}
model_state="best.th"
weight_file="${serialization_dir}/${model_state}"

override="'trainer.cuda_device':0"

for idx in "${!path[@]}"; do
    
    # if [[ "$data_file" == *"mlc"* ]]; then
    #     valid_reader="'validation_dataset_reader.type':'${test_data}'"
    #     overrides="{${override},${valid_reader}}"
    #     input_data="$data_file/$test_data/*.jsonl"
    # else
    echo ''
    echo ''
    n="${name[$idx]}"
    echo $n
    overrides="{${override}}"    
    input_data="${path[$idx]}"
    
    output_file="test_metric_${n}.json"
    output_file="${serialization_dir}/${output_file}"
    echo $input_data
    echo $output_file
    # python mytest.py $serialization_dir \
    #                 $archive_file \
    #                 "$data_file/$test_data" \
    #                 $output_file \
    #                 $weight_file \
    #                 0    
    
    srun allennlp evaluate $archive_file $input_data \
    --output-file $output_file \
    --weights-file $weight_file \
    --include-package seal  \
    -o $overrides
    
done
# exit
echo "$(pwd)/$serialization_dir"

python make_summary.py "$(pwd)/$serialization_dir" $data_file
